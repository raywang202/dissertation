# same as modelsimjit_approx.py but with switching costs in EDUCATION
# at both the zero and last-choice level
# reduces the complexity of the wage shock integration for later
# time observations

import numpy as np
import numba
import pandas as pd
import itertools
import scipy.stats

from numba import jitclass          # import the decorator
from numba import int64, float64    # import the types
import simplexmap

labor_spec = [
    ('horizon', int64),
    ('final_horizon',int64),
    ('gamma_p', float64),
    ('wage_coeffs', float64[:,:]),
    ('N',int64),
    ('N_later',int64),
    ('STEM',int64),
    ('grade',float64),
    ('beta',float64),
    ('wage_shocks',float64[:,:]),
    ('wage_shocks_later',float64[:,:]),
    ('sectors',int64),
    ('flows',float64[:]),
    ('base_wage',float64[:]),
    ('EmaxList',float64[:,:,:]),
    ('choose',int64[:,:]),
    ('quality',int64),
    ('time_zero_flows_penalized',float64[:]),
    ('switch_costs',float64[:]),
    ('zero_exp_penalty',float64[:]),
    ('flows_penalized',float64[:]),
    ('year_four_intercept',float64),
]


# Generates labor Emax functions
# Labor Emax functions are calculated by simulation over wage shocks,

@jitclass(labor_spec)
class EmaxLaborFunctionsJIT:

    def __init__(self,horizon,final_horizon,
        gamma_p,beta,wage_coeffs,STEM,
        grade,flows_penalized,wage_shocks,choose,quality,
        time_zero_flows_penalized,switch_costs,zero_exp_penalty,
        wage_shocks_later):
        """
        Initializes a labor market model with the following variables:
        yearly_horizon: time horizon in labor market for yearly solutions

        gamma_p: individual preference for income rel. to pref. shocks
        beta: intertemporal discount factor
        wage_coeffs:
            3x11 array, for skilled 1-3, or 1x11 for unskilled
            [C,M,G_N,G_S,x1,x2,x3,own sq, own 1st year, quality, age]
        STEM: 1 if STEM, 0 otherwise
        grade: terminal GPA, in units of raw GPA, i.e. 0 - 4.0
        flows: N vector of flow utilities for each sector. leisure normalized 0
        wageshocks: distribution of wageshocks to use (precalculated), e.g.
            zscores=scipy.stats.norm.ppf(np.array(range(1,normReps+1))/
            (normReps+1))
            base_draws=np.matrix.transpose(np.matrix(list(
            itertools.product(zscores,repeat=(sectors)))))
            lmat=np.linalg.cholesky(skilled_wage_covar)
            wageshocks=np.array(np.transpose(np.matmul(lmat,base_draws)))
        choose: Pascal's triangle. Run
            choose=simplexmap.pascal(self.horizon,self.sectors)
        quality: 1 or 0 for if the school is high quality
        time_zero_flows: specific skilled labor flow utilities for the first
            year after college (time = 5 for graduates, or 0 for every other).
            Must have same dimension as flows, namely N
        switch_costs: specific skilled labor flow utility penalty for switching
            into the sector if the prior major was NOT. values are POSITIVE
            for switching costs
        wage_shocks_sequence: a N x sector x (final_horizon-yearly_horizon)
            sequence of wage shocks, which the agent observes prior to making
            their decision at the terminal period

        """
        self.horizon=horizon
        self.final_horizon=final_horizon
        self.gamma_p=gamma_p
        self.wage_coeffs=wage_coeffs
        self.STEM=STEM
        self.sectors=wage_coeffs.shape[0]
        self.N=wage_shocks.shape[0]
        self.N_later = wage_shocks_later.shape[0]
        self.grade=grade
        self.beta=beta
        self.wage_shocks=wage_shocks
        self.choose=choose
        # EmaxList arguments: 1st is time, second is a scalar representing
        # the accumulated exp, and 3rd is the prior choice
        self.EmaxList=np.zeros((self.final_horizon+1,
            self.choose[self.final_horizon+self.sectors,self.sectors],
            self.sectors+1),
        dtype=np.float64)
        self.flows_penalized=np.zeros(self.sectors+1,dtype=np.float64)
        self.switch_costs = np.zeros(self.sectors+1,dtype=np.float64)
        self.switch_costs[0:self.sectors]=switch_costs
        self.flows_penalized[0:self.sectors]=flows_penalized
        self.zero_exp_penalty = np.zeros(self.sectors+1)
        self.zero_exp_penalty[0:self.sectors] = zero_exp_penalty
        self.quality = quality
        self.wage_shocks_later = wage_shocks_later
        # experience-agnostic log-wage payout
        # i.e. constant, M, G_N, G_S 
        self.base_wage=np.zeros(self.sectors,dtype=np.float64)
        for i in range(self.sectors):
            if self.STEM==0:
                mg=self.wage_coeffs[i][2]*self.grade
            else:
                mg=self.wage_coeffs[i][1]+self.wage_coeffs[i][3]*self.grade
            self.base_wage[i]=(self.wage_coeffs[i][0]+mg+self.wage_coeffs[i][9]*
                self.quality)
        self.time_zero_flows_penalized=np.zeros(self.sectors+1)
        self.time_zero_flows_penalized[0:self.sectors]=time_zero_flows_penalized
        self.zero_exp_penalty = np.zeros(self.sectors+1)
        self.zero_exp_penalty[0:self.sectors] = zero_exp_penalty


    # each row of out is just the experience tuple at years # of experience
    def expCombos(self,years):
        num_combos=self.choose[years+self.sectors][self.sectors]
        out=np.zeros((num_combos,self.sectors+1),dtype=np.int64)
        for x in range(num_combos):
            out[x]=simplexmap.array_to_combo(x,years,self.sectors,self.choose)
        return out

  
    def EmaxLabor(self,time):
        """
        Determines the Emax function at a given time, assuming that all
        later value functions have already been calculated

        """
        # add 1 sector for HP
        exp_allocs=self.expCombos(time)
        len_allocs=exp_allocs.shape[0]
        if time>self.horizon:
            N = self.N_later
            wage_shocks = self.wage_shocks_later
        else:
            N = self.N
            wage_shocks = self.wage_shocks
        # terminal condition, so no recursion
        if time==self.final_horizon:
            # exponential MVN wage shocks + column of zeros for HP
            # generate logwages net wage shock for experience profile i
            for i in range(len_allocs):

                flows = self.flows_penalized.copy()
                for j in range(self.sectors):
                    if exp_allocs[i][j]==0:
                        flows[j]=flows[j]+self.zero_exp_penalty[j]

                wages=np.hstack((np.exp(wage_shocks+
                    self.ELogWage(exp_allocs[i])),np.zeros((N,1))))
                net_flow=flows+wages*self.gamma_p
                # incorporates the switching costs (or non-payment of penalty)
                for last_choice in range(self.sectors+1):
                    net_flow_final = net_flow.copy()
                    net_flow_final[:,last_choice] = (
                        net_flow_final[:,last_choice]+
                        self.switch_costs[last_choice])
                    outcome=np.zeros(N,dtype=np.float64)
                    for j in range(N):
                        maxvalue=np.max(net_flow_final[j])
                        outcome[j]=maxvalue+np.log(np.sum(np.exp(
                            net_flow_final[j]-maxvalue)))
                    self.EmaxList[self.final_horizon,i,last_choice]=np.mean(outcome)+0.57722            


        else:
            for i in range(len_allocs):
                flows = self.flows_penalized.copy()
                for j in range(self.sectors):
                    if exp_allocs[i][j]==0:
                        flows[j]=flows[j]+self.zero_exp_penalty[j]

                wages=np.hstack((np.exp(wage_shocks+
                    self.ELogWage(exp_allocs[i])),np.zeros((N,1))))
                current_flow=flows+wages*self.gamma_p
                outcome=np.zeros(N,dtype=np.float64)

                # this differs in non-terminal case
                # generate all experience incrementations and convert to tuple
                next_exp=np.zeros((self.sectors+1,self.sectors+1),
                    dtype=np.int64)
                disc_next_payoff=np.zeros(self.sectors+1,dtype=np.float64)

                for j in range(self.sectors+1):
                    next_exp[j][j]=1
                next_exp=next_exp+exp_allocs[i]

                # next_payoff is NOT discounted
                for j in range(self.sectors+1):
                    disc_next_payoff[j]=self.beta*self.EmaxList[time+1,
                    simplexmap.combo_to_array(time+1,next_exp[j],self.choose),
                    j]

                # recursively call the next period's Emax values
                # and create an N x Sector matrix
                net_payoff=current_flow+disc_next_payoff
                for last_choice in range(self.sectors+1):
                    net_payoff_final=net_payoff.copy()
                    net_payoff_final[:,last_choice]=(
                        net_payoff_final[:,last_choice]+
                        self.switch_costs[last_choice])
                    for j in range(N):
                        maxvalue=np.max(net_payoff_final[j])
                        outcome[j]=maxvalue+np.log(np.sum(np.exp(
                            net_payoff_final[j]-maxvalue)))
                    self.EmaxList[time,i,last_choice]=np.mean(outcome)+0.57722

    def solveLabor(self):
        i=self.final_horizon
        while i>=1:
            self.EmaxLabor(i)
            i=i-1

        # time = 0 step
        time=0
        exp_allocs=self.expCombos(time)
        len_allocs=exp_allocs.shape[0]
        for i in range(len_allocs):
            wages=np.hstack((np.exp(self.wage_shocks+
                self.ELogWage(exp_allocs[i])),np.zeros((self.N,1))))

            current_flow=(self.time_zero_flows_penalized+self.zero_exp_penalty+
                wages*self.gamma_p)
            outcome=np.zeros(self.N,dtype=np.float64)

            # this differs in non-terminal case
            # generate all experience incrementations and convert to tuple
            next_exp=np.zeros((self.sectors+1,self.sectors+1),
                dtype=np.int64)
            disc_next_payoff=np.zeros(self.sectors+1,dtype=np.float64)

            for j in range(self.sectors+1):
                next_exp[j][j]=1
            next_exp=next_exp+exp_allocs[i]

            # next_payoff is NOT discounted
            for j in range(self.sectors+1):
                disc_next_payoff[j]=self.beta*self.EmaxList[time+1,
                simplexmap.combo_to_array(time+1,next_exp[j],self.choose),j]

            # recursively call the next period's Emax values
            # and create an N x Sector matrix
            net_payoff=current_flow+disc_next_payoff
            for j in range(self.N):
                maxvalue=np.max(net_payoff[j])
                outcome[j]=maxvalue+np.log(np.sum(np.exp(
                    net_payoff[j]-maxvalue)))
            sol = np.mean(outcome)+0.57722

            for j in range(self.sectors+1):
                self.EmaxList[time,i,j]=sol

    def ELogWage(self,exp):
        """
        Given experience vector over the various sectors INCLUDING HP, output
        the expected log-wages, given already calculated wages/grades
        Here it is experience terms, including hp
        """
        out=np.zeros(self.sectors,dtype=np.float64)
        total_exp = np.sum(exp)

        for i in range(self.sectors):
            at_least_one_exp = 1 if exp[i]>0 else 0

            # no need to cap wages
            if -2*exp[i]*self.wage_coeffs[i][7]<=self.wage_coeffs[i][i+4]:
                out[i]=(self.base_wage[i]+exp[0]*self.wage_coeffs[i][4]+
                    exp[1]*self.wage_coeffs[i][5]+exp[2]*self.wage_coeffs[i][6]+
                    exp[i]**2*self.wage_coeffs[i][7]+
                    total_exp*self.wage_coeffs[i][10]+
                    at_least_one_exp*self.wage_coeffs[i][8])

            # need to cap quadratic loss in earnings
            else:
                out[i]=(self.base_wage[i]+exp[0]*self.wage_coeffs[i][4]+
                    exp[1]*self.wage_coeffs[i][5]+exp[2]*self.wage_coeffs[i][6]-
                    exp[i]*self.wage_coeffs[i][i+4]-
                    self.wage_coeffs[i][i+4]**2/(4*self.wage_coeffs[i][7])+
                    total_exp*self.wage_coeffs[i][10]+
                    at_least_one_exp*self.wage_coeffs[i][8])
        return out


    # sector num is indexed at zero, i.e. sector 1 has value 0
    def ElogWageOneSector(self,sector_num_idx_zero,exp):
        i=sector_num_idx_zero
        total_exp = np.sum(exp)

        at_least_one_exp = 1 if exp[i]>0 else 0
        if -2*exp[i]*self.wage_coeffs[i][7]<=self.wage_coeffs[i][i+4]:

            out=(self.base_wage[i]+
                exp[i]*self.wage_coeffs[i][4]+
                exp[i]**2*self.wage_coeffs[i][7]+
                total_exp*self.wage_coeffs[i][10]+
                at_least_one_exp*self.wage_coeffs[i][8])
        else:
            out=(self.base_wage[i]-
                self.wage_coeffs[i][4]**2/(4*self.wage_coeffs[i][7])+
                total_exp*self.wage_coeffs[i][10]+
                at_least_one_exp*self.wage_coeffs[i][8])
        return out


# unskilled variant of EmaxLaborFunctionsJIT
@jitclass(labor_spec)
class EmaxLaborFunctionsJITUnskilled:

    def __init__(self,horizon,gamma_p,beta,wage_coeffs,STEM,
        grade,flows_penalized,wage_shocks,choose,switch_costs,zero_exp_penalty,
        quality=0):
        """
        Initializes a labor market model with the following variables:
        horizon: time horizon in labor market
        gamma_p: individual preference for income rel. to pref. shocks
        beta: intertemporal discount factor
        wage_coeffs: 1x11 vector for unskilled.
            [C,M,G_N,G_S,x1,x2,x3,own sq, own 1st year, quality, age]
            x1 refers to own, in this case, so we have 0, 4, 7, 8 filled
        STEM: 1 if STEM, 0 otherwise
        grade: terminal GPA, in units of raw GPA, i.e. 0 - 4.0
        flows: N vector of flow utilities for each sector. leisure normalized 0
        wageshocks: distribution of wageshocks to use (precalculated), e.g.
            zscores=scipy.stats.norm.ppf(np.array(range(1,normReps+1))/
            (normReps+1))
            base_draws=np.matrix.transpose(np.matrix(list(
            itertools.product(zscores,repeat=(sectors)))))
            lmat=np.linalg.cholesky(skilled_wage_covar)
            wageshocks=np.array(np.transpose(np.matmul(lmat,base_draws)))
        choose: Pascal's triangle. Run
                choose=simplexmap.pascal(self.horizon,self.sectors)

        """
        self.horizon=horizon
        self.gamma_p=gamma_p
        self.wage_coeffs=wage_coeffs
        self.STEM=STEM
        self.sectors=wage_coeffs.shape[0]
        self.N=wage_shocks.shape[0]
        self.grade=grade
        self.beta=beta
        self.wage_shocks=wage_shocks
        self.choose=choose
        self.EmaxList=np.zeros((
            self.horizon+1,self.choose[self.horizon+self.sectors,self.sectors],
                self.sectors+1),dtype=np.float64)
        self.switch_costs=np.zeros(self.sectors+1,dtype=np.float64)
        self.switch_costs[0:self.sectors]=switch_costs
        self.flows_penalized = np.zeros(self.sectors+1)
        self.flows_penalized[0:self.sectors]=flows_penalized
        # experience-agnostic log-wage payout
        # i.e. constant, M, G, MxG
        self.base_wage=np.zeros(self.sectors,dtype=np.float64)
        for i in range(self.sectors):
            self.base_wage[i]=(self.wage_coeffs[i][0])
        self.zero_exp_penalty = zero_exp_penalty


    # each row of out is just the experience tuple at years # of experience
    def expCombos(self,years):
        num_combos=self.choose[years+self.sectors][self.sectors]
        out=np.zeros((num_combos,self.sectors+1),dtype=np.int64)
        for x in range(num_combos):
            out[x]=simplexmap.array_to_combo(x,years,self.sectors,self.choose)
        return out

    def EmaxLabor(self,time):
        """
        Determines the Emax function at a given time, assuming that all
        later value functions have already been calculated
        EmaxValues is recursively generated and is a dict of form
        {[time,(experience)] : Emax}
        The last sector is always the leisure sector
        """
        # add 1 sector for HP
        exp_allocs=self.expCombos(time)
        len_allocs=exp_allocs.shape[0]
        # terminal condition, so no recursion
        if time==self.horizon:
            # exponential MVN wage shocks + column of zeros for HP
            # generate logwages net wage shock for experience profile i

            #####
            for i in range(len_allocs): ########
                wages=np.hstack((np.exp(self.wage_shocks+
                    self.ELogWage(exp_allocs[i])),np.zeros((self.N,1))))
                net_flow=self.flows_penalized+wages*self.gamma_p

                # zero exp condition (0 = unskilled)
                if exp_allocs[i][0]==0:
                    net_flow[0]=net_flow[0] + self.zero_exp_penalty[0]

                outcome=np.zeros(self.N,dtype=np.float64)

                for last_choice in range(self.sectors+1):
                    net_flow_final = net_flow.copy()

                    net_flow_final[:,last_choice] = (
                        net_flow_final[:,last_choice]+
                        self.switch_costs[last_choice])
                    for j in range(self.N):
                        maxvalue=np.max(net_flow_final[j])
                        outcome[j]=maxvalue+np.log(np.sum(np.exp(
                            net_flow_final[j]-maxvalue)))
                    self.EmaxList[time,i,last_choice]=np.mean(outcome)+0.57722


        else:
            for i in range(len_allocs):
                wages=np.hstack((np.exp(self.wage_shocks+
                    self.ELogWage(exp_allocs[i])),np.zeros((self.N,1))))
                current_flow=self.flows_penalized+wages*self.gamma_p

                # zero exp condition (0 = unskilled)
                if exp_allocs[i][0]==0:
                    current_flow[0]=current_flow[0] + self.zero_exp_penalty[0]

                outcome=np.zeros(self.N,dtype=np.float64)

                # this differs in non-terminal case
                # generate all experience incrementations and convert to tuple
                next_exp=np.zeros((self.sectors+1,self.sectors+1),
                    dtype=np.int64)
                disc_next_payoff=np.zeros(self.sectors+1,dtype=np.float64)

                for j in range(self.sectors+1):
                    next_exp[j][j]=1
                next_exp=next_exp+exp_allocs[i]

                # next_payoff is NOT discounted
                for j in range(self.sectors+1):
                    disc_next_payoff[j]=self.beta*self.EmaxList[time+1,
                    simplexmap.combo_to_array(time+1,next_exp[j],self.choose),
                    j]

                # recursively call the next period's Emax values
                # and create an N x Sector matrix
                net_payoff=current_flow+disc_next_payoff
                for last_choice in range(self.sectors+1):
                    net_payoff_final=net_payoff.copy()
                    net_payoff_final[:,last_choice]=(
                        net_payoff_final[:,last_choice]+
                        self.switch_costs[last_choice])
                    for j in range(self.N):
                        maxvalue=np.max(net_payoff_final[j])
                        outcome[j]=maxvalue+np.log(np.sum(np.exp(
                            net_payoff_final[j]-maxvalue)))
                    self.EmaxList[time,i,last_choice]=(np.mean(outcome)+0.57722)

    def solveLabor(self):
        i=self.horizon
        while i>=0:
            self.EmaxLabor(i)
            i=i-1

    def ELogWage(self,exp):
        """
        Given experience vector over the various sectors INCLUDING HP, output
        the expected log-wages, given already calculated wages/grades
        """
        out=np.zeros(self.sectors,dtype=np.float64)
        total_exp = np.sum(exp)

        for i in range(self.sectors):
            at_least_one_exp = 1 if exp[i]>0 else 0
            if -2*exp[i]*self.wage_coeffs[i][7]<=self.wage_coeffs[i][i+4]:

                out[i]=(self.base_wage[i]+
                    exp[i]*self.wage_coeffs[i][4]+
                    exp[i]**2*self.wage_coeffs[i][7]+
                    total_exp*self.wage_coeffs[i][10]+
                    at_least_one_exp*self.wage_coeffs[i][8])
            else:
                out[i]=(self.base_wage[i]-
                    self.wage_coeffs[i][4]**2/(4*self.wage_coeffs[i][7])+
                    total_exp*self.wage_coeffs[i][10]+
                    at_least_one_exp*self.wage_coeffs[i][8])
        return out


ed_spec = [
    ('dropout_payouts', float64[:,:]),
    ('STEM_payouts', float64[:]),
    ('nonSTEM_payouts', float64[:]),
    ('grade_params', float64[:,:]),
    ('gamma_p', float64),
    ('beta',float64),
    ('tuition',float64[:]),
    ('exog_chars',float64[:]),
    ('flows',float64[:]),
    ('ability',float64[:]),
    ('unskilled_coeffs',float64[:]),
    ('unskilled_meanvar',float64[:]),
    ('EmaxEducationValues',float64[:,:,:]),
    ('norm_quantiles',float64[:]),
    ('num_quantiles',int64),
    ('choose',int64[:,:]),
    ('gpa_vals',int64[:]),
    ('gpa_num_blocks',int64),
    ('gpa_levels',int64[:]),
    ('unskilled_wages',float64[:]),
    ('STEM_cond_val_first',float64),
    ('nonSTEM_cond_val_first',float64),
    ('sigma_S1',float64),
    ('sigma_N1',float64),
    ('sigma_S2',float64),
    ('sigma_N2',float64),
    ('year_four_wages',float64[:]),
    ('switching_costs',float64[:]),
    ('grad_payoff',float64),
    ('year_four_flow_penalized',float64)
]


@jitclass(ed_spec)
class EmaxEducationJIT:

    def __init__(self, dropout_payouts, STEM_payouts, nonSTEM_payouts,
        grade_params, gamma_p, beta, flows_penalized, tuition, exog_chars,
        ability, unskilled_meanvar, norm_quantiles,year_four_intercept,
        year_four_flow_penalized, switching_costs,grad_payoff):
        """
        dropout_payouts is a 4x2 array with the payouts. 1st entry in second
            column is 0 if accumulated exp is 0, 1 if not
        STEM_payouts is a 21-length array with each row the ex-ante payout
            at zero experience for STEM grads (ascending from 200 to 400,
            every 10)
        nonSTEM_payouts is a 21-length array with each row the ex-ante payout
            at zero experience for STEM grads

        grade_params is a 9x2 vector by major:
            [SAT_M, SAT_V, hs_GPA,lambda_1,lambda_2,lambda_3,lambda_4,
            sigma^2_m1,sigma^2_m3]
            STEM comes first, and sigma^2_m1 and m3 refer to first 2 years or
            last 2
        ability = [STEM_ability,nonSTEM_ability]
        gamma_p is preference for money (tuition)
        beta is discount factor
        flows_penalized is [STEM penalized, nonSTEM penalized,
            unskilled_penalized]
            STEM/nonSTEM penalized is per-period flow minus switching cost,
            unskilled is flow minus switching cost. ALSO INCLUDES zero exp
        tuition is the yearly tuition, a 4x1 vector
        exog_chars are a vector [SAT_M,SAT_V, hs_GPA]. SAT are in raw units,
            hs_GPA is in units of 100 (e.g. a perfect GPA is 400)
        ability is [A_STEM,A_nonSTEM]
        unskilled_meanvar is a [mean,var] for log-normal unskilled wages at zero
            experience 
        norm_quantiles is the array of normal quantiles used to integrate
            over grade shocks as well as wage shocks
        switching_costs is the cost of switching INTO STEM or non-STEM, length 
            2 array
        """

        # Discretize college value function over 0.05 GPA points, by major
        # while in college
        (self.dropout_payouts, self.STEM_payouts, self.nonSTEM_payouts,
            self.grade_params, self.gamma_p, self.beta, self.flows,
            self.tuition, self.exog_chars, self.ability, self.unskilled_coeffs,
            self.norm_quantiles, self.switching_costs, self.grad_payoff) = (
            dropout_payouts, STEM_payouts,nonSTEM_payouts, grade_params,
            gamma_p, beta, flows_penalized,tuition, exog_chars, ability,
            unskilled_meanvar, norm_quantiles, switching_costs, grad_payoff)

        # GPA granularity
        gpa_interval=0.1
        gpa_precision=int((400-200)/100/gpa_interval)+1
        self.gpa_vals=np.linspace(200,400,gpa_precision).astype(np.int64)
        self.gpa_levels=np.linspace(0,400,81).astype(np.int64)
        self.gpa_num_blocks=len(self.gpa_levels)
        self.EmaxEducationValues=np.zeros((6,self.gpa_num_blocks,2),
            dtype=np.float64)
        self.num_quantiles=len(self.norm_quantiles)

        # pre-calculate all unskilled wages
        self.unskilled_wages=np.exp(self.unskilled_coeffs[0]+
                self.norm_quantiles*np.sqrt(self.unskilled_coeffs[1]))

        self.year_four_wages =np.exp(year_four_intercept+
                self.norm_quantiles*np.sqrt(self.unskilled_coeffs[1]))
        self.year_four_flow_penalized = year_four_flow_penalized
        self.sigma_S1=100*self.grade_params[0][-2]**0.5
        self.sigma_S2 = 100*self.grade_params[0][-1]**0.5
        self.sigma_N1=100*self.grade_params[1][-2]**0.5
        self.sigma_N2=100*self.grade_params[1][-1]**0.5

    # expected grade. Major is 1 for STEM, 0 for non-STEM 
    def Egrade(self, major, year):
        if major==1:
            major_params=self.grade_params[0]
            maj_ability=self.ability[0]
        else:
            major_params=self.grade_params[1]
            maj_ability=self.ability[1]
        year_param=major_params[year+2]
        return (major_params[0]*self.exog_chars[0]+
            major_params[1]*self.exog_chars[1]+
            major_params[2]*self.exog_chars[2]+year_param+maj_ability)


    def __round_to(self, n, precision):
        correction = 0.5 if n >= 0 else -0.5
        return int( n/precision+correction ) * precision

    def __round_to_5(self,n):
        return self.__round_to(n, 5)

    def __round_to_10(self,n):
        return self.__round_to(n,10)

    # converts intermediate GPA (int) to index
    # hard coded to level of precision 0.05
    def __gpa_to_index(self,gpa):
        return int(gpa/5)

    # converts terminal GPA to index
    def __tgpa_to_index(self,gpa):
        return int((gpa-200)/10)

    def solve(self):
        self.Emax4()
        self.Emax3()
        self.Emax2()
        self.Emax1()
        
    # generate ex-ante value function at t=4
    def Emax4(self):
        # 4 possible outcomes: STEM, non-STEM, leisure, unskilled labor
        # STEM eligible

        # hard coded atm
        # iterate over the major/GPA going in

        # STEM case
        for gpa_idx in range(self.gpa_num_blocks):
            gpa=self.gpa_levels[gpa_idx]
            # generate next grade, rounded to nearest 5, and top/bottom capped
            # at 0, 400 GPA
            sem_grades=self.norm_quantiles*self.sigma_S2+100*self.Egrade(1,4)
            for x in range(self.num_quantiles):
                if sem_grades[x]>400:
                    sem_grades[x]=400
                elif sem_grades[x]<0:
                    sem_grades[x]=0
            next_grades=np.zeros(self.num_quantiles,dtype=np.int64)
            for x in range(len(sem_grades)):
                next_grades[x]=self.__round_to_5((sem_grades[x]+3*gpa)/4)
            # link grade to labor market payoff
            graduation_payoff=np.zeros(self.num_quantiles,dtype=np.float64)
            for x in range(self.num_quantiles):
                # successfully graduate
                if next_grades[x]>=200:
                    graduation_payoff[x]=self.STEM_payouts[
                    self.__tgpa_to_index(next_grades[x])] + self.grad_payoff
                # failed out
                else:
                    graduation_payoff[x]=self.dropout_payouts[3,0]

            flow_school=(self.beta*np.mean(graduation_payoff)-self.gamma_p*
            self.tuition[3]+self.flows[0]+self.switching_costs[0])
            flow_hp=self.beta*self.dropout_payouts[3,0]

            flow_unskilled=(self.gamma_p*self.year_four_wages+
                self.year_four_flow_penalized+ self.beta*self.dropout_payouts[3,1])

            # normalization sleight of hand
            self.EmaxEducationValues[4][gpa_idx]=(flow_school+
                np.mean(np.log(np.exp(flow_hp-flow_school)+
                1+np.exp(flow_unskilled-flow_school)))+0.57722)


        # # non-STEM case
        for gpa_idx in range(self.gpa_num_blocks):
            gpa=self.gpa_levels[gpa_idx]
            # generate next grade, rounded to nearest 5, and top/bottom capped
            # at 0, 400 GPA
            sem_grades=self.norm_quantiles*self.sigma_N2+100*self.Egrade(0,4)
            for x in range(self.num_quantiles):
                if sem_grades[x]>400:
                    sem_grades[x]=400
                elif sem_grades[x]<0:
                    sem_grades[x]=0
            next_grades=np.zeros(self.num_quantiles,dtype=np.int64)
            for x in range(self.num_quantiles):
                next_grades[x]=self.__round_to_5((sem_grades[x]+3*gpa)/4)
            # link grade to labor market payoff
            graduation_payoff=np.zeros(self.num_quantiles,dtype=np.float64)
            for x in range(self.num_quantiles):
                # successfully graduate
                if next_grades[x]>=200:
                    graduation_payoff[x]= self.nonSTEM_payouts[
                    self.__tgpa_to_index(next_grades[x])] + self.grad_payoff
                # failed out
                else:
                    graduation_payoff[x]=self.dropout_payouts[3,0]


            flow_school=(self.beta*np.mean(graduation_payoff)-self.gamma_p*
            self.tuition[3]+self.flows[1]+self.switching_costs[1])
            flow_hp=self.beta*self.dropout_payouts[3,0]
            flow_unskilled=(self.gamma_p*self.year_four_wages+
                self.year_four_flow_penalized+
                self.beta*self.dropout_payouts[3,1])
            # normalization sleight of hand
            self.EmaxEducationValues[5][gpa_idx]=(flow_school+
                np.mean(np.log(np.exp(flow_hp-flow_school)+
                1+np.exp(flow_unskilled-flow_school)))+0.57722)

    def Emax3(self):
        # 4 possible outcomes: STEM, non-STEM, leisure, unskilled labor
        # STEM eligible

        # hard coded atm
        # iterate over the major/GPA going in


        for gpa_idx in range(self.gpa_num_blocks):
            gpa=self.gpa_levels[gpa_idx]

            # generate next grade, rounded to nearest 5, and top/bottom capped
            # at 0, 400 GPA
            STEM_grades=self.norm_quantiles*self.sigma_S2+100*self.Egrade(1,3)
            for x in range(self.num_quantiles):
                if STEM_grades[x]>400:
                    STEM_grades[x]=400
                elif STEM_grades[x]<0:
                    STEM_grades[x]=0
            next_STEM_grades=np.zeros(len(STEM_grades),dtype=np.int64)

            # calculate next sem grades

            nonSTEM_grades=self.norm_quantiles*self.sigma_N2+100*self.Egrade(0,3)
            for x in range(self.num_quantiles):
                if nonSTEM_grades[x]>400:
                    nonSTEM_grades[x]=400
                elif nonSTEM_grades[x]<0:
                    nonSTEM_grades[x]=0
            next_nonSTEM_grades=np.zeros(self.num_quantiles,dtype=np.int64)
            for x in range(self.num_quantiles):
                next_STEM_grades[x]=self.__round_to_5((STEM_grades[x]+2*gpa)/3)
                next_nonSTEM_grades[x]=self.__round_to_5((
                    nonSTEM_grades[x]+2*gpa)/3)

            # link grade to next period payoff

            next_STEM_payoffs=np.zeros(self.num_quantiles,dtype=np.float64)
            next_nonSTEM_payoffs=np.zeros(self.num_quantiles,dtype=np.float64)

            for x in range(self.num_quantiles):
                next_STEM_payoffs[x]=self.EmaxEducationValues[4,
                self.__gpa_to_index(next_STEM_grades[x]),0]
                next_nonSTEM_payoffs[x]=self.EmaxEducationValues[5,
                self.__gpa_to_index(next_nonSTEM_grades[x]),1]


            flow_STEM=(self.beta*np.mean(next_STEM_payoffs)-
            self.gamma_p*self.tuition[2]+self.flows[0])
            flow_nonSTEM=(self.beta*np.mean(next_nonSTEM_payoffs)-
                self.gamma_p*self.tuition[2]+self.flows[1])

            flow_hp=self.beta*self.dropout_payouts[2,0]
            flow_unskilled=(self.gamma_p*self.unskilled_wages+self.flows[2]+
                self.beta*self.dropout_payouts[2,1])

            # switching cost variants of flows
            flow_prior_STEM = flow_STEM+self.switching_costs[0]
            flow_prior_nonSTEM = flow_nonSTEM+self.switching_costs[1]

            # prior choice STEM
            self.EmaxEducationValues[2,gpa_idx,0]=(0.57722+
                flow_prior_STEM+np.mean(np.log(np.exp(flow_hp-flow_prior_STEM)+
                    1+np.exp(flow_nonSTEM-flow_prior_STEM)+
                    np.exp(flow_unskilled-flow_prior_STEM))))
            # prior choice nonSTEM
            self.EmaxEducationValues[2,gpa_idx,1]=(0.57722+
                flow_STEM+np.mean(np.log(np.exp(flow_hp-flow_STEM)+
                    1+np.exp(flow_prior_nonSTEM-flow_STEM)+
                    np.exp(flow_unskilled-flow_STEM))))

            # not possible for prior choice STEM, so map to same value
            self.EmaxEducationValues[3,gpa_idx]=(0.57722+
                flow_prior_nonSTEM+np.mean(np.log(np.exp(
                    flow_hp-flow_prior_nonSTEM)+1+
                np.exp(flow_unskilled-flow_prior_nonSTEM))))


    def Emax2(self):

        for gpa_idx in range(self.gpa_num_blocks):
            gpa=self.gpa_levels[gpa_idx]

            # generate next grade, rounded to nearest 5, and top/bottom capped
            # at 0, 400 GPA
            STEM_grades=self.norm_quantiles*self.sigma_S1+100*self.Egrade(1,2)
            for x in range(self.num_quantiles):
                if STEM_grades[x]>400:
                    STEM_grades[x]=400
                elif STEM_grades[x]<0:
                    STEM_grades[x]=0
            next_STEM_grades=np.zeros(len(STEM_grades),dtype=np.int64)
            for x in range(self.num_quantiles):
                next_STEM_grades[x]=self.__round_to_5((STEM_grades[x]+gpa)/2)

            # calculate next sem grades

            nonSTEM_grades=self.norm_quantiles*self.sigma_N1+100*self.Egrade(0,2)
            for x in range(self.num_quantiles):
                if nonSTEM_grades[x]>400:
                    nonSTEM_grades[x]=400
                elif nonSTEM_grades[x]<0:
                    nonSTEM_grades[x]=0
            next_nonSTEM_grades=np.zeros(self.num_quantiles,dtype=np.int64)
            for x in range(self.num_quantiles):
                next_nonSTEM_grades[x]=self.__round_to_5((
                    nonSTEM_grades[x]+gpa)/2)

            # calculate flow utilities for each of the 3 possible states


            t2_STEM_t3_met=np.zeros(self.num_quantiles,dtype=np.float64)
            t2_nonSTEM_t3_met=np.zeros(self.num_quantiles,dtype=np.float64)
            t2_nonSTEM_t3_not_met=np.zeros(self.num_quantiles,dtype=np.float64)
            
            for x in range(self.num_quantiles):
                t2_STEM_t3_met[x]=self.EmaxEducationValues[2,
                self.__gpa_to_index(next_STEM_grades[x]),0]
                t2_nonSTEM_t3_met[x]=self.EmaxEducationValues[2,
                self.__gpa_to_index(next_nonSTEM_grades[x]),1]
                t2_nonSTEM_t3_not_met[x]=self.EmaxEducationValues[3,
                self.__gpa_to_index(next_nonSTEM_grades[x]),1]


            flow_STEM_prior_nonSTEM=(self.beta*np.mean(t2_STEM_t3_met)-
                self.gamma_p*self.tuition[1]+self.flows[0])
            flow_STEM_prior_STEM = (self.beta*np.mean(t2_STEM_t3_met)-
                self.gamma_p*self.tuition[1]+self.flows[0] +
                self.switching_costs[0])

            flow_nonSTEM_t3_met_prior_STEM = (
                self.beta*np.mean(t2_nonSTEM_t3_met)-
                self.gamma_p*self.tuition[1]+self.flows[1])
            flow_nonSTEM_t3_not_met_prior_nonSTEM=(
                self.beta*np.mean(t2_nonSTEM_t3_not_met)-
                self.gamma_p*self.tuition[1]+self.flows[1] +
                self.switching_costs[1])            
            flow_hp=self.beta*self.dropout_payouts[1,0]
            flow_unskilled=(self.gamma_p*self.unskilled_wages+self.flows[2]+
                self.beta*self.dropout_payouts[1,1])

            # STEM req met when prior choice was STEM (necessarily)
            self.EmaxEducationValues[0,gpa_idx]=(0.57722+
                flow_STEM_prior_STEM+np.mean(np.log(
                    np.exp(flow_hp-flow_STEM_prior_STEM)+1+
                    np.exp(flow_nonSTEM_t3_met_prior_STEM -
                        flow_STEM_prior_STEM)+
                    np.exp(flow_unskilled - flow_STEM_prior_STEM))))


            # STEM req not met when prior choice was nonSTEM
            self.EmaxEducationValues[1,gpa_idx]=(0.57722+
                flow_STEM_prior_nonSTEM+np.mean(np.log(
                    np.exp(flow_hp - flow_STEM_prior_nonSTEM)+1+
                    np.exp(flow_unskilled - flow_STEM_prior_nonSTEM)+
                np.exp(flow_nonSTEM_t3_not_met_prior_nonSTEM - 
                    flow_STEM_prior_nonSTEM))))

    def Emax1(self):

        STEM_grades=self.norm_quantiles*self.sigma_S1+100*self.Egrade(1,1)
        nonSTEM_grades=self.norm_quantiles*self.sigma_N1+100*self.Egrade(0,1)

        for x in range(self.num_quantiles):
            if STEM_grades[x]>400:
                STEM_grades[x]=400
            elif STEM_grades[x]<0:
                STEM_grades[x]=0
        for x in range(self.num_quantiles):
            if nonSTEM_grades[x]>400:
                nonSTEM_grades[x]=400
            elif nonSTEM_grades[x]<0:
                nonSTEM_grades[x]=0

        next_STEM_grades=np.zeros(len(STEM_grades),dtype=np.int64)
        next_nonSTEM_grades=np.zeros(len(nonSTEM_grades),dtype=np.int64)

        for x in range(self.num_quantiles):
            next_STEM_grades[x]=self.__round_to_5(STEM_grades[x])
            next_nonSTEM_grades[x]=self.__round_to_5(nonSTEM_grades[x])

        next_STEM_payoffs=np.zeros(self.num_quantiles,dtype=np.float64)
        next_nonSTEM_payoffs=np.zeros(self.num_quantiles,dtype=np.float64)

        for x in range(self.num_quantiles):
            next_STEM_payoffs[x]=self.EmaxEducationValues[0,
            self.__gpa_to_index(next_STEM_grades[x]),0]
            next_nonSTEM_payoffs[x]=self.EmaxEducationValues[1,
            self.__gpa_to_index(next_nonSTEM_grades[x]),1]

        # t=1 choice does not have switching costs 
        self.STEM_cond_val_first=(np.mean(next_STEM_payoffs)*self.beta+
            self.flows[0]+self.switching_costs[0]-self.gamma_p*self.tuition[0])
        self.nonSTEM_cond_val_first=(np.mean(next_nonSTEM_payoffs)*self.beta+
            self.flows[1]+self.switching_costs[1]-self.gamma_p*self.tuition[0])

def main():
    # Initialize parameters
    grad_horizon=10
    flowsSkilled=np.array([-1,-2,-2],dtype=np.float64)
    flows_zero_skilled = np.array([-4,-5,-5],dtype=np.float64)
    switch_costs = np.array([1,1,1],dtype=np.float64)
    switch_costs_unskilled = np.array([1],dtype=np.float64)

    flows_zero_skilled_penalized = flows_zero_skilled - switch_costs
    flows_skilled_penalized = flowsSkilled - switch_costs
    flowUnskilled=-1
    flow_unskilled_penalized = flowUnskilled-switch_costs_unskilled[0]
    flowSTEM=-1
    flownonSTEM=-1
    sectors=len(flowsSkilled)
    wage_covar=[[0.4, 0.2, 0], [0.2, 0.6, -0.3], [0, -0.3, 0.4]]
    gamma_p=0.08
    beta=0.95
    unskilled_var=np.array([[0.3]],dtype=np.float64)
    wage_coeffs_full=np.array([[3.5,-0.3, 0.3, 0.2, 0.04,0.02,0.02, -0.005,0.10,
        0.15,0.01],
    [3.4,0.3, 0.2, 0.0, 0.02, 0.03,0.02,-0.005, 0.1,0.15,0.01],
    [3.2,-0.2, 0.1, 0.1, .02, 0.02, 0.03,-0.003,0.1,0.15,0.01],
    [2.8,0, 0, 0, 0.1,0,0, -.003,0.1,0,0.01]],dtype='float64')

    year_four_intercept=2.9

    grade_params=np.array([[.0022,.000189,.00447,-0.09,.039,.023,.19,.25,0.21],
    [.00178,.00086,0.00529,-0.36,-0.01,0.04,0.13,.40,0.35]],dtype=np.float64)

    normReps=9
    zscores=scipy.stats.norm.ppf(np.array(range(1,normReps+1))/(normReps+1))
    base_draws=np.matrix.transpose(np.matrix(list(
        itertools.product(zscores,repeat=(sectors)))))
    lmat=np.linalg.cholesky(wage_covar)
    wage_shocks=np.array(np.transpose(np.matmul(lmat,base_draws)))

    LaborGradeRange=np.linspace(2,4,21)

    num_quantiles=20
    norm_quantiles=scipy.stats.norm.ppf(
        np.array(range(1,num_quantiles))/num_quantiles)

    choose=simplexmap.pascal(grad_horizon+4,sectors)

    zero_exp_penalty = np.array([1,1,1],dtype=np.float64)
    # generate Emax for college grads
    def GenerateEmaxLaborFunctions(grad_horizon,gamma_p,beta,wage_coeffs,
        flows_penalized,major,grade,wage_shocks,choose,quality,
        time_zero_flows_penalized,switch_costs):


        solution=EmaxLaborFunctionsJIT(5,grad_horizon,gamma_p,beta,
            wage_coeffs,major,grade,flows_penalized,wage_shocks,choose,quality,
            time_zero_flows_penalized,switch_costs,zero_exp_penalty,
            wage_shocks)
        solution.solveLabor()
        return solution.EmaxList

    labor_Emax={}
    STEM_payouts=np.zeros(21,dtype=np.float64)
    nonSTEM_payouts=np.zeros(21,dtype=np.float64)
    
    # quality = 1 for these schools
    for idx,grade in enumerate(LaborGradeRange):
        labor_Emax[('STEM',int(100*grade))]=GenerateEmaxLaborFunctions(
            grad_horizon,gamma_p,beta,wage_coeffs_full[:-1],
            flows_skilled_penalized,1,grade,wage_shocks,choose,1,
            flows_zero_skilled_penalized,switch_costs)
        labor_Emax[('nonSTEM',int(100*grade))]=GenerateEmaxLaborFunctions(
            grad_horizon,gamma_p,beta,wage_coeffs_full[:-1],
            flows_skilled_penalized,0,grade,wage_shocks,choose,1,
            flows_zero_skilled_penalized,switch_costs)
        STEM_payouts[idx]=labor_Emax[('STEM',int(100*grade))][0,0,0]
        nonSTEM_payouts[idx]=labor_Emax[('nonSTEM',int(100*grade))][0,0,0]

    # interpolation step?


    # unskilled payouts
    dropout_payouts=np.zeros((4,2),dtype=np.float64)
    unskilled_reps=20
    unskilled_wage_shocks=(unskilled_var[0][0]**0.5*
    np.array(np.transpose(np.matrix(scipy.stats.norm.ppf(
        np.array(range(1,unskilled_reps+1))/(unskilled_reps+1))))))

    for drop_time in range(4):
        if drop_time==3:
            unskilled_wage_coeffs_final = wage_coeffs_full[-1].copy()
            unskilled_wage_coeffs_final[0] = year_four_intercept

            unskilled_Emax=EmaxLaborFunctionsJITUnskilled(
                grad_horizon+4-drop_time-1,gamma_p,beta,
                np.array([unskilled_wage_coeffs_final],dtype=np.float64),0,0,
                np.array([flow_unskilled_penalized],dtype=np.float64),
                unskilled_wage_shocks,choose,switch_costs_unskilled,0)
        else:
            unskilled_Emax=EmaxLaborFunctionsJITUnskilled(
                grad_horizon+4-drop_time-1,gamma_p,beta,
                np.array([wage_coeffs_full[-1]],dtype=np.float64),0,0,
                np.array([flow_unskilled_penalized],dtype=np.float64),
                unskilled_wage_shocks,choose,switch_costs_unskilled,0)
        unskilled_Emax.solveLabor()
        dropout_payouts[drop_time,0]=unskilled_Emax.EmaxList[1,0,1]
        dropout_payouts[drop_time,1]=unskilled_Emax.EmaxList[1,1,0]


    # Emax function
    flows_educ=np.array([flowSTEM,flownonSTEM,flow_unskilled_penalized],
        dtype=np.float64)
    tuition=np.array([2,2,2,2],dtype=np.float64)
    # SAT of 500 in both, and 3.5 GPA
    exog_chars=np.array([500,500,350],dtype=np.float64)
    ability=np.array([0,0],dtype=np.float64)
    unskilled_meanvar=np.array([wage_coeffs_full[-1][0],unskilled_var[0][0]],
        dtype=np.float64)

    education_switch_costs = np.array([1,1],dtype=np.float64)
    educ_Emax=EmaxEducationJIT(dropout_payouts, STEM_payouts, nonSTEM_payouts,
        grade_params, gamma_p, beta, flows_educ, tuition, exog_chars, ability,
        unskilled_meanvar, norm_quantiles,year_four_intercept,
        education_switch_costs)
    educ_Emax.solve()
    print(educ_Emax.EmaxEducationValues)
if __name__ == '__main__':
    main()