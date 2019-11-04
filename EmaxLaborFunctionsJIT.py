import numpy as np
import numba

from numba import jitclass          # import the decorator
from numba import int64, float64    # import the types
import simplexmap


# JIT variable declarations
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

#===============================================================================
# Generates labor Emax functions (ex-ante value functions) for skilled labor
# market.
#===============================================================================

@jitclass(labor_spec)
class EmaxLaborFunctionsJIT:

    def __init__(self,horizon,final_horizon,
        gamma_p,beta,wage_coeffs,STEM,
        grade,flows_penalized,wage_shocks,choose,quality,
        time_zero_flows_penalized,switch_costs,zero_exp_penalty,
        wage_shocks_later):
        """
        Initializes a labor market model with the following variables:

        horizon: first time horizon for first set of wage shocks
        final_horizon: second time horizon for later set of wage shocks, usu. 
            lower dim.
        gamma_p: individual preference for income rel. to pref. shocks
        beta: discount factor
        wage_coeffs:
            3x11 array, for skilled sectors 1-3
            [C,M,G_N,G_S,x1,x2,x3,own sq, own 1st year, quality, total exp]
        STEM: 1 if STEM, 0 otherwise
        grade: terminal GPA, in units of raw GPA, i.e. 0 - 4.0
        flows_penalized: length 3 vector of flow utilities for each sector,
            where leisure is 0. Penalized = subtract out switching cost but NOT
            zero exp penalty
        wage_shocks: distribution of wage shocks to use (precalculated) up to
            time horizon, e.g.
            zscores=scipy.stats.norm.ppf(np.array(range(1,normReps+1))/
            (normReps+1))
            base_draws=np.matrix.transpose(np.matrix(list(
            itertools.product(zscores,repeat=(sectors)))))
            lmat=np.linalg.cholesky(skilled_wage_covar)
            wageshocks=np.array(np.transpose(np.matmul(lmat,base_draws)))
        choose: Array of binomial coefficients of Pascal's triangle. Run
            choose=simplexmap.pascal(horizon_later,sectors)
        quality: 1 or 0 for if the school is high quality
        time_zero_flows_penalized: specific skilled labor flow utilities for the
            first year after college (time = 5 for graduates). Penalized = sub.
            switching cost
        switch_costs: specific skilled labor flow utility penalty for switching
            into the sector if the prior choice was different. Values are
            POSITIVE for switching costs
        zero_exp_penalty: penalty for having no experience. Negative.
        wage_shocks_later:  Analogue of wage_shocks, but from horizon up to
            final_horizon. Should be lower dimension to reduce comp. burden.

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
        self.flows_penalized=np.zeros(self.sectors+1,dtype=np.float64)
        self.switch_costs = np.zeros(self.sectors+1,dtype=np.float64)
        self.switch_costs[0:self.sectors]=switch_costs
        self.flows_penalized[0:self.sectors]=flows_penalized
        self.zero_exp_penalty = np.zeros(self.sectors+1)
        self.zero_exp_penalty[0:self.sectors] = zero_exp_penalty
        self.quality = quality
        self.wage_shocks_later = wage_shocks_later
        self.time_zero_flows_penalized=np.zeros(self.sectors+1)
        self.time_zero_flows_penalized[0:self.sectors]=time_zero_flows_penalized
        self.zero_exp_penalty = np.zeros(self.sectors+1)
        self.zero_exp_penalty[0:self.sectors] = zero_exp_penalty

        #=======================================================================
        # experience-agnostic log-wage payout
        # i.e. log-wage including constant, returns to major and GPA, and
        # quality
        #=======================================================================
        self.base_wage=np.zeros(self.sectors,dtype=np.float64)
        for i in range(self.sectors):
            if self.STEM==0:
                mg=self.wage_coeffs[i][2]*self.grade
            else:
                mg=self.wage_coeffs[i][1]+self.wage_coeffs[i][3]*self.grade
            self.base_wage[i]=(self.wage_coeffs[i][0]+mg+self.wage_coeffs[i][9]*
                self.quality)
        #=======================================================================
        # EmaxList arguments: 1st is time, second is a scalar representing
        # the accumulated exp, and 3rd is the prior choice
        #=======================================================================
        self.EmaxList=np.zeros((self.final_horizon+1,
            self.choose[self.final_horizon+self.sectors,self.sectors],
            self.sectors+1), dtype=np.float64)

    #===========================================================================
    # generates an array at years number of experience, with each row being an
    # experience tuple such that the total experience adds up to years
    #===========================================================================
    def expCombos(self,years):
        num_combos=self.choose[years+self.sectors][self.sectors]
        out=np.zeros((num_combos,self.sectors+1),dtype=np.int64)
        for x in range(num_combos):
            out[x]=simplexmap.array_to_combo(x,years,self.sectors,self.choose)
        return out

    #===========================================================================
    # Determines the Emax function at a given time, assuming that all
    # later value functions have already been calculated.
    # If time = final_horizon, then no future value term; otherwise need to 
    # add in discounted future value term.
    # Per period flow of each alternative is given by
    # non-pecuniary flow + wages * gamma_p
    # The ex-ante value function is the average of the log-sum-exponentials over
    # all choice options, conditional on the particular wage shock draw. When
    # calculating the log-sum-exponential, first subtract the max then add it
    # back in (i.e. log(exp(a)+exp(b))=a+log(1+exp(b-a)) ) to prevent overflow.
    #===========================================================================

    def EmaxLabor(self,time):
        exp_allocs=self.expCombos(time)
        len_allocs=exp_allocs.shape[0]

        # which version of the wage shock will I integrate over?
        if time>self.horizon:
            N = self.N_later
            wage_shocks = self.wage_shocks_later
        else:
            N = self.N
            wage_shocks = self.wage_shocks

        # terminal condition so no next-period payoff
        if time==self.final_horizon:
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
                    self.EmaxList[self.final_horizon,i,last_choice]=(
                        np.mean(outcome)+0.57722)            

        # non-terminal, so have to include next period payoff
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

    #===========================================================================
    # calculate ex-ante value functions, moving backwards from terminal period
    # given by horizon_later
    #===========================================================================

    def solveLabor(self):
        i=self.final_horizon
        while i>=1:
            self.EmaxLabor(i)
            i=i-1

        # Final t = 0 case
        time=0
        exp_allocs=self.expCombos(time)
        len_allocs=exp_allocs.shape[0]
        for i in range(len_allocs):
            wages=np.hstack((np.exp(self.wage_shocks+
                self.ELogWage(exp_allocs[i])),np.zeros((self.N,1))))

            current_flow=(self.time_zero_flows_penalized+self.zero_exp_penalty+
                wages*self.gamma_p)
            outcome=np.zeros(self.N,dtype=np.float64)

            next_exp=np.zeros((self.sectors+1,self.sectors+1),
                dtype=np.int64)
            disc_next_payoff=np.zeros(self.sectors+1,dtype=np.float64)

            for j in range(self.sectors+1):
                next_exp[j][j]=1
            next_exp=next_exp+exp_allocs[i]

            for j in range(self.sectors+1):
                disc_next_payoff[j]=self.beta*self.EmaxList[time+1,
                simplexmap.combo_to_array(time+1,next_exp[j],self.choose),j]

            net_payoff=current_flow+disc_next_payoff
            for j in range(self.N):
                maxvalue=np.max(net_payoff[j])
                outcome[j]=maxvalue+np.log(np.sum(np.exp(
                    net_payoff[j]-maxvalue)))
            sol = np.mean(outcome)+0.57722

            for j in range(self.sectors+1):
                self.EmaxList[time,i,j]=sol

    #===========================================================================
    # Generate an array of the log-wages, net the wage shocks, conditional on
    # the experience exp as well as the wage coefficients. Array is over all 3
    # sectors.
    # There are two cases here. If experience is too high (case 2) and
    # past the peak of the quadratic experience profile, set
    # experience-returns to max of quadratic
    #===========================================================================
    
    def ELogWage(self,exp):

        out=np.zeros(self.sectors,dtype=np.float64)
        total_exp = np.sum(exp)


        for i in range(self.sectors):
            at_least_one_exp = 1 if exp[i]>0 else 0

            if -2*exp[i]*self.wage_coeffs[i][7]<=self.wage_coeffs[i][i+4]:
                out[i]=(self.base_wage[i]+exp[0]*self.wage_coeffs[i][4]+
                    exp[1]*self.wage_coeffs[i][5]+exp[2]*self.wage_coeffs[i][6]+
                    exp[i]**2*self.wage_coeffs[i][7]+
                    total_exp*self.wage_coeffs[i][10]+
                    at_least_one_exp*self.wage_coeffs[i][8])

            else:
                out[i]=(self.base_wage[i]+exp[0]*self.wage_coeffs[i][4]+
                    exp[1]*self.wage_coeffs[i][5]+exp[2]*self.wage_coeffs[i][6]-
                    exp[i]*self.wage_coeffs[i][i+4]-
                    self.wage_coeffs[i][i+4]**2/(4*self.wage_coeffs[i][7])+
                    total_exp*self.wage_coeffs[i][10]+
                    at_least_one_exp*self.wage_coeffs[i][8])
        return out