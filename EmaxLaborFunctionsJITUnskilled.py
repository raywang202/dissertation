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
# Unskilled labor market variant of EmaxLaborFunctionsJIT
# Only one occupational sector and no need to split the horizons into two pieces
# due to smaller state space.
#===============================================================================

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
            Major, GPA, quality, and age returns are zero by restriction
        STEM: irrelevant, 1 or 0
        grade: irrelevant, terminal GPA, in units of raw GPA, i.e. 0 - 4.0
        flows_penalized: length 1 array of flow utilities for unskilled sector,
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
        switch_costs: specific skilled labor flow utility penalty for switching
            into the sector if the prior choice was different. Values are
            POSITIVE for switching costs
        zero_exp_penalty: penalty for having no experience. Negative.
        quality: 1 or 0 for if the school is high quality
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
        self.switch_costs=np.zeros(self.sectors+1,dtype=np.float64)
        self.switch_costs[0:self.sectors]=switch_costs
        self.flows_penalized = np.zeros(self.sectors+1)
        self.flows_penalized[0:self.sectors]=flows_penalized
        self.zero_exp_penalty = zero_exp_penalty

        #=======================================================================
        # Experience-agnostic log-wage level
        # Here it's just the intercept
        #=======================================================================

        self.base_wage=np.zeros(self.sectors,dtype=np.float64)
        for i in range(self.sectors):
            self.base_wage[i]=(self.wage_coeffs[i][0])        
        self.EmaxList=np.zeros((
            self.horizon+1,self.choose[self.horizon+self.sectors,self.sectors],
                self.sectors+1),dtype=np.float64)

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

        # terminal condition, so no next-period payoff
        if time==self.horizon:
            for i in range(len_allocs):
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

        # non-terminal, so have to include next period payoff
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

    #===========================================================================
    # calculate ex-ante value functions, moving backwards from terminal period
    # given by horizon_later
    #===========================================================================

    def solveLabor(self):
        i=self.horizon
        while i>=0:
            self.EmaxLabor(i)
            i=i-1

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