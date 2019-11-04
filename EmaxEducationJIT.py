import numpy as np
import numba

from numba import jitclass          # import the decorator
from numba import int64, float64    # import the types
import simplexmap


# JIT variable declarations
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

#===============================================================================
# Generates ex-ante value functions for 4 years of college. Needs the payouts to
# dropout as well as graduating by major/gpa to be calculated first.
#===============================================================================

@jitclass(ed_spec)
class EmaxEducationJIT:

    def __init__(self, dropout_payouts, STEM_payouts, nonSTEM_payouts,
        grade_params, gamma_p, beta, flows_penalized, tuition, exog_chars,
        ability, unskilled_meanvar, norm_quantiles, year_four_intercept,
        year_four_flow_penalized, switching_costs, grad_payoff):
        """
        dropout_payouts: 4x2 array with the next period payouts,
            corresponding to each year of dropout, and col. 0 = HP,
            1 = unskilled
        STEM_payouts: 21-length array with each row the ex-ante payout
            at zero experience for STEM grads (ascending from 200 to 400,
            every 10)
        nonSTEM_payouts: 21-length array with each row the ex-ante payout
            at zero experience for STEM grads
        grade_params: 9x2 vector by major:
            [SAT_M, SAT_V, hs_GPA,lambda_1,lambda_2,lambda_3,lambda_4,
            sigma^2_m1,sigma^2_m3]
            STEM comes first, and sigma^2_m1 and m3 refer to first 2 years or
            last 2
        gamma_p: preference for money (tuition)
        beta: discount factor
        flows_penalized: [STEM penalized, nonSTEM penalized,
            unskilled_penalized]
            STEM/nonSTEM penalized is per-period flow minus switching cost
            unskilled is flow minus switching cost and zero exp penalty, for
            years 2-3
        tuition: yearly tuition, a 4x1 vector
        exog_chars: array [SAT_M,SAT_V, hs_GPA]. SAT are in raw units,
            hs_GPA is in units of 100 (e.g. a perfect GPA is 400)
        ability: [STEM_ability,nonSTEM_ability]
        unskilled_meanvar: [mean,var] for log-normal unskilled wages at zero
            experience, for years 2 or 3 dropout
        norm_quantiles: array of normal quantiles used to integrate
            over grade shocks as well as wage shocks
        year_four_intercept: log-wage intercept for year 4 dropout
        year_four_flow_penalized: year four flow utility for unskilled labor,
            which includes switching + zero exp penalty
        switching_costs: length 2 array of cost of switching into STEM or
            non-STEM
        grad_payoff: t = 5 non-pecuniary payoff of graduation
        """

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

    #===========================================================================
    # Student's expected grade given grade coeffs, exog chars, and major/year
    # Major is 1 for STEM, 0 for non-STEM, year is 1-4 
    #===========================================================================

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

    #===========================================================================
    # I discretize the state space over grades, so I use a number of rounding
    # functions and conversions between array index and actual grade values 
    #===========================================================================

    def __round_to(self, n, precision):
        correction = 0.5 if n >= 0 else -0.5
        return int( n/precision+correction ) * precision

    def __round_to_5(self,n):
        return self.__round_to(n, 5)

    def __round_to_10(self,n):
        return self.__round_to(n,10)

    def __gpa_to_index(self,gpa):
        return int(gpa/5)

    def __tgpa_to_index(self,gpa):
        return int((gpa-200)/10)

    def solve(self):
        self.Emax4()
        self.Emax3()
        self.Emax2()
        self.Emax1()
        
    #===========================================================================
    # Generate ex-ante value by calculating conditional value functions and
    # taking the Emax. For educational choices, integrate over possible grade
    # outcomes in the current period (choice major, then realize grades).
    # Ex-ante value functions are indexed 0-5 depending on choice history
    # 0:
    # 1:
    # 2: STEM pre-req met at t=3
    # 3: STEM pre-req not met at t=3
    # 4: non-STEM at t=4 (t=3 chose STEM)
    # 5: STEM at t=4 (t=3 chose non-STEM)
    #===========================================================================

    def Emax4(self):
        # STEM case (time = 3 chose STEM, so only STEM option)
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

            # link grade to labor market payoff. fail out if GPA < 2.0
            graduation_payoff=np.zeros(self.num_quantiles,dtype=np.float64)
            for x in range(self.num_quantiles):
                if next_grades[x]>=200:
                    graduation_payoff[x]=self.STEM_payouts[
                    self.__tgpa_to_index(next_grades[x])] + self.grad_payoff
                else:
                    graduation_payoff[x]=self.dropout_payouts[3,0]

            flow_school=(self.beta*np.mean(graduation_payoff)-self.gamma_p*
            self.tuition[3]+self.flows[0]+self.switching_costs[0])
            flow_hp=self.beta*self.dropout_payouts[3,0]

            flow_unskilled=(self.gamma_p*self.year_four_wages+
                self.year_four_flow_penalized +
                self.beta*self.dropout_payouts[3,1])

            self.EmaxEducationValues[4][gpa_idx]=(flow_school+
                np.mean(np.log(np.exp(flow_hp-flow_school)+
                1+np.exp(flow_unskilled-flow_school)))+0.57722)


        # # non-STEM case
        for gpa_idx in range(self.gpa_num_blocks):
            gpa=self.gpa_levels[gpa_idx]
            sem_grades=self.norm_quantiles*self.sigma_N2+100*self.Egrade(0,4)

            for x in range(self.num_quantiles):
                if sem_grades[x]>400:
                    sem_grades[x]=400
                elif sem_grades[x]<0:
                    sem_grades[x]=0
            next_grades=np.zeros(self.num_quantiles,dtype=np.int64)
            for x in range(self.num_quantiles):
                next_grades[x]=self.__round_to_5((sem_grades[x]+3*gpa)/4)
            graduation_payoff=np.zeros(self.num_quantiles,dtype=np.float64)
            for x in range(self.num_quantiles):
                if next_grades[x]>=200:
                    graduation_payoff[x]= self.nonSTEM_payouts[
                    self.__tgpa_to_index(next_grades[x])] + self.grad_payoff
                else:
                    graduation_payoff[x]=self.dropout_payouts[3,0]

            flow_school=(self.beta*np.mean(graduation_payoff)-self.gamma_p*
            self.tuition[3]+self.flows[1]+self.switching_costs[1])
            flow_hp=self.beta*self.dropout_payouts[3,0]
            flow_unskilled=(self.gamma_p*self.year_four_wages+
                self.year_four_flow_penalized+
                self.beta*self.dropout_payouts[3,1])
            self.EmaxEducationValues[5][gpa_idx]=(flow_school+
                np.mean(np.log(np.exp(flow_hp-flow_school)+
                1+np.exp(flow_unskilled-flow_school)))+0.57722)

    def Emax3(self):
        # 4 possible outcomes: STEM, non-STEM, leisure, unskilled labor
        # STEM eligible

        for gpa_idx in range(self.gpa_num_blocks):
            gpa=self.gpa_levels[gpa_idx]

            STEM_grades=self.norm_quantiles*self.sigma_S2+100*self.Egrade(1,3)
            for x in range(self.num_quantiles):
                if STEM_grades[x]>400:
                    STEM_grades[x]=400
                elif STEM_grades[x]<0:
                    STEM_grades[x]=0
            next_STEM_grades=np.zeros(len(STEM_grades),dtype=np.int64)

            # calculate next sem grades
            nonSTEM_grades = (self.norm_quantiles*self.sigma_N2+
                100*self.Egrade(0,3))
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

            next_STEM_payoffs=np.zeros(self.num_quantiles,dtype=np.float64)
            next_nonSTEM_payoffs=np.zeros(self.num_quantiles,dtype=np.float64)

            for x in range(self.num_quantiles):
                next_STEM_payoffs[x]=self.EmaxEducationValues[4,
                self.__gpa_to_index(next_STEM_grades[x]),0]
                next_nonSTEM_payoffs[x]=self.EmaxEducationValues[5,
                self.__gpa_to_index(next_nonSTEM_grades[x]),1]

            # calculate conditional choice values
            flow_STEM=(self.beta*np.mean(next_STEM_payoffs)-
            self.gamma_p*self.tuition[2]+self.flows[0])
            flow_nonSTEM=(self.beta*np.mean(next_nonSTEM_payoffs)-
                self.gamma_p*self.tuition[2]+self.flows[1])

            flow_hp=self.beta*self.dropout_payouts[2,0]
            flow_unskilled=(self.gamma_p*self.unskilled_wages+self.flows[2]+
                self.beta*self.dropout_payouts[2,1])

            # incorporate switching costs variants of flows
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

            STEM_grades=self.norm_quantiles*self.sigma_S1+100*self.Egrade(1,2)
            for x in range(self.num_quantiles):
                if STEM_grades[x]>400:
                    STEM_grades[x]=400
                elif STEM_grades[x]<0:
                    STEM_grades[x]=0
            next_STEM_grades=np.zeros(len(STEM_grades),dtype=np.int64)
            for x in range(self.num_quantiles):
                next_STEM_grades[x]=self.__round_to_5((STEM_grades[x]+gpa)/2)

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


            # incorporate switching costs

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