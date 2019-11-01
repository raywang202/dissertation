# approximates the integrals in later periods
# and also reduces the number of Emaxes to be calculated in the skilled likelihood routine
# switching costs in education too!

import numpy as np
import pandas as pd
from numba import jit
import multiprocessing
import scipy
import itertools
import time
import scipy.interpolate


from modelsimjit_final import *
from DataPreProcessApprox import *
import simplexmap

def main():
    sectors=3 # occupational sectors
    LaborGradeRange=np.linspace(2,4,11) # discretization of GPA

    # Import data .csv and pre-process it
    # This generates the file DFDataFinal which is the set of observations
    #fileLocation='/finaldata3.csv'
    #[DFDataFinal,SATTuition]=PreProcess(fileLocation,sectors,LaborGradeRange)
    #DFData=DFDataFinal.set_index('id')
    # alread preprocessed
    DFDataFinal=pd.read_csv(os.path.abspath(os.curdir)+
        '/finaleduc_05-28.csv')

    SATTuition=list(set(list(DFDataFinal[
        ['A_N','A_S','SAT_M','SAT_V','hs_GPA','quality']].\
        itertuples(index=False,name=None))))


    DFData=DFDataFinal.set_index('id')

    # Fixed Parameters

    grad_horizon=10 # number of years after grad over which individual can work
    normReps=4 # number of times to approximate normal integrals
    simReps=9
    # max size of labor state space, given by grad horizon choose 3
    final_size=int((grad_horizon+2)*(grad_horizon+1)*grad_horizon/6)

    # vector of per-period flow utilities. order is all skilled, nonskilled,
    # STEM, nonSTEM. Note that leisure is always normalized to 0 so it is
    # excluded
    flows_skilled=[-1,-2,-1.5]
    time_zero_flows = [-4,-5,-6.5]
    # switching costs for skilled 1,2,3, unskilled
    switch_costs_full = [1,1,1,0.5]

    flows_skilled_penalized = [-2,-3,-2.5]
    flowUnskilled=-1
    unskilled_switch_cost = switch_costs_full[3]
    flow_unskilled_penalized=flowUnskilled - unskilled_switch_cost
    # penalized flows can be calculated by subtracting skilled switching costs
    # from time zero flows
    time_zero_flows_penalized = [-5,-6,-7.5]
    flowSTEM=-1
    flownonSTEM=-0.5

    flowsFull=(flows_skilled_penalized+
        [flow_unskilled_penalized,flowSTEM,flownonSTEM]+
        time_zero_flows_penalized)
    flows=np.array(flowsFull,dtype=np.float64)
    gamma_p=0.08 # preference for money
    beta=0.95 # discount factor
    ability=np.array([0,0]) # unobserved ability
    # covariance matrix for college graduates
    skilled_wage_covar=np.array([[0.001, 0, 0], [0.00, 0.0015, 0],
        [0, 0, 0.001]])

    # variance for unskilled
    unskilled_var=np.array([[.01]])
    # wage coefficients, for skilled sectors 1-3, then unskilled
    # params given by C, M, GPA_N, GPA_S, x1, x2, x3, ownsq, >1, quality, age

    wage_coeffs_full=np.array([
        [3.5,-0.75, 0, .25, 0.05,0.01,0.01, -0.002,0.07, 0.05,0.02],
        [3.15,0, 0.09, 0, 0.01,0.07,0.01, -0.004,0.02,0.14,0.02],
        [3.1,-0.4, 0, 0.15, 0.01, 0.01,0.04,-0.001,0.05,0.05,0.02],
        [3,0, 0, 0, 0.07, 0,0,-0.002,0.03,0,0.02]],dtype=np.float64)

    year_four_intercept=3.1

    # 8x2 vector by major:
    #   [SAT_M, SAT_V, hs_GPA, lambda_1,lambda_2,lambda_3,lambda_4,sigma^2_m]
    # first 
    ####
    grade_params=np.array([[.0022,.000189,.00447,-0.09,.039,.023,.19,.25,0.21],
    [.00178,.00086,0.00529,-0.36,-0.01,0.04,0.13,.40,0.35]],dtype=np.float64)



    out=LogLikeFullSolution(DFData,SATTuition,grad_horizon,sectors,gamma_p,beta,
    ability,flows,wage_coeffs_full,skilled_wage_covar,unskilled_var,
    grade_params,normReps,simReps,LaborGradeRange,final_size,
    np.array(switch_costs_full,dtype=np.float64),year_four_intercept,
    True)
    print('loglike is'+str(out))


@jit(nopython=True)
def takeClosest(myNumber, myList):
    idx = (np.abs(myList - myNumber)).argmin()
    return myList[idx]

# rounding functions
@jit(nopython=True)
def round_to(n, precision):
    correction = 0.5 if n >= 0 else -0.5
    return int( n/precision+correction ) * precision

@jit(nopython=True)
def round_to_5(n):
    return round_to(n, 5)

# returns array of expected wages for skilled sector,
# given grades (in units of 100)
# experience is for sectors 1, 2, 3, and HP
@jit(nopython=True)
def elogwage(skilled_wage_coeffs,experience,dSTEM,GPA,quality):
    out=np.zeros(skilled_wage_coeffs.shape[0],dtype=np.float64)
    total_exp = np.sum(experience)
    for i in range(skilled_wage_coeffs.shape[0]):
        at_least_one_exp = 1 if experience[i]>0 else 0

        if (-2*experience[i]*skilled_wage_coeffs[i][7]<=
            skilled_wage_coeffs[i][i+4]):
            if dSTEM==0:
                out[i]=(skilled_wage_coeffs[i][0]+
                skilled_wage_coeffs[i][2]*GPA/100+
                skilled_wage_coeffs[i][4]*experience[0]+
                skilled_wage_coeffs[i][5]*experience[1]+
                skilled_wage_coeffs[i][6]*experience[2]+
                skilled_wage_coeffs[i][7]*experience[i]**2+
                skilled_wage_coeffs[i][8]*at_least_one_exp+
                skilled_wage_coeffs[i][9]*quality+
                skilled_wage_coeffs[i][10]*total_exp)
            else:
                out[i]=(skilled_wage_coeffs[i][0]+skilled_wage_coeffs[i][1]+
                skilled_wage_coeffs[i][3]*GPA/100+
                skilled_wage_coeffs[i][4]*experience[0]+
                skilled_wage_coeffs[i][5]*experience[1]+
                skilled_wage_coeffs[i][6]*experience[2]+
                skilled_wage_coeffs[i][7]*experience[i]**2+
                skilled_wage_coeffs[i][8]*at_least_one_exp+
                skilled_wage_coeffs[i][9]*quality+
                skilled_wage_coeffs[i][10]*total_exp)

        # need to adjust wages        
        else:
            if dSTEM==0:
                out[i]=(skilled_wage_coeffs[i][0]+
                skilled_wage_coeffs[i][2]*GPA/100+
                skilled_wage_coeffs[i][4]*experience[0]+
                skilled_wage_coeffs[i][5]*experience[1]+
                skilled_wage_coeffs[i][6]*experience[2]-
                skilled_wage_coeffs[i][i+4]*experience[i]-
                skilled_wage_coeffs[i][i+4]**2/(4*skilled_wage_coeffs[i][7])+
                skilled_wage_coeffs[i][8]*at_least_one_exp+
                skilled_wage_coeffs[i][9]*quality+
                skilled_wage_coeffs[i][10]*total_exp)
            else:

                out[i]=(skilled_wage_coeffs[i][0]+skilled_wage_coeffs[i][1]+
                skilled_wage_coeffs[i][3]*GPA/100+
                skilled_wage_coeffs[i][4]*experience[0]+
                skilled_wage_coeffs[i][5]*experience[1]+
                skilled_wage_coeffs[i][6]*experience[2]-
                skilled_wage_coeffs[i][i+4]*experience[i]-
                skilled_wage_coeffs[i][i+4]**2/(4*skilled_wage_coeffs[i][7])+
                skilled_wage_coeffs[i][8]*at_least_one_exp+
                skilled_wage_coeffs[i][9]*quality+
                skilled_wage_coeffs[i][10]*total_exp)

    return out

# HARD CODED FOR 3 SECTORS
@jit(nopython=True)
def create_skilled_experience(skilled1,skilled2,skilled3,hp):
    out=np.zeros((len(skilled1),4),dtype=np.int64)
    for x in range(len(skilled1)):
        out[x]=np.array([int(skilled1[x]),int(skilled2[x]),
            int(skilled3[x]),int(hp[x])])
    return out

# contains a bunch of zeros for major, grade, and sectors 2 and 3.
# ordering is identical to skilled sector
# note that if it is for year 4, there needs to be a modified intercept already
# accounted for

@jit(nopython=True)
def elogwageUnskilled(unskilled_wage_coeffs,unskilled_experience,
    hp_experience):
    total_exp = unskilled_experience+hp_experience
    at_least_one_exp = 1 if unskilled_experience>0 else 0
    if (-2*unskilled_experience*unskilled_wage_coeffs[7]<=
        unskilled_wage_coeffs[4]):
        out=(unskilled_wage_coeffs[0]+
            unskilled_wage_coeffs[4]*unskilled_experience+
            unskilled_wage_coeffs[7]*unskilled_experience**2+
            unskilled_wage_coeffs[8]*at_least_one_exp+
            unskilled_wage_coeffs[10]*total_exp)
    else:
        out=(unskilled_wage_coeffs[0]-
            unskilled_wage_coeffs[4]**2/(4*unskilled_wage_coeffs[7])+
            unskilled_wage_coeffs[8]*at_least_one_exp+
            unskilled_wage_coeffs[10]*total_exp)
    return out

@jit(nopython=True)
def calculate_wage_shock(outcome,col_type,choice,state,skilled_wage_coeffs,
    unskilled_wage_coeffs,experience,unskilled_exp,dSTEM,GPA,quality,tdropout,
    year_four_intercept,year_four_exp,year_four_quadratic,year_four_year_1):
    out=np.zeros(outcome.shape[0])
    for x in range(outcome.shape[0]):
        if col_type[x]==2 and state[x]==7: # if skilled wage
            wage_vector=elogwage(skilled_wage_coeffs,
                experience[x],dSTEM[x],GPA[x],quality[x])
            # 4 is the offset due to STEM/nonSTEM/hp/unskilled
            out[x]=outcome[x]-wage_vector[choice[x]-4]
        elif col_type[x]==2:
            if tdropout[x]==4:
                unskilled_wage_coeffs_final = unskilled_wage_coeffs.copy()
                unskilled_wage_coeffs_final[0] = year_four_intercept

                unskilled_wage_coeffs_final[4] = year_four_exp
                unskilled_wage_coeffs_final[7] = year_four_quadratic
                unskilled_wage_coeffs_final[8] = year_four_year_1
                out[x]=outcome[x]-elogwageUnskilled(unskilled_wage_coeffs_final,
                    unskilled_exp[x],experience[x][3])
            else:
                out[x]=outcome[x]-elogwageUnskilled(unskilled_wage_coeffs,
                    unskilled_exp[x],experience[x][3])
    return out

@jit(nopython=True)
def calculate_wage_shock_skilled(outcome,col_type,choice,state,skilled_wage_coeffs,
    experience,dSTEM,GPA,quality):
    out=np.zeros(outcome.shape[0])
    for x in range(outcome.shape[0]):
        wage_vector=elogwage(skilled_wage_coeffs,
            experience[x],dSTEM[x],GPA[x],quality[x])
        # 4 is the offset due to STEM/nonSTEM/hp/unskilled
        out[x]=outcome[x]-wage_vector[choice[x]-4]

    return out

def MVNposterior(covariance,draws):
    dim=len(covariance)
    zscores=scipy.stats.norm.ppf(np.array(range(1,draws+1))/(draws+1))
    zscores_unskilled=scipy.stats.norm.ppf(np.array(range(1,draws))/(draws))
    base_draws=np.matrix.transpose(np.matrix(list(\
        itertools.product(zscores,repeat=(dim-1)))))
    meanterm=np.zeros((dim+1,dim),dtype=np.float64)
    covariance=np.matrix(covariance)
    covar=np.zeros((dim+1,dim,dim),dtype=np.float64)
    skilled_shocks=np.zeros((dim,draws**(dim-1),dim),dtype=np.float64)

    # x is the sector we observe
    for x in range(dim):
        # permute so that observed sector is in [1,1] slot in matrix
        permute=np.identity(dim)
        permute[0,0]=0
        permute[x,x]=0
        permute[x,0]=1
        permute[0,x]=1
        # calculate posterior matrices, and generate output
        permutecovar=np.matmul(np.matmul(permute,covariance),\
            np.matrix.transpose(permute))
        sigmax=permutecovar[0,0]
        sigmayx=permutecovar[1:dim,0]
        sigmaxy=permutecovar[0,1:dim]
        sigmay=permutecovar[1:dim,1:dim]
        posteriorcovar=sigmay-np.matmul(sigmayx,sigmaxy)/sigmax
        lmat=np.linalg.cholesky(posteriorcovar)
        meanout=np.append(0,np.asarray(sigmaxy/sigmax))
        meanout[0]=meanout[x]
        meanout[x]=0
        finalcovar=np.insert(posteriorcovar,0,np.array([0]*(dim-1)),0)
        finalcovar=np.insert(finalcovar,0,np.array([0]*dim),1)
        finalcovar=np.matmul(np.matmul(permute,finalcovar),\
            np.matrix.transpose(permute))
        meanterm[x+1]=meanout
        covar[x+1]=finalcovar
        shock_draws=np.insert(np.matmul(lmat,base_draws),0,\
            np.array([0]*base_draws.shape[1]),0)
        skilled_shocks[x]=np.array(
            np.matrix.transpose(np.matmul(permute,shock_draws)),
            dtype=np.float64)

    # add in case of home production (no posterior):
    meanterm[0]=np.zeros(dim)
    # shock term also causes the 'same' term to +1
    for x in range(dim):
        meanterm[x+1,x]=1
    covar[0]=covariance
    lmat=np.linalg.cholesky(covariance)
    baseDrawsHP=np.matrix.transpose(np.matrix(list(
        itertools.product(zscores_unskilled,repeat=(dim)))))
    hp_shocks=np.array(np.matrix.transpose(np.matmul(
        lmat,baseDrawsHP)),dtype=np.float64)
    return tuple([meanterm,covar,skilled_shocks,hp_shocks])

# Helper function for logit likelihood
# choiceflows is vector of flow utilities
# choice is the index of the choice
# returns the likelihood which is exp(choice)/sum(exp(all choices))
@jit(nopython=True)
def LogitLike(choiceflows,choice):
    red1=choiceflows-choiceflows[choice]
    pmax=np.max(red1)
    red2=red1-pmax
    like=np.exp(-pmax-np.log(np.sum(np.exp(red2))))
    return like

@jit(nopython=True)
def LogitLikeZero(other_choices):
    denom=np.sum(np.exp(other_choices))
    return 1/(1+denom)

@jit(nopython=True)
def LogitLikeNoHP(sector_flows,sector):
    choiceflows=np.concatenate((np.zeros(1),sector_flows))
    return LogitLike(choiceflows,sector)


# Take average of logit likelihoods, over each wage shock
# returns a LOG likelihood
# Logic for terminal case too
# simwageshocks is a tuple of matrices of shocks, 0 corresponding to HP
# meanterm is a sectors * sectors array
# skilled_choice corresponds directly to skilled sector (0 = HP)
# prior skilled is numbered 0 = skilled 1 ... 3 = HP

@jit(nopython=True)
def WageShockIntegrateNP(grad_horizon,sectors,time,skilled_choice,
    skilled_shock,Emax_func,sim_wage_shocks,experience,dSTEM,tGPA,
    meanterm,flows_penalized,gamma_p,beta,skilled_wage_coeffs,choose,quality,
    time_zero_flows_penalized,switch_costs,prior_skilled,zero_exp_penalty):
    sims=sim_wage_shocks.shape[0]
    partlike=np.empty(sims)
    full_exp=np.zeros(len(experience),dtype=np.int64)
    for i in range(len(experience)):
        full_exp[i]=experience[i]
    total_exp=time-5
    full_exp[sectors]=total_exp-np.sum(experience)

    if time==5:
        flows_final = time_zero_flows_penalized+zero_exp_penalty
    else:
        flows_final = flows_penalized.copy()
        if prior_skilled>=0 and prior_skilled<3:
            flows_final[prior_skilled]=(flows_final[prior_skilled]+
                switch_costs[prior_skilled])

        for i in range(sectors):
            if experience[i]==0:
                flows_final[i]=flows_final[i]+zero_exp_penalty[i]
    # home production, terminal case
    if skilled_choice==0 and time==(grad_horizon+4):
        ewage=elogwage(skilled_wage_coeffs,experience,dSTEM,tGPA,quality)
        choiceset=np.exp(ewage+sim_wage_shocks)*gamma_p+flows_final
        for sim in range(sims):
            partlike[sim]=LogitLikeZero(choiceset[sim,:])
        return np.log(np.mean(partlike))
    
    # # actual sector choice (requires posterior)
    elif time==(grad_horizon+4):
        ewage=elogwage(skilled_wage_coeffs,experience,dSTEM,tGPA,quality)
        wage_shock=sim_wage_shocks+meanterm*skilled_shock
        choiceset=np.exp(ewage+wage_shock)*gamma_p+flows_final
        for sim in range(sims):
            partlike[sim]=LogitLikeNoHP(choiceset[sim,:],skilled_choice)
        return np.log(np.mean(partlike))

    elif skilled_choice==0:
        next_payoffs=np.empty(sectors+1,dtype=np.float64)
        for i in range(sectors+1):
            new_exp=np.copy(full_exp)
            new_exp[i]=new_exp[i]+1
            next_payoffs[i]=beta*Emax_func[total_exp+1,
            simplexmap.combo_to_array(total_exp+1,new_exp,choose),i]
        ewage=elogwage(skilled_wage_coeffs,experience,dSTEM,tGPA,quality)
        utils=np.hstack((np.exp(ewage+sim_wage_shocks)*gamma_p+flows_final,
            np.zeros((sims,1))))
        choiceset=utils+next_payoffs
        for sim in range(sims):
            partlike[sim]=LogitLike(choiceset[sim,:],sectors)
        return np.log(np.mean(partlike))
    else:
        next_payoffs=np.empty(sectors+1,dtype=np.float64)
        # get next period discounted payoff
        for i in range(sectors+1):
            new_exp=np.copy(full_exp)
            new_exp[i]=new_exp[i]+1
            next_payoffs[i]=beta*Emax_func[total_exp+1,
            simplexmap.combo_to_array(total_exp+1,new_exp,choose),i]
        ewage=elogwage(skilled_wage_coeffs,experience,dSTEM,tGPA,quality)
        wage_shock=sim_wage_shocks+meanterm*skilled_shock
        utils=np.hstack((np.exp(ewage+wage_shock)*gamma_p+flows_final,
            np.zeros((sims,1))))
        choiceset=utils+next_payoffs

        # -1 here because HP is at the end
        for sim in range(sims):
            partlike[sim]=LogitLike(choiceset[sim,:],skilled_choice-1)
        return np.log(np.mean(partlike))



# choice likelihood when the wage is missing but the agent chooses a sector
# have to integrate over unobserved wage shocks (everything)
@jit(nopython=True)
def WageShockIntegrateNPNoWage(grad_horizon,sectors,time,skilled_choice,
    skilled_shock,Emax_func,sim_wage_shocks,experience,dSTEM,tGPA,
    meanterm,flows_penalized,gamma_p,beta,skilled_wage_coeffs,choose,quality,
    time_zero_flows_penalized,switch_costs,prior_skilled,zero_exp_penalty):
    sims=sim_wage_shocks.shape[0]
    partlike=np.empty(sims)
    full_exp=np.zeros(len(experience),dtype=np.int64)
    for i in range(len(experience)):
        full_exp[i]=experience[i]
    total_exp=time-5
    full_exp[sectors]=total_exp-np.sum(experience)

    # determine flow utilities
    if time==5:
        flows_final = time_zero_flows_penalized+zero_exp_penalty
    else:
        flows_final = flows_penalized.copy()
        if prior_skilled>=0 and prior_skilled<3:
            flows_final[prior_skilled]=(flows_final[prior_skilled]+
                switch_costs[prior_skilled])

        for i in range(sectors):
            if experience[i]==0:
                flows_final[i]=flows_final[i]+zero_exp_penalty[i]


    if time==(grad_horizon+4):
        ewage=elogwage(skilled_wage_coeffs,experience,dSTEM,tGPA,quality)
        choiceset=np.exp(ewage+sim_wage_shocks)*gamma_p+flows_final
        for sim in range(sims):
            partlike[sim]=LogitLikeNoHP(choiceset[sim,:],skilled_choice)
        return np.log(np.mean(partlike))
    else:
        next_payoffs=np.empty(sectors+1,dtype=np.float64)
        for i in range(sectors+1):
            new_exp=np.copy(full_exp)
            new_exp[i]=new_exp[i]+1
            next_payoffs[i]=beta*Emax_func[total_exp+1,
            simplexmap.combo_to_array(total_exp+1,new_exp,choose),i]
        ewage=elogwage(skilled_wage_coeffs,experience,dSTEM,tGPA,quality)
        utils=np.hstack((np.exp(ewage+sim_wage_shocks)*gamma_p+flows_final,
            np.zeros((sims,1))))
        choiceset=utils+next_payoffs
        for sim in range(sims):
            partlike[sim]=LogitLike(choiceset[sim,:],skilled_choice-1)
        return np.log(np.mean(partlike))


# flow_unskilled is penalized
@jit(nopython=True)
def ChoiceLikeUnskilled(grad_horizon,sectors,time,dropout_time,
    unskilled_choice,unskilled_exp,hp_exp,unskilled_wage_coeffs,logwage,
    Emax_func,gamma_p,beta,flow_unskilled,
    unskilledWageShocks,choose,unskilled_switch_cost,prior_skilled,
    year_four_intercept,year_four_flow,zero_exp_penalty,
    year_four_exp,year_four_quadratic,year_four_year_1,year_four_switching):

    unskilled_wage_coeffs_final = unskilled_wage_coeffs.copy()
    year_four_flow_penalized = year_four_flow - year_four_switching
    if dropout_time == 4:
        unskilled_wage_coeffs_final[0]=year_four_intercept

        unskilled_wage_coeffs_final[4] = year_four_exp
        unskilled_wage_coeffs_final[7] = year_four_quadratic
        unskilled_wage_coeffs_final[8] = year_four_year_1

        flow_unskilled = year_four_flow_penalized
        unskilled_switch_cost = year_four_switching

    # prior choice was unskilled
    if prior_skilled==4:
        flow_unskilled = flow_unskilled + unskilled_switch_cost

    # zero exp penalty
    if unskilled_exp==0:
        flow_unskilled = flow_unskilled + zero_exp_penalty[0]

    # unskilled work (no need to integrate, 3 is from choice)
    if unskilled_choice==1:

        # terminal case
        if time==(grad_horizon+4):
            return np.log(LogitLikeNoHP(np.array(
                [gamma_p*np.exp(logwage)])+flow_unskilled,1))
        else:
            work_cont=beta*Emax_func[time-dropout_time+1,
            simplexmap.combo_to_array(time-dropout_time+1,
                (unskilled_exp+1,time-unskilled_exp-dropout_time),choose),0]
            hp_flow=beta*Emax_func[time-dropout_time+1,
            simplexmap.combo_to_array(time-dropout_time+1,
                (unskilled_exp,time-unskilled_exp-dropout_time+1),choose),1]
            work_flow=work_cont+gamma_p*np.exp(logwage)+flow_unskilled
            return np.log(LogitLike(np.array([hp_flow,work_flow]),1))

    # hp (need to integrate)
    else:
        sims=len(unskilledWageShocks)
        partlike=np.empty(sims)

        if time==(grad_horizon+4):
            for sim in range(sims):
                other_flow=np.array([gamma_p*np.exp(
                    unskilledWageShocks[sim]+elogwageUnskilled(
                            unskilled_wage_coeffs_final,unskilled_exp,hp_exp))+
                flow_unskilled])
                partlike[sim]=LogitLikeZero(other_flow)
            return np.log(np.mean(partlike))
        else:
            work_cont=beta*Emax_func[time-dropout_time+1,
            simplexmap.combo_to_array(time-dropout_time+1,
                (unskilled_exp+1,time-unskilled_exp-dropout_time),
                choose),0]
            hp_flow=beta*Emax_func[time-dropout_time+1,
            simplexmap.combo_to_array(time-dropout_time+1,
                (unskilled_exp,time-unskilled_exp-dropout_time+1),
                choose),1]
            wage_det=elogwageUnskilled(unskilled_wage_coeffs_final,
                unskilled_exp,hp_exp)

            for sim in range(sims):
                work_flow=(work_cont+
                    gamma_p*np.exp(wage_det+unskilledWageShocks[sim])+
                    flow_unskilled)
                partlike[sim]=LogitLike(np.array([work_flow,hp_flow]),1)
            return np.log(np.mean(partlike))

# flow_unskilled is penalized
@jit(nopython=True)
def ChoiceLikeUnskilledNoWage(grad_horizon,sectors,time,dropout_time,
    unskilled_choice,unskilled_exp,hp_exp,unskilled_wage_coeffs,logwage,
    Emax_func,gamma_p,beta,flow_unskilled,
    unskilledWageShocks,choose,unskilled_switch_cost,prior_skilled,
    year_four_intercept,year_four_flow,zero_exp_penalty,
    year_four_exp,year_four_quadratic,year_four_year_1,year_four_switching):

    unskilled_wage_coeffs_final = unskilled_wage_coeffs.copy()
    year_four_flow_penalized = year_four_flow - year_four_switching
    if dropout_time == 4:
        unskilled_wage_coeffs_final[0]=year_four_intercept
        flow_unskilled = year_four_flow_penalized
        unskilled_switch_cost = year_four_switching


        unskilled_wage_coeffs_final[4] = year_four_exp
        unskilled_wage_coeffs_final[7] = year_four_quadratic
        unskilled_wage_coeffs_final[8] = year_four_year_1

    # prior choice was unskilled
    if prior_skilled==4:
        flow_unskilled = flow_unskilled + unskilled_switch_cost

    if unskilled_exp==0:
        flow_unskilled = flow_unskilled + zero_exp_penalty[0]

    # unskilled work (no need to integrate, 3 is from choice)
    if unskilled_choice==1:
        sims = len(unskilledWageShocks)
        partlike = np.empty(sims)

        # terminal case
        if time==(grad_horizon+4):
            for sim in range(sims):
                work_flow=(gamma_p*np.exp(unskilledWageShocks[sim]+
                    elogwageUnskilled(unskilled_wage_coeffs_final,unskilled_exp,
                        hp_exp))+flow_unskilled)
                partlike[sim]=LogitLike(np.array([work_flow,0]),0)
            return np.log(np.mean(partlike))
        else:
            for sim in range(sims):
                work_cont=beta*Emax_func[time-dropout_time+1,
                simplexmap.combo_to_array(time-dropout_time+1,
                    (unskilled_exp+1,time-unskilled_exp-dropout_time),choose),0]
                hp_flow=beta*Emax_func[time-dropout_time+1,
                simplexmap.combo_to_array(time-dropout_time+1,
                    (unskilled_exp,time-unskilled_exp-dropout_time+1),choose),1]
                work_flow=(work_cont+gamma_p*np.exp(
                    unskilledWageShocks[sim]+elogwageUnskilled(
                        unskilled_wage_coeffs_final,unskilled_exp,hp_exp))+
                flow_unskilled)
                partlike[sim]=LogitLike(np.array([hp_flow,work_flow]),1)
            return np.log(np.mean(partlike))


@jit(nopython=True)
def Egrade(year,major_params,exogChars):
    year_param=major_params[year+2]
    return (major_params[0]*exogChars[0]+major_params[1]*exogChars[1]+
        major_params[2]*exogChars[2]+year_param)

# Likelihood of grade observation, conditional on unobserved ability type
# Ability is (STEM, non-STEM)
# grade_params is a 8x2 vector by major:
#        [SAT_M, SAT_V, hs_GPA, lambda_1,lambda_2,lambda_3,lambda_4,sigma^2_m]
#        STEM comes first.
# exogChars is [SATM,SATV]

# check for STEM or nonSTEM
@jit(nopython=True)
def GradeLike(exogChars,time,STEM_choice,grade,grade_params,ability):
    if time==1 or time==2:
        var_idx = 7
    else:
        var_idx = 8
    if STEM_choice==1:
        loglike=(-0.5*np.log(grade_params[0][var_idx])-
            (grade/100-Egrade(time,grade_params[0],exogChars)-
                ability[0])**2/(2*grade_params[0][var_idx]))
    else:
        loglike=(-0.5*np.log(grade_params[1][var_idx])-
            (grade/100-Egrade(time,grade_params[1],exogChars)-
                ability[1])**2/(2*grade_params[1][var_idx]))
    return loglike

# current_year ranges 1 to 4
@jit(nopython=True)
def FutureGrade(current_year,currentGPA,dSTEM,exogChars,ability,
    grade_params,grade_quantiles):
    # STEM
    if dSTEM==1:
        if current_year==1 or current_year==2:
            sigma=100*grade_params[0][7]**0.5
        else:
            sigma=100*grade_params[0][8]**0.5
        majorParams=grade_params[0]
        majorAbility=ability[0]
    # nonSTEM
    else:
        if current_year==1 or current_year==2:
            sigma=100*grade_params[1][7]**0.5
        else:
            sigma=100*grade_params[1][8]**0.5
        majorParams=grade_params[1]
        majorAbility=ability[1]

    meanGrade=Egrade(current_year,majorParams,exogChars)+majorAbility

    # generate next grade, rounded to nearest 5, and top/bottom capped
    # at 0, 400 GPA
    semGrades=grade_quantiles*sigma+100*meanGrade
    for x in range(len(semGrades)):
        if semGrades[x]>400:
            semGrades[x]=400
        elif semGrades[x]<0:
            semGrades[x]=0

        nextGrades=np.zeros(len(semGrades))
    for x in range(len(semGrades)):
        nextGrades[x]=round_to_5((currentGPA*(current_year-1)+semGrades[x])/
            current_year)
    return nextGrades


# check if state==unskilled
# choice in hp/STEM/nonSTEM = no int
# iterate over all of the states

# MAPPING: school choice column: 0: STEM, 1: nonSTEM, 2: hp, 3: unskilled
# STATES: 0: t1STEM, 1: t1nonSTEM, 2: t3STEMReqMet, 3: t3STEMReqNotMet,
#         4: t3STEM, 5: t3nonSTEM, 6: t1


@jit(nopython=True)
def gpa_to_index(gpa):
    return int(gpa/5)

@jit(nopython=True)
def tgpa_to_index(gpa):
    return int((gpa-200)/10)

@jit(nopython=True)
def ChoiceLikeSchool(grad_horizon,school_choice,school_state,cum_GPA,
    exogChars,STEM_payouts,nonSTEM_payouts,grade_quantiles,dropout_payouts,
    Emax_func,Emax_STEM_zero,Emax_nonSTEM_zero,gamma_p,beta,flow_educ,
    tuition,ability,firstUnskilledDraws,LaborGradeInt,logwage,grade_params,
    year_four_first_draws,year_four_flow_penalized,last_choice,
    ed_switching_costs,univ_type_shifters,univ_type_num,grad_payoff):
    
    if univ_type_num>0:
        # STEM
        flow_STEM_penalized=(flow_educ[0]+
            univ_type_shifters[2*univ_type_num-2])
        # nonSTEM
        flow_nonSTEM_penalized=(flow_educ[1]+
            univ_type_shifters[2*univ_type_num-1])
    else:
        flow_STEM_penalized = flow_educ[0]
        flow_nonSTEM_penalized = flow_educ[1]

    flow_unskilled_penalized = flow_educ[2]
    
    if school_choice!=3:
     # will need to integrate over unskilled wage draw
     # and enumerate over all possible outcomes if school_choice!=3:

        partlike=np.empty(len(firstUnskilledDraws),dtype=np.float64)

        # t4 STEM option only
        if school_state==4:
            t4grades=FutureGrade(4,cum_GPA,1,exogChars,ability,grade_params,
                grade_quantiles)
            payout=np.empty(len(t4grades))
            for idx in range(len(t4grades)):
                grade=t4grades[idx]
                if grade<200:
                    payout[idx]=dropout_payouts[3,0]
                else:
                    rounding=takeClosest(grade,LaborGradeInt)
                    payout[idx]=(STEM_payouts[tgpa_to_index(rounding)] + 
                        grad_payoff)

            utilitySTEM=(flow_STEM_penalized+ed_switching_costs[0]-
                gamma_p*tuition+beta*np.mean(payout))
            utilityHP=beta*dropout_payouts[3,0]
            for sim in range(len(year_four_first_draws)):
                utilityUnskilled=(year_four_flow_penalized+
                    gamma_p*year_four_first_draws[sim]+
                    beta*dropout_payouts[3,1])
                if school_choice==2:
                    choice=0
                elif school_choice==0:
                    choice=1
                partlike[sim]=LogitLike(np.array([utilityHP,utilitySTEM,
                    utilityUnskilled],dtype=np.float64),choice)
            return np.log(np.mean(partlike))

        # t4 nonSTEM option only
        elif school_state==5:
            t4grades=FutureGrade(4,cum_GPA,0,exogChars,ability,grade_params,
                grade_quantiles)
            payout=np.empty(len(t4grades))
            for idx in range(len(t4grades)):
                grade=t4grades[idx]
                if grade<200:
                    payout[idx]=dropout_payouts[3,0]
                else:
                    rounding=takeClosest(grade,LaborGradeInt)
                    payout[idx]=(nonSTEM_payouts[tgpa_to_index(rounding)] + 
                        grad_payoff)

            utilitynonSTEM=(flow_nonSTEM_penalized+ed_switching_costs[1]-
                gamma_p*tuition+beta*np.mean(payout))
            utilityHP=beta*dropout_payouts[3,0]
            for sim in range(len(year_four_first_draws)):
                utilityUnskilled=(year_four_flow_penalized+
                    gamma_p*year_four_first_draws[sim]+
                    beta*dropout_payouts[3,1])
                if school_choice==2:
                    choice=0
                elif school_choice==1:
                    choice=1
                partlike[sim]=LogitLike(np.array([utilityHP,utilitynonSTEM,
                    utilityUnskilled],dtype=np.float64),choice)
            return np.log(np.mean(partlike))

        # t=3, STEM req met
        elif school_state==2:
            t3STEMgrades=FutureGrade(3,cum_GPA,1,exogChars,ability,
                grade_params,grade_quantiles)
            t3nonSTEMgrades=FutureGrade(3,cum_GPA,0,exogChars,ability,
                grade_params,grade_quantiles)
            STEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t3STEMgrades)):
                STEM_exante[idx]=(Emax_func[4][
                    gpa_to_index(t3STEMgrades[idx])][0])

            nonSTEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t3nonSTEMgrades)):
                nonSTEM_exante[idx]=(Emax_func[5][
                    gpa_to_index(t3nonSTEMgrades[idx])][1])

            # look at last choice
            if last_choice == 0:
                utilitySTEM=(beta*np.mean(STEM_exante)+flow_STEM_penalized-
                    gamma_p*tuition + ed_switching_costs[0])
                utilitynonSTEM=(beta*np.mean(nonSTEM_exante)+
                    flow_nonSTEM_penalized - gamma_p*tuition)
            else:
                utilitySTEM=(beta*np.mean(STEM_exante)+flow_nonSTEM_penalized-
                    gamma_p*tuition)
                utilitynonSTEM=(beta*np.mean(nonSTEM_exante)+
                    flow_nonSTEM_penalized-gamma_p*tuition +
                    ed_switching_costs[1])                
           
            utilityHP=beta*dropout_payouts[2,0]

            for sim in range(len(firstUnskilledDraws)):
                utilityUnskilled=(flow_unskilled_penalized+
                    gamma_p*firstUnskilledDraws[sim]+
                    beta*dropout_payouts[2,1])
                partlike[sim]=LogitLike(np.array([utilitySTEM,
                    utilitynonSTEM,utilityHP,utilityUnskilled],
                    dtype=np.float64),school_choice)
            return np.log(np.mean(partlike))
            
        # t= 3, STEM req not met
        # means the student MUST have chosen non-STEM at t=2 (and t=1)
        elif school_state==3:

            t3nonSTEMgrades=FutureGrade(3,cum_GPA,0,exogChars,ability,
                grade_params,grade_quantiles)

            nonSTEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t3nonSTEMgrades)):
                nonSTEM_exante[idx]=(Emax_func[5][
                    gpa_to_index(t3nonSTEMgrades[idx])][1])

            utilitynonSTEM=(beta*np.mean(nonSTEM_exante)+
                flow_nonSTEM_penalized-gamma_p*tuition + 
                ed_switching_costs[1])
            utilityHP=beta*dropout_payouts[2,0]

            for sim in range(len(firstUnskilledDraws)):
                utilityUnskilled=(flow_unskilled_penalized+
                    gamma_p*firstUnskilledDraws[sim]+
                    beta*dropout_payouts[2,1])
                partlike[sim]=LogitLike(np.array([utilitynonSTEM,
                    utilityHP,utilityUnskilled],
                    dtype=np.float64),school_choice-1)
            return np.log(np.mean(partlike))

        # t1 STEM. Means t=1 prior choice MUST have been STEM
        elif school_state==0:

            t2STEMgrades=FutureGrade(2,cum_GPA,1,exogChars,ability,
                grade_params,grade_quantiles)
            t2nonSTEMgrades=FutureGrade(2,cum_GPA,0,exogChars,ability,
                grade_params,grade_quantiles)

            STEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t2STEMgrades)):
                STEM_exante[idx]=(Emax_func[2][
                    gpa_to_index(t2STEMgrades[idx])][0])

            utilitySTEM=(beta*np.mean(STEM_exante)+flow_STEM_penalized-
            gamma_p*tuition + ed_switching_costs[0])

            nonSTEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t2nonSTEMgrades)):
                nonSTEM_exante[idx]=(Emax_func[2][
                    gpa_to_index(t2nonSTEMgrades[idx])][1])

            utilitynonSTEM=(beta*np.mean(nonSTEM_exante)+
                flow_nonSTEM_penalized-gamma_p*tuition)
           
            utilityHP=beta*dropout_payouts[1,0]

            for sim in range(len(firstUnskilledDraws)):
                utilityUnskilled=(flow_unskilled_penalized+
                    gamma_p*firstUnskilledDraws[sim]+
                    beta*dropout_payouts[1,1])
                partlike[sim]=LogitLike(np.array([utilitySTEM,
                    utilitynonSTEM,utilityHP,utilityUnskilled],
                    dtype=np.float64),school_choice)
            return np.log(np.mean(partlike))

        # no STEM in t1, so STEM req is not met if student chooses nonSTEM
        # prior choice MUST have been non-STEM
        elif school_state==1:

            t2STEMgrades=FutureGrade(2,cum_GPA,1,exogChars,ability,
                grade_params,grade_quantiles)
            t2nonSTEMgrades=FutureGrade(2,cum_GPA,0,exogChars,ability,
                grade_params,grade_quantiles)

            STEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t2STEMgrades)):
                STEM_exante[idx]=(Emax_func[2][
                    gpa_to_index(t2STEMgrades[idx])][0])

            utilitySTEM=(beta*np.mean(STEM_exante)+flow_STEM_penalized-
            gamma_p*tuition)

            nonSTEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t2nonSTEMgrades)):
                nonSTEM_exante[idx]=(Emax_func[3][
                    gpa_to_index(t2nonSTEMgrades[idx])][1])

            utilitynonSTEM=(beta*np.mean(nonSTEM_exante)+
                flow_nonSTEM_penalized-gamma_p*tuition+
                ed_switching_costs[1])
           
            utilityHP=beta*dropout_payouts[1,0]

            for sim in range(len(firstUnskilledDraws)):
                utilityUnskilled=(flow_unskilled_penalized+
                    gamma_p*firstUnskilledDraws[sim]+
                    beta*dropout_payouts[1,1])
                partlike[sim]=LogitLike(np.array([utilitySTEM,
                    utilitynonSTEM,utilityHP,utilityUnskilled],
                    dtype=np.float64),school_choice)
            return np.log(np.mean(partlike))

        # t=1, note that these ex-ante value functions were already
        # calculated
        elif school_state==6:
            utilitySTEM=Emax_STEM_zero
            utilitynonSTEM=Emax_nonSTEM_zero
           
            utilityHP=beta*dropout_payouts[0,0]
            for sim in range(len(firstUnskilledDraws)):
                utilityUnskilled=(flow_unskilled_penalized+
                    gamma_p*firstUnskilledDraws[sim]+
                    beta*dropout_payouts[0,1])
                partlike[sim]=LogitLike(np.array([utilitySTEM,
                    utilitynonSTEM,utilityHP,utilityUnskilled],
                    dtype=np.float64),school_choice)
            return np.log(np.mean(partlike))

    # already observe the unskilled wage shock so no need to integrate
    else:
        if school_state==4:
            t4grades=FutureGrade(4,cum_GPA,1,exogChars,ability,grade_params,
                grade_quantiles)
            payout=np.empty(len(t4grades))
            for idx in range(len(t4grades)):
                grade=t4grades[idx]
                if grade<200:
                    payout[idx]=dropout_payouts[3,0]
                else:
                    rounding=takeClosest(grade,LaborGradeInt)
                    payout[idx]=STEM_payouts[tgpa_to_index(rounding)]

            utilitySTEM=(flow_STEM_penalized-gamma_p*tuition+
                beta*np.mean(payout)+ed_switching_costs[0])

            utilityHP=beta*dropout_payouts[3,0]

            if np.isnan(logwage):
                logwage=np.log(np.mean(firstUnskilledDraws))

            utilityUnskilled=(year_four_flow_penalized+gamma_p*np.exp(logwage)+
                beta*dropout_payouts[3,1])
            return np.log(LogitLike(np.array([utilityUnskilled,
                utilitySTEM,utilityHP]),0))

        elif school_state==5:

            t4grades=FutureGrade(4,cum_GPA,0,exogChars,ability,grade_params,
                grade_quantiles)
            payout=np.empty(len(t4grades))
            for idx in range(len(t4grades)):
                grade=t4grades[idx]
                if grade<200:
                    payout[idx]=dropout_payouts[3,0]
                else:
                    rounding=takeClosest(grade,LaborGradeInt)
                    payout[idx]=nonSTEM_payouts[tgpa_to_index(rounding)]

            utilitynonSTEM=(flow_nonSTEM_penalized-gamma_p*tuition+
                beta*np.mean(payout)+ed_switching_costs[1])
            utilityHP=beta*dropout_payouts[3,0]

            if np.isnan(logwage):
                logwage=np.log(np.mean(firstUnskilledDraws))

            utilityUnskilled=(year_four_flow_penalized+gamma_p*np.exp(logwage)+
                beta*dropout_payouts[3,1])
            return np.log(LogitLike(np.array([utilityUnskilled,
                utilitynonSTEM,utilityHP]),0))

        elif school_state==2:
            t3STEMgrades=FutureGrade(3,cum_GPA,1,exogChars,ability,
                grade_params,grade_quantiles)
            t3nonSTEMgrades=FutureGrade(3,cum_GPA,0,exogChars,ability,
                grade_params,grade_quantiles)
            STEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t3STEMgrades)):
                STEM_exante[idx]=(Emax_func[4][
                    gpa_to_index(t3STEMgrades[idx])][0])


            nonSTEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t3nonSTEMgrades)):
                nonSTEM_exante[idx]=(Emax_func[5][
                    gpa_to_index(t3nonSTEMgrades[idx])][1])


            # look at last choice
            if last_choice == 0:
                utilitySTEM=(beta*np.mean(STEM_exante)+flow_STEM_penalized-
                    gamma_p*tuition + ed_switching_costs[0])
                utilitynonSTEM=(beta*np.mean(nonSTEM_exante)+
                    flow_nonSTEM_penalized-gamma_p*tuition)
            else:
                utilitySTEM=(beta*np.mean(STEM_exante)+flow_STEM_penalized-
                    gamma_p*tuition)
                utilitynonSTEM=(beta*np.mean(nonSTEM_exante)+
                    flow_nonSTEM_penalized-gamma_p*tuition +
                    ed_switching_costs[1])         

            utilityHP=beta*dropout_payouts[2,0]

            if np.isnan(logwage):
                logwage=np.log(np.mean(firstUnskilledDraws))

            utilityUnskilled=(flow_unskilled_penalized+gamma_p*np.exp(logwage)+
                beta*dropout_payouts[2,1])
            return np.log(LogitLike(np.array([utilityUnskilled,utilitySTEM,
                utilitynonSTEM,utilityHP]),0))

        elif school_state==3:

            t3nonSTEMgrades=FutureGrade(3,cum_GPA,0,exogChars,ability,
                grade_params,grade_quantiles)

            nonSTEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t3nonSTEMgrades)):
                nonSTEM_exante[idx]=(Emax_func[5][
                    gpa_to_index(t3nonSTEMgrades[idx])][1])

            utilitynonSTEM=(beta*np.mean(nonSTEM_exante)+
                flow_nonSTEM_penalized-gamma_p*tuition +
                ed_switching_costs[1])
            utilityHP=beta*dropout_payouts[2,0]

            if np.isnan(logwage):
                logwage=np.log(np.mean(firstUnskilledDraws))

            utilityUnskilled=(flow_unskilled_penalized+gamma_p*np.exp(logwage)+
                beta*dropout_payouts[2,1])
            return np.log(LogitLike(np.array([utilityUnskilled,
                utilityHP,utilitynonSTEM]),0))

        # t1STEM
        elif school_state==0:

            t2STEMgrades=FutureGrade(2,cum_GPA,1,exogChars,ability,
                grade_params,grade_quantiles)
            t2nonSTEMgrades=FutureGrade(2,cum_GPA,0,exogChars,ability,
                grade_params,grade_quantiles)

            STEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t2STEMgrades)):
                STEM_exante[idx]=(Emax_func[2][
                    gpa_to_index(t2STEMgrades[idx])][0])

            utilitySTEM=(beta*np.mean(STEM_exante)+flow_STEM_penalized-
                gamma_p*tuition+ ed_switching_costs[0])

            nonSTEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t2nonSTEMgrades)):
                nonSTEM_exante[idx]=(Emax_func[2][
                    gpa_to_index(t2nonSTEMgrades[idx])][1])

            utilitynonSTEM=(beta*np.mean(nonSTEM_exante)+
                flow_nonSTEM_penalized-gamma_p*tuition)
           
            utilityHP=beta*dropout_payouts[1,0]

            if np.isnan(logwage):
                logwage=np.log(np.mean(firstUnskilledDraws))

            utilityUnskilled=(flow_unskilled_penalized+gamma_p*np.exp(logwage)+
                beta*dropout_payouts[1,1])

            return np.log(LogitLike(np.array([utilityUnskilled,utilityHP,
                utilitySTEM,utilitynonSTEM]),0))

        # t1 nonSTEM
        elif school_state==1:

            t2STEMgrades=FutureGrade(2,cum_GPA,1,exogChars,ability,
                grade_params,grade_quantiles)
            t2nonSTEMgrades=FutureGrade(2,cum_GPA,0,exogChars,ability,
                grade_params,grade_quantiles)

            STEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t2STEMgrades)):
                STEM_exante[idx]=(Emax_func[2][
                    gpa_to_index(t2STEMgrades[idx])][0])
            utilitySTEM=(beta*np.mean(STEM_exante)+flow_STEM_penalized-
                gamma_p*tuition)
            nonSTEM_exante=np.zeros(len(grade_quantiles),dtype=np.float64)
            for idx in range(len(t2nonSTEMgrades)):
                nonSTEM_exante[idx]=(Emax_func[3][
                    gpa_to_index(t2nonSTEMgrades[idx])][1])

            utilitynonSTEM=(beta*np.mean(nonSTEM_exante)+
                flow_nonSTEM_penalized-gamma_p*tuition+
                ed_switching_costs[1])
            utilityHP=beta*dropout_payouts[1,0]

            if np.isnan(logwage):
                logwage=np.log(np.mean(firstUnskilledDraws))

            utilityUnskilled=(flow_unskilled_penalized+gamma_p*np.exp(logwage)+
                beta*dropout_payouts[1,1])
            return np.log(LogitLike(np.array([utilityUnskilled,utilityHP,
                utilitynonSTEM,utilitySTEM]),0))

        elif school_state==6:

            utilitySTEM=Emax_STEM_zero
            utilitynonSTEM=Emax_nonSTEM_zero
            utilityHP=beta*dropout_payouts[0,0]

            if np.isnan(logwage):
                logwage=np.log(np.mean(firstUnskilledDraws))

            utilityUnskilled=(flow_unskilled_penalized+gamma_p*np.exp(logwage)+
                beta*dropout_payouts[0,1])
            return np.log(LogitLike(np.array([utilityHP,utilitySTEM,
                utilitynonSTEM,utilityUnskilled]),3))


def LogLikeSkilled(DFData,SATTuition,college_values,grad_horizon,sectors,gamma_p,beta,
    ability,flows_penalized,wage_coeffs_full,skilled_wage_covar,unskilled_var,
    grade_params,normReps,simReps,LaborGradeRange,final_size,
    switch_costs_skilled,zero_exp_penalty,normReps_later=2,horizon=20,return_array=False):
    """
        Log-likelihood subset for skilled labor observations
        Log-likelihood is approximated for the various integrals 
        Args:
            DFData (DataFrame): data set. Needs to be preprocessed with
                PreProcess(DF), and set_index('id')
            SATTuition (list): list of tuples of exogenous endowments and
                tuitions that affect educational payoff
            grad_horizon (int): Years of work post-college graduation
            sectors (int): number of skilled sectors
            gamma_p (float): pref. for money
            beta (float): discount factor
            ability ([float]): list of unobserved ability [A_S,A_N]
            flows ([float]): list of flow utilities (9 terms)
                flows_penalized+[flow_unskilled_penalized,flowSTEM,flownonSTEM]+
                time_zero_flows_penalized
            wage_coeffs_full ([[float]]): nested list of log-wage coefficients
                skilled, then unskilled last. wage coeffs are of form
                [intercept, STEM, G_N, G_S, x1, x2, x3, ownsq, >1, quality]
            skilled_wage_covar ([[float]]): nested list of covariance of
                log-wage shocks in the skiled sector
            unskilled_var ([[float]]): variance of unskilled wage shocks,
                nested inside double list
            grade_params ([[float]]): [SAT_M, SAT_V, lambda_1,lambda_2,lambda_3,
                lambda_4,sigma^2_m] for STEM, non-STEM
            normReps (int): sectors^normReps simulations of log-wage shocks
            simReps (int): number of simulations of unskilled wage shocks
            LaborGradeRange (np.array): grades to use to discretize graduation
                outcomes. Segmenting of GPA from 2.0 to 4.0. Education is always
                assuming 0.1, but I will interpolate payouts over labor mkt

        Returns:
            float: log-likelihood

    """

    # initialize all the variables
    choose=simplexmap.pascal(grad_horizon+4,sectors)
    unskilled_wage_coeffs=wage_coeffs_full[-1]
    skilled_wage_coeffs=wage_coeffs_full[:-1]
    skilled_flows_penalized=flows_penalized[0:sectors]
    time_zero_flows_penalized = flows_penalized[(sectors+3):(2*sectors+3)]
    flowUnskilled=flows_penalized[sectors]
    flowSTEM=flows_penalized[sectors+1]
    flownonSTEM=flows_penalized[sectors+2]
    LaborGradeInt=np.array([int(x) for x in LaborGradeRange*100])
    LaborFinal=np.linspace(200,400,21,dtype=np.int64)

    wage_coeffs=wage_coeffs_full[:-1]
    # Generate Emax for entire population
    # This is just EmaxLink.py
    # Generate shocks to be used in approximating normal integrals

    zscores=scipy.stats.norm.ppf(np.array(range(1,normReps+1))/(normReps+1))
    num_quantiles=20
    norm_quantiles=scipy.stats.norm.ppf(
        np.array(range(1,num_quantiles))/num_quantiles)

    base_draws=np.matrix.transpose(np.matrix(list(
        itertools.product(zscores,repeat=(sectors)))))
    lmat=np.linalg.cholesky(skilled_wage_covar)
    wage_shocks=np.array(np.transpose(np.matmul(lmat,base_draws)))

    STEM_payouts_raw=np.zeros(11,dtype=np.float64)
    nonSTEM_payouts_raw=np.zeros(11,dtype=np.float64)

    # HARD CODED (220 refers to total combinations of exp at end)

    zscores=scipy.stats.norm.ppf(np.array(range(1,normReps_later+1))/
        (normReps_later+1))

    base_draws=np.matrix.transpose(np.matrix(list(
        itertools.product(zscores,repeat=(sectors)))))
    lmat=np.linalg.cholesky(skilled_wage_covar)
    wageshocks_later=np.array(np.transpose(np.matmul(lmat,base_draws)))

    skilled_Emax = np.zeros((len(college_values),grad_horizon,final_size,sectors+1),
        dtype=np.float64)
    for idx,x in enumerate(college_values):
        quality=x[0]
        dSTEM=x[1]
        grade=x[2]/100
        emax_solve=EmaxLaborFunctionsJIT(horizon,grad_horizon-1,
        gamma_p,beta,wage_coeffs,dSTEM,
        grade,skilled_flows_penalized,wage_shocks,choose,quality,
        time_zero_flows_penalized,switch_costs_skilled,zero_exp_penalty,
        wageshocks_later)
        emax_solve.solveLabor()
        skilled_Emax[idx]=emax_solve.EmaxList


    # skilled_Emax=np.zeros((2*11*2,grad_horizon,final_size,sectors+1))
    # for quality in range(2):
    #     for idx,grade in enumerate(LaborGradeRange):
    #         stem=EmaxLaborFunctionsJIT(grad_horizon-1,gamma_p,beta,wage_coeffs,
    #             1,grade,flows_penalized,wage_shocks,choose,quality,
    #             time_zero_flows_penalized,switch_costs_skilled)
    #         stem.solveLabor()
    #         nonstem=EmaxLaborFunctionsJIT(grad_horizon-1,gamma_p,beta,
    #             wage_coeffs,0,grade,flows_penalized,wage_shocks,choose,quality,
    #             time_zero_flows_penalized,switch_costs_skilled)
    #         nonstem.solveLabor()
    #         skilled_Emax[idx+11+22*quality]=stem.EmaxList
    #         skilled_Emax[idx+22*quality]=nonstem.EmaxList
    #         STEM_payouts_raw[idx]=stem.EmaxList[0,0,0]
    #         nonSTEM_payouts_raw[idx]=nonstem.EmaxList[0,0,0]

    # Fully solve the unskilled labor market, over the 4 dropout times
    dropout_payouts=np.zeros((4,2),dtype=np.float64)
    unskilled_reps=20
    unskilled_wage_shocks=np.array(np.transpose(np.matrix(scipy.stats.norm.ppf(
        np.array(range(1,unskilled_reps+1))/
        (unskilled_reps+1)))))*unskilled_var[0][0]**0.5


    # Solve education by extracting each unique tuition and generating an
    # Emax function for that.

    flow_educ=np.array([flowSTEM,flownonSTEM,flowUnskilled],dtype=np.float64)

    unskilled_mean=wage_coeffs_full[-1][0]
    unskilled_meanvar=np.array((unskilled_mean,unskilled_var[0][0]),
        dtype=np.float64)
    ed_Emax=np.zeros((len(SATTuition),6,81),dtype=np.float64)
    STEM1=np.zeros(len(SATTuition),dtype=np.float64)
    nonSTEM1=np.zeros(len(SATTuition),dtype=np.float64)




    # this part is hard coded for the number of sectors
    skilled_experience=create_skilled_experience(np.array(DFData.skilled1),
        np.array(DFData.skilled2),np.array(DFData.skilled3),
        np.array(DFData.hp))



    wage_shock=calculate_wage_shock_skilled(np.array(DFData.outcome),
        np.array(DFData.col_type),np.array(DFData.numeric_choice),
        np.array(DFData.numeric_state),skilled_wage_coeffs,
        skilled_experience,np.array(DFData.dSTEM),np.array(DFData.tGPA),
        np.array(DFData.quality))

    # Calculates likelihood of labor market decisions
    # This is the product of the likelihood of each wage offer
    # and the likelihood of making that choice, integrated over a conditional
    # wage offer of everything else
    # Integrating over a multivariate normal is done via simulation at
    # a number of points, given by cWagePoints

    # returns wage log-likelihood for all of an individual's observations
    # ignores normalizations like 1/sqrt(2pi)

    # calculates 2 likelihoods and returns a 2xN numpy array
    


    # run list of posteriors to integrate wages over
    (meanterm,covar,skilled_shocks,hp_wage_shocks)=(
        MVNposterior(skilled_wage_covar,4))
    skilled_shocks_list=[x for x in skilled_shocks]
    skilled_wage_shocks=tuple(skilled_shocks_list)




    # Skilled Choice likelihood
    # For each wage observation, the choice likelihood is a logit
    # conditional on the distribution of every other wage observation
    # cases differ for home production (no posterior)
    # vs non-posterior


    unskilledWageShocks=(np.transpose(scipy.stats.norm.ppf(
        (np.array(range(simReps))+1)/(simReps+1))*(unskilled_var[0][0])**0.5))
    firstUnskilledDraws=np.exp(unskilled_wage_coeffs[0]+unskilledWageShocks)

    # Unskilled choice likelihood
    # HP requires integration over unobserved unskilled wage shocks
    # condition: state=='unskilled' and choice=='unskilled'
    # dropout_time = 



    # Need to round this to coarseness of the rounding dataset
    # returns an array of the future grades the individual

    num_grades=20
    grade_quantiles=scipy.stats.norm.ppf(
        np.array(range(1,num_grades))/num_grades)


    out=calculate_likelihood_skilled(grad_horizon,sectors,np.array(DFData.time),
        np.array(DFData.numeric_choice),np.array(DFData.numeric_state),
        np.array(DFData.cumulativeGPA),np.array(DFData.tdropout),
        np.array(DFData.A_N),np.array(DFData.A_S),np.array(DFData.SAT_M),
        np.array(DFData.SAT_V),np.array(DFData.hs_GPA),np.array(DFData.tuition),
        None, None, grade_quantiles,
        dropout_payouts, wage_shock,
        skilled_wage_shocks,hp_wage_shocks,skilled_experience,
        np.array(DFData.outcome),
        np.array(DFData.dSTEM),np.array(DFData.tGPA),meanterm,
        skilled_Emax,None,None,
        np.array(DFData.skilled_emax_mapping_abridged),None,None,
        skilled_flows_penalized,skilled_wage_covar,
        gamma_p,beta,skilled_wage_coeffs,
        choose,grade_params,STEM1,nonSTEM1,
        LaborGradeInt,np.array(DFData.quality),time_zero_flows_penalized,
        np.array(DFData.prior_skilled),switch_costs_skilled,zero_exp_penalty)
    if return_array:
        return out
    return np.sum(out)
    # return [out,wage_shock,skilled_experience,STEM_payouts,nonSTEM_payouts,
    # unskilled_Emax,ed_Emax, skilled_Emax]

@jit(nopython=True)
def calculate_likelihood_skilled(grad_horizon,sectors,time,choice,state,cum_GPA,
    tdropout,A_N,A_S,SAT_M,SAT_V,hs_GPA,tuition,
    STEM_payouts, nonSTEM_payouts, grade_quantiles,
    dropout_payouts, wage_shock,skilled_wage_shocks,hp_wage_shocks,
    experience,outcome,
    dSTEM,tGPA,meanterm,skilled_Emax,ed_Emax,
    ed_emax_mapping, skilled_emax_mapping,
    flowUnskilled,flowSchool,
    flows_penalized,skilled_wage_covar,
    gamma_p,beta,skilled_wage_coeffs,choose,grade_params,STEM1,nonSTEM1,
    LaborGradeInt,quality,time_zero_flows_penalized,prior_skilled,
    switch_costs_skilled,zero_exp_penalty):

    length=time.shape[0]
    # first column is choice likelihood, second is obs likelihood
    out=np.zeros((length,2),dtype=np.float64)
    for x in range(length):
        # Skilled sector (WageLikeSkilled)
        # Skilled sector chosen
        if choice[x]!=2:

            choose_sector=int(choice[x]-3)

            # if wage data is missing, need to integrate over unobserved
            # wage draw
            if np.isnan(wage_shock[x]):
                out[x,0]=WageShockIntegrateNPNoWage(grad_horizon,sectors,
                    time[x],choose_sector,wage_shock[x],
                    skilled_Emax[skilled_emax_mapping[x]],
                    skilled_wage_shocks[choose_sector-1],
                    experience[x],dSTEM[x],tGPA[x],meanterm[choose_sector],
                    flows_penalized,gamma_p,beta,skilled_wage_coeffs,choose,
                    quality[x],time_zero_flows_penalized,switch_costs_skilled,
                    prior_skilled[x],zero_exp_penalty)
            else:    
                out[x,1]=(-0.5*np.log(
                    skilled_wage_covar[choose_sector-1,choose_sector-1])-
                (wage_shock[x])**2/
                (2*skilled_wage_covar[choose_sector-1,choose_sector-1]))
                out[x,0]=WageShockIntegrateNP(grad_horizon,sectors,time[x],
                    choose_sector,wage_shock[x],
                    skilled_Emax[skilled_emax_mapping[x]],
                    skilled_wage_shocks[choose_sector-1],
                    experience[x],dSTEM[x],tGPA[x],meanterm[choose_sector],
                    flows_penalized,gamma_p,beta,skilled_wage_coeffs,choose,
                    quality[x],time_zero_flows_penalized,switch_costs_skilled,
                    prior_skilled[x],zero_exp_penalty)

        # HP chosen
        else:
            out[x,0]=WageShockIntegrateNP(grad_horizon,sectors,time[x],
                0,wage_shock[x],
                skilled_Emax[skilled_emax_mapping[x]],
                hp_wage_shocks,
                experience[x],dSTEM[x],tGPA[x],np.zeros(3),
                flows_penalized,gamma_p,beta,skilled_wage_coeffs,choose,
                quality[x],time_zero_flows_penalized,switch_costs_skilled,
                prior_skilled[x],zero_exp_penalty)

    return out


def LogLikeUnskilled(DFData,SATTuition,grad_horizon,sectors,gamma_p,beta,
    ability,flows,wage_coeffs_full,skilled_wage_covar,unskilled_var,
    grade_params,normReps,simReps,LaborGradeRange,final_size,
    unskilled_switch_cost,year_four_intercept,year_four_flow,zero_exp_penalty,
    year_four_exp,year_four_quadratic,year_four_year_1,
    year_four_switching,return_array=False):

    """
        Log-likelihood subset for unskilled labor observations
        Log-likelihood is approximated for the various integrals 
        Args:
            DFData (DataFrame): data set. Needs to be preprocessed with
                PreProcess(DF), and set_index('id')
            SATTuition (list): list of tuples of exogenous endowments and
                tuitions that affect educational payoff
            grad_horizon (int): Years of work post-college graduation
            sectors (int): number of skilled sectors
            gamma_p (float): pref. for money
            beta (float): discount factor
            ability ([float]): list of unobserved ability [A_S,A_N]
            flows ([float]): list of flow utilities
                flows_penalized+[flow_unskilled_penalized,flowSTEM,flownonSTEM]
            wage_coeffs_full ([[float]]): nested list of log-wage coefficients
                skilled, then unskilled last. wage coeffs are of form
                [intercept,M,G,MxG,x,x^2]
            skilled_wage_covar ([[float]]): nested list of covariance of
                log-wage shocks in the skiled sector
            unskilled_var ([[float]]): variance of unskilled wage shocks,
                nested inside double list
            grade_params ([[float]]): [SAT_M, SAT_V, lambda_1,lambda_2,lambda_3,
                lambda_4,sigma^2_m] for STEM, non-STEM
            normReps (int): sectors^normReps simulations of log-wage shocks
            simReps (int): number of simulations of unskilled wage shocks
            LaborGradeRange (np.array): grades to use to discretize graduation
                outcomes. Segmenting of GPA from 2.0 to 4.0. Education is always
                assuming 0.1, but I will interpolate payouts over labor mkt

        Returns:
            float: log-likelihood

    """

    # initialize all the variables
    choose=simplexmap.pascal(grad_horizon+4,sectors)
    unskilled_wage_coeffs=wage_coeffs_full[-1]
    skilled_wage_coeffs=wage_coeffs_full[:-1]
    flows_penalized=flows[0:sectors]
    flowUnskilled=flows[sectors]
    flowSTEM=flows[sectors+1]
    flownonSTEM=flows[sectors+2]
    flows_educ=[flowSTEM,flownonSTEM,flowUnskilled]
    flowSchool=np.array([flowUnskilled,flowSTEM,flownonSTEM],dtype=np.float64)
    LaborGradeInt=np.array([int(x) for x in LaborGradeRange*100])
    LaborFinal=np.linspace(200,400,21,dtype=np.int64)

    wage_coeffs=wage_coeffs_full[:-1]
    year_four_wage_coeffs = wage_coeffs_full[-1].copy()
    year_four_wage_coeffs[0]=year_four_intercept


    year_four_wage_coeffs[4]=year_four_exp
    year_four_wage_coeffs[7]=year_four_quadratic
    year_four_wage_coeffs[8]=year_four_year_1

    # Generate Emax for entire population
    # This is just EmaxLink.py
    # Generate shocks to be used in approximating normal integrals

    zscores=scipy.stats.norm.ppf(np.array(range(1,normReps+1))/(normReps+1))
    num_quantiles=20
    norm_quantiles=scipy.stats.norm.ppf(
        np.array(range(1,num_quantiles))/num_quantiles)

    base_draws=np.matrix.transpose(np.matrix(list(
        itertools.product(zscores,repeat=(sectors)))))
    lmat=np.linalg.cholesky(skilled_wage_covar)
    wage_shocks=np.array(np.transpose(np.matmul(lmat,base_draws)))

    STEM_payouts_raw=np.zeros(11,dtype=np.float64)
    nonSTEM_payouts_raw=np.zeros(11,dtype=np.float64)
    skilled_Emax=np.zeros((2*11*2,grad_horizon,final_size))



    # Fully solve the unskilled labor market, over the 4 dropout times
    dropout_payouts=np.zeros((4,2),dtype=np.float64)
    unskilled_reps=20
    unskilled_wage_shocks=np.array(np.transpose(np.matrix(scipy.stats.norm.ppf(
        np.array(range(1,unskilled_reps+1))/
        (unskilled_reps+1)))))*unskilled_var[0][0]**0.5


    quality=0
    unskilled_Emax=[None]*4
    for drop_time in range(4):
        if drop_time==3:
            unskilled=EmaxLaborFunctionsJITUnskilled(grad_horizon+4-drop_time-1,
                gamma_p,beta,np.array([year_four_wage_coeffs]),0,0,
                np.array([year_four_flow - year_four_switching]),
                unskilled_wage_shocks,choose,year_four_switching,
                zero_exp_penalty,0)
        else:
            unskilled=EmaxLaborFunctionsJITUnskilled(grad_horizon+4-drop_time-1,
                gamma_p,beta,np.array([wage_coeffs_full[-1]]),0,0,
                np.array([flowUnskilled]),unskilled_wage_shocks,choose,
                unskilled_switch_cost,zero_exp_penalty,0)
        unskilled.solveLabor()
        dropout_payouts[drop_time,0]=unskilled.EmaxList[1,0,1]
        dropout_payouts[drop_time,1]=unskilled.EmaxList[1,1,0]
        unskilled_Emax[drop_time]=unskilled.EmaxList

    unskilled_Emax=tuple(unskilled_Emax)


    # Solve education by extracting each unique tuition and generating an
    # Emax function for that.

    flow_educ=np.array([flowSTEM,flownonSTEM,flowUnskilled],dtype=np.float64)

    unskilled_mean=wage_coeffs_full[-1][0]
    unskilled_meanvar=np.array((unskilled_mean,unskilled_var[0][0]),
        dtype=np.float64)

    skilled_experience=create_skilled_experience(np.array(DFData.skilled1),
        np.array(DFData.skilled2),np.array(DFData.skilled3),
        np.array(DFData.hp))

    wage_shock=calculate_wage_shock(np.array(DFData.outcome),
        np.array(DFData.col_type),np.array(DFData.numeric_choice),
        np.array(DFData.numeric_state),skilled_wage_coeffs,
        unskilled_wage_coeffs,skilled_experience,
        np.array(DFData.unskilled),np.array(DFData.dSTEM),np.array(DFData.tGPA),
        np.array(DFData.quality),np.array(DFData.tdropout),year_four_intercept,
        year_four_exp, year_four_quadratic,year_four_year_1)

    # Calculates likelihood of labor market decisions
    # This is the product of the likelihood of each wage offer
    # and the likelihood of making that choice, integrated over a conditional
    # wage offer of everything else
    # Integrating over a multivariate normal is done via simulation at
    # a number of points, given by cWagePoints

    # returns wage log-likelihood for all of an individual's observations
    # ignores normalizations like 1/sqrt(2pi)

    # calculates 2 likelihoods and returns a 2xN numpy array
    


    # run list of posteriors to integrate wages over
    (meanterm,covar,skilled_shocks,hp_wage_shocks)=(
        MVNposterior(skilled_wage_covar,4))
    skilled_shocks_list=[x for x in skilled_shocks]
    skilled_wage_shocks=tuple(skilled_shocks_list)



    # Skilled Choice likelihood
    # For each wage observation, the choice likelihood is a logit
    # conditional on the distribution of every other wage observation
    # cases differ for home production (no posterior)
    # vs non-posterior


    unskilledWageShocks=(np.transpose(scipy.stats.norm.ppf(
        (np.array(range(simReps))+1)/(simReps+1))*(unskilled_var[0][0])**0.5))
    firstUnskilledDraws=np.exp(unskilled_wage_coeffs[0]+unskilledWageShocks)

    # Unskilled choice likelihood
    # HP requires integration over unobserved unskilled wage shocks
    # condition: state=='unskilled' and choice=='unskilled'
    # dropout_time = 


    # Need to round this to coarseness of the rounding dataset
    # returns an array of the future grades the individual

    num_grades=20
    grade_quantiles=scipy.stats.norm.ppf(
        np.array(range(1,num_grades))/num_grades)



    out=calculate_likelihood_unskilled(grad_horizon,sectors,
        np.array(DFData.time),np.array(DFData.numeric_choice),
        np.array(DFData.numeric_state),np.array(DFData.cumulativeGPA),
        np.array(DFData.tdropout),np.array(DFData.A_N),np.array(DFData.A_S),
        np.array(DFData.SAT_M),np.array(DFData.SAT_V),np.array(DFData.hs_GPA),
        np.array(DFData.tuition),
        None, None, grade_quantiles,
        dropout_payouts, wage_shock,
        skilled_wage_shocks,hp_wage_shocks,skilled_experience,
        np.array(DFData.unskilled),np.array(DFData.outcome),
        np.array(DFData.dSTEM),np.array(DFData.tGPA),meanterm,
        skilled_Emax,unskilled_Emax,None,
        np.array(DFData.ed_emax_mapping),
        np.array(DFData.unskilled_emax_mapping),
        np.array(DFData.skilled_emax_mapping),
        flowUnskilled,flowSchool,
        flows_penalized,skilled_wage_covar,
        gamma_p,beta,skilled_wage_coeffs,unskilled_wage_coeffs,
        unskilled_var,choose,unskilledWageShocks,grade_params,None,None,
        firstUnskilledDraws,LaborGradeInt,np.array(DFData.prior_skilled),
        unskilled_switch_cost,year_four_intercept,year_four_flow,zero_exp_penalty,
        year_four_exp,year_four_quadratic,year_four_year_1,year_four_switching)
    if return_array:
        return out
    return np.sum(out)



@jit(nopython=True)
def calculate_likelihood_unskilled(grad_horizon,sectors,time,choice,state,
    cum_GPA,tdropout,A_N,A_S,SAT_M,SAT_V,hs_GPA,tuition,
    STEM_payouts, nonSTEM_payouts, grade_quantiles,
    dropout_payouts, wage_shock,skilled_wage_shocks,hp_wage_shocks,
    experience,unskilled_exp,outcome,
    dSTEM,tGPA,meanterm,skilled_Emax,unskilled_Emax,ed_Emax,
    ed_emax_mapping, unskilled_emax_mapping, skilled_emax_mapping,
    flowUnskilled,flowSchool,
    flows_penalized,skilled_wage_covar,
    gamma_p,beta,skilled_wage_coeffs,unskilled_wage_coeffs,
    unskilled_var,choose,unskilledWageShocks,grade_params,STEM1,nonSTEM1,
    firstUnskilledDraws,LaborGradeInt,prior_skilled,unskilled_switch_cost,
    year_four_intercept,year_four_flow,zero_exp_penalty,
    year_four_exp,year_four_quadratic,year_four_year_1,year_four_switching):

    length=time.shape[0]
    # first column is choice likelihood, second is obs likelihood
    out=np.zeros((length,2),dtype=np.float64)


    for x in range(length):

        # unskilled work
        if choice[x]==3:
            # missing wage
            if np.isnan(wage_shock[x]):
                out[x,0]=ChoiceLikeUnskilledNoWage(grad_horizon,sectors,
                    time[x],tdropout[x],1,unskilled_exp[x],experience[x][3],
                    unskilled_wage_coeffs,outcome[x],
                    unskilled_Emax[unskilled_emax_mapping[x]],gamma_p,beta,
                    flowUnskilled,unskilledWageShocks,choose,
                    unskilled_switch_cost,prior_skilled[x],year_four_intercept,
                    year_four_flow,zero_exp_penalty,
                    year_four_exp,year_four_quadratic,year_four_year_1,year_four_switching)
            else:
                out[x,1]=(-0.5*np.log(unskilled_var[0][0])-
                    (wage_shock[x])**2/(2*unskilled_var[0][0]))
                out[x,0]=ChoiceLikeUnskilled(grad_horizon,sectors,time[x],
                    tdropout[x],1,unskilled_exp[x],experience[x][3],
                    unskilled_wage_coeffs,outcome[x],
                    unskilled_Emax[unskilled_emax_mapping[x]],gamma_p,beta,
                    flowUnskilled,unskilledWageShocks,choose,
                    unskilled_switch_cost,prior_skilled[x],year_four_intercept,
                    year_four_flow,zero_exp_penalty,
                    year_four_exp,year_four_quadratic,year_four_year_1,year_four_switching)

        else:
            out[x,0]=ChoiceLikeUnskilled(grad_horizon,sectors,time[x],
                tdropout[x],0,unskilled_exp[x],experience[x][3],
                unskilled_wage_coeffs,outcome[x],
                unskilled_Emax[unskilled_emax_mapping[x]],gamma_p,beta,
                flowUnskilled,unskilledWageShocks,choose,unskilled_switch_cost,
                prior_skilled[x],year_four_intercept,year_four_flow,
                zero_exp_penalty,
                year_four_exp,year_four_quadratic,year_four_year_1,year_four_switching)


    return out


# conditional on types. Wrapper logic occurs outside, i.e.
# STEM_payouts and nonSTEM_payouts vary by type
# MAKE SURE IT DOESN'T CONTAIN ANY NON-EDUCATION DATA (i.e. labor)
def LogLikeEducation(DFData,SATTuition,grad_horizon,sectors,beta,
    ability,flows_penalized,unskilled_var,grade_params_by_quality,normReps,simReps,
    LaborGradeRange,final_size,dropout_payouts, STEM_payouts_by_quality,
    nonSTEM_payouts_by_quality,gamma_p, unskilled_meanvar, norm_quantiles,
    skilled_wage_coeffs,unskilled_wage_coeffs, skilled_wage_covar,LaborGradeInt,
    choose,year_four_intercept,year_four_flow_penalized, ed_switching_costs,
    univ_type_shifters,grad_payoff,return_array=False):
    """
        Generates log-likelihood for education decisions, avoiding
        having to solve the labor market again and again
        Log-likelihood is approximated for the various integrals 
        Args:
            DFData (DataFrame): data set. Needs to be preprocessed with
                PreProcess(DF), and set_index('id')
            SATTuition (list): list of tuples of exogenous endowments and
                tuitions (set to zero) that affect educational payoff
            grad_horizon (int): Years of work post-college graduation
            sectors (int): number of skilled sectors
            beta (float): discount factor
            ability ([float]): array of ability
            gamma_p (float): pref. for money
            beta (float): discount factor
            ability ([float]): list of unobserved ability [A_S,A_N]
            flows_penalized ([float]): list of flow utilities penalized:
                flows_penalized+[flow_unskilled_penalized,flowSTEM,flownonSTEM]
                ED IS PENALIZED
            wage_coeffs_full ([[float]]): nested list of log-wage coefficients
                skilled, then unskilled last. wage coeffs are of form
                [intercept,M,G,MxG,x,x^2]
            skilled_wage_covar ([[float]]): nested list of covariance of
                log-wage shocks in the skiled sector
            unskilled_var ([[float]]): variance of unskilled wage shocks,
                nested inside double list
            grade_params ([[float]]): [SAT_M, SAT_V, hs_GPA,lambda_1,lambda_2,
                lambda_3,lambda_4,sigma^2_m] for STEM, non-STEM
            normReps (int): sectors^normReps simulations of log-wage shocks
            simReps (int): number of simulations of unskilled wage shocks
            LaborGradeRange (np.array): grades to use to discretize graduation
                outcomes. Segmenting of GPA from 2.0 to 4.0. Education is always
                assuming 0.1, but I will interpolate payouts over labor mkt
            ed_switching_costs: switching costs for STEM and non-STEM
        Returns:
            float: log-likelihood

    """

    year_four_exp = 0
    year_four_quadratic = 0
    year_four_year_1 = 0


    flowUnskilled=flows_penalized[sectors]
    flowSTEM=flows_penalized[sectors+1]
    flownonSTEM=flows_penalized[sectors+2]
    flow_educ=np.array([flowSTEM,flownonSTEM,flowUnskilled],dtype=np.float64)
    ed_Emax=np.zeros((len(SATTuition),6,81,2),dtype=np.float64)
    STEM1=np.zeros(len(SATTuition),dtype=np.float64)
    nonSTEM1=np.zeros(len(SATTuition),dtype=np.float64)

    flows_by_univ_type = [np.zeros(3,dtype=np.float64) for x in range(4)]
    flows_by_univ_type[0][0]=flow_educ[0]
    flows_by_univ_type[0][1]=flow_educ[1]
    flows_by_univ_type[0][2]=flow_educ[2]
    for x in range(1,4):
        flows_by_univ_type[x][0]=flow_educ[0]+univ_type_shifters[2*x-2]
        flows_by_univ_type[x][1]=flow_educ[1]+univ_type_shifters[2*x-1]
        flows_by_univ_type[x][2]=flows_penalized[sectors]

    for idx,x in enumerate(SATTuition):
        # differentiate by quality
        if x[5] == 1:
            STEM_payouts = STEM_payouts_by_quality[1]
            nonSTEM_payouts = nonSTEM_payouts_by_quality[1]
            grade_params = grade_params_by_quality[1]
        else:
            STEM_payouts = STEM_payouts_by_quality[0]
            nonSTEM_payouts = nonSTEM_payouts_by_quality[0]
            grade_params = grade_params_by_quality[0]

        tuition = x[6]
        flow_educ_univ_type=flows_by_univ_type[x[7]]

        Ed=EmaxEducationJIT(dropout_payouts,STEM_payouts,nonSTEM_payouts,
            grade_params,gamma_p,beta,flow_educ_univ_type,
            np.array(([tuition,tuition,tuition,tuition]),dtype=np.float64),
            np.array((x[2],x[3],x[4]),dtype=np.float64),
            np.array((ability[0],ability[1]),dtype=np.float64),
            unskilled_meanvar, norm_quantiles, year_four_intercept,
            year_four_flow_penalized, ed_switching_costs, grad_payoff)
        Ed.solve()
        ed_Emax[idx]=Ed.EmaxEducationValues
        STEM1[idx]=Ed.STEM_cond_val_first
        nonSTEM1[idx]=Ed.nonSTEM_cond_val_first
        del Ed



    # this part is hard coded for the number of sectors
    skilled_experience=create_skilled_experience(np.array(DFData.skilled1),
        np.array(DFData.skilled2),np.array(DFData.skilled3),
        np.array(DFData.hp))



    wage_shock=calculate_wage_shock(np.array(DFData.outcome),
        np.array(DFData.col_type),np.array(DFData.numeric_choice),
        np.array(DFData.numeric_state),skilled_wage_coeffs,
        unskilled_wage_coeffs,skilled_experience,
        np.array(DFData.unskilled),np.array(DFData.dSTEM),np.array(DFData.tGPA),
        np.array(DFData.quality),np.array(DFData.tdropout),year_four_intercept,
        year_four_exp,year_four_quadratic,year_four_year_1)

    # Calculates likelihood of labor market decisions
    # This is the product of the likelihood of each wage offer
    # and the likelihood of making that choice, integrated over a conditional
    # wage offer of everything else
    # Integrating over a multivariate normal is done via simulation at
    # a number of points, given by cWagePoints

    # returns wage log-likelihood for all of an individual's observations
    # ignores normalizations like 1/sqrt(2pi)

    # calculates 2 likelihoods and returns a 2xN numpy array
    


    # run list of posteriors to integrate wages over
    (meanterm,covar,skilled_shocks,hp_wage_shocks)=(
        MVNposterior(skilled_wage_covar,4))
    skilled_shocks_list=[x for x in skilled_shocks]
    skilled_wage_shocks=tuple(skilled_shocks_list)




    # Skilled Choice likelihood
    # For each wage observation, the choice likelihood is a logit
    # conditional on the distribution of every other wage observation
    # cases differ for home production (no posterior)
    # vs non-posterior


    unskilledWageShocks=(np.transpose(scipy.stats.norm.ppf(
        (np.array(range(simReps))+1)/(simReps+1))*(unskilled_var[0][0])**0.5))
    firstUnskilledDraws=np.exp(unskilled_wage_coeffs[0]+unskilledWageShocks)

    year_four_first_draws=np.exp(year_four_intercept+unskilledWageShocks)
    # Unskilled choice likelihood
    # HP requires integration over unobserved unskilled wage shocks
    # condition: state=='unskilled' and choice=='unskilled'
    # dropout_time = 



    # Need to round this to coarseness of the rounding dataset
    # returns an array of the future grades the individual

    num_grades=20
    grade_quantiles=scipy.stats.norm.ppf(
        np.array(range(1,num_grades))/num_grades)

    out=calculate_likelihood_education(grad_horizon,sectors,
        np.array(DFData.time),
        np.array(DFData.numeric_choice),np.array(DFData.numeric_state),
        np.array(DFData.cumulativeGPA),np.array(DFData.tdropout),ability,
        np.array(DFData.SAT_M),
        np.array(DFData.SAT_V),np.array(DFData.hs_GPA),np.array(DFData.tuition),
        STEM_payouts_by_quality, nonSTEM_payouts_by_quality, grade_quantiles,
        dropout_payouts, wage_shock,
        skilled_wage_shocks,hp_wage_shocks,skilled_experience,
        np.array(DFData.unskilled),np.array(DFData.outcome),
        np.array(DFData.dSTEM),np.array(DFData.tGPA),meanterm,
        ed_Emax,np.array(DFData.ed_emax_mapping),
        flowUnskilled,flow_educ,skilled_wage_covar,
        gamma_p,beta,skilled_wage_coeffs,unskilled_wage_coeffs,
        unskilled_var,choose,unskilledWageShocks,grade_params_by_quality,STEM1,
        nonSTEM1,firstUnskilledDraws,LaborGradeInt,np.array(DFData.quality),
        year_four_first_draws,year_four_flow_penalized,ed_switching_costs,
        np.array(DFData.lastchoice),univ_type_shifters,
        np.array(DFData.univ_type_num),grad_payoff)

    del ed_Emax
    if return_array:
        return out
    return np.sum(out)

@jit(nopython=True)
def calculate_likelihood_education(grad_horizon,sectors,time,choice,
    state,cum_GPA,tdropout,ability,SAT_M,SAT_V,hs_GPA,tuition,
    STEM_payouts_by_quality, nonSTEM_payouts_by_quality, grade_quantiles,
    dropout_payouts, wage_shock,skilled_wage_shocks,hp_wage_shocks,
    experience,unskilled_exp,outcome,dSTEM,tGPA,meanterm,ed_Emax,
    ed_emax_mapping,flowUnskilled,flow_educ,
    skilled_wage_covar,gamma_p,beta,skilled_wage_coeffs,unskilled_wage_coeffs,
    unskilled_var,choose,unskilledWageShocks,grade_params_by_quality,STEM1,
    nonSTEM1,firstUnskilledDraws,LaborGradeInt,quality,year_four_first_draws,
    year_four_flow_penalized,ed_switching_costs,last_choice,univ_type_shifters,
    univ_type_num,grad_payoff):

 
    length=time.shape[0]
    # first column is choice likelihood, second is obs likelihood
    out=np.zeros((length,2),dtype=np.float64)
    for x in range(length):

        if quality[x] == 1:
            STEM_payouts = STEM_payouts_by_quality[1]
            nonSTEM_payouts = nonSTEM_payouts_by_quality[1]
            grade_params = grade_params_by_quality[1]
        else:
            STEM_payouts = STEM_payouts_by_quality[0]
            nonSTEM_payouts = nonSTEM_payouts_by_quality[0]
            grade_params = grade_params_by_quality[0]

        # Education sector

        exogChars=np.array([SAT_M[x],SAT_V[x],hs_GPA[x]],dtype=np.float64)
        # go to college (1-choice because 0 = )
        if choice[x]==0 or choice[x]==1:
            out[x,1]=GradeLike(exogChars,time[x],
                (1-choice[x]),outcome[x],grade_params,ability)
        # drop out
        elif choice[x]==3 and not np.isnan(outcome[x]):
            out[x,1]=(-0.5*np.log(unskilled_var[0][0])-
                    (wage_shock[x])**2/(2*unskilled_var[0][0]))

        out[x,0]=ChoiceLikeSchool(grad_horizon,choice[x],
            state[x],cum_GPA[x-1],exogChars,STEM_payouts,
            nonSTEM_payouts,grade_quantiles,dropout_payouts,
            ed_Emax[ed_emax_mapping[x]],STEM1[ed_emax_mapping[x]],
            nonSTEM1[ed_emax_mapping[x]],gamma_p,beta,flow_educ,
            tuition[x],ability,firstUnskilledDraws,LaborGradeInt,
            outcome[x],grade_params,year_four_first_draws,
            year_four_flow_penalized,last_choice[x],ed_switching_costs,
            univ_type_shifters,univ_type_num[x],grad_payoff)

    return out
if __name__ == '__main__':
    main()