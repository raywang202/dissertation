# Calculates objective function used in E-M algorithm (I wrap this in the KNITRO
# package during my actual optimization procedure)

import numpy as np
import pandas as pd
import os
import simplexmap
import scipy.interpolate
import scipy.stats
import itertools

from numba import jit

from EmaxLaborFunctionsJITUnskilled import EmaxLaborFunctionsJITUnskilled
from EmaxLaborFunctionsJIT import EmaxLaborFunctionsJIT
from EmaxEducationJIT import EmaxEducationJIT
import FSLikelihoodJITfinal as fs

#===============================================================================
# Calculates E-M algorithm objective function, namely posterior probabilities
# multiplied by the log-likelihood of the observations conditioned on the type
#===============================================================================

def LaborLikeEM(x,DFDataSkilled,SATTuition,college_values,grad_horizon,
    num_types,sectors,gamma_p,beta,ability,normReps,simReps,LaborGradeRange,
    final_size,horizon,posteriors_endowment):

    gamma_p = 0.045
    #===========================================================================
    # Parse all of the parameters
    #===========================================================================

    zero_exp_penalty = np.array(x[68:71],dtype=np.float64)
    switching_costs =  np.array(x[73:76],dtype=np.float64)

    # note that these are s1, s2, s3, unskilled, STEM, non-STEM, t0s1, 
    # t0s2, t0s3
    flows_skilled_by_type_penalized = [np.zeros(2*sectors+3,
        dtype=np.float64) for i in range(num_types)]
    for i in range(num_types):
        flows_skilled_by_type_penalized[i][0:3]=(x[(49+3*i):(52+3*i)]-
            switching_costs)
        flows_skilled_by_type_penalized[i][(sectors+3):(2*sectors+3)]=(
            [x[49+3*i]+x[65],x[50+3*i]+x[66],x[51+3*i]+x[67]]-switching_costs)
    # iterate over both endowment and preference types
    flowsUnskilled=np.array([0,0,0,x[48],0,0],dtype=np.float64) 

    wage_coeffs_full_by_type=[np.array([[0]+x[0:9]+[x[61]],
    [0]+x[9:18]+[x[62]],[0]+x[18:27]+[x[63]],
    [x[31],0,0,0,x[32],0,0,x[33],x[34],0,x[64]]],dtype=np.float64) 
    for i in range(num_types)]
    for i in range(num_types):
        for j in range(3):
            wage_coeffs_full_by_type[i][j][0]=x[36+3*i+j]
    skilled_wage_covar=np.array([[x[27],0,0],[0,x[28],0],[0,0,x[29]]])

    # dummy variables
    grade_params=np.array([[0]*7,[0]*7],dtype=np.float64)

    unskilled_var=np.array([[x[35]]],dtype=np.float64)
    ability=np.zeros(2)


    skilled_cont=[np.sum(fs.LogLikeSkilled(DFDataSkilled,SATTuition,
        college_values,grad_horizon,sectors,gamma_p,beta,ability,
        flows_skilled_by_type_penalized[i],
        wage_coeffs_full_by_type[i],skilled_wage_covar,unskilled_var,
        grade_params,normReps,simReps,LaborGradeRange,final_size,
        switching_costs,zero_exp_penalty,2,horizon,True),axis=1)*
    posteriors_endowment[:,i] for i in range(num_types)]

    return np.sum(sum(skilled_cont))


def education_emax_mapping(A_N,A_S,SAT_M,SAT_V,hs_GPA,quality,tuition,univ_type,
    SATTuition):
    out=np.zeros(len(A_N),dtype=np.int64)
    for x in range(len(A_N)):
        key=tuple([A_N[x],A_S[x],SAT_M[x],SAT_V[x],hs_GPA[x],quality[x],
            tuition[x],univ_type[x]])
        out[x]=SATTuition.index(key)
    return out

def skilled_emax_mapping(dSTEM,quality,state,tGPA,college_values):
    out=np.zeros(len(dSTEM),dtype=np.int64)

    for x in range(len(dSTEM)):
        if state[x]==7:
            key=tuple([quality[x],dSTEM[x],tGPA[x]])
            out[x]=college_values.index(key)   
        else:
            out[x]=-1
    return out

def normalize_posterior(arr,prior):
    size=arr.shape[1]
    raw_out=np.zeros((len(prior),size),dtype=np.float64)
    out=np.zeros((len(prior),size),dtype=np.float64)
    for i in range(size):
        row_max=np.max(arr[:,i])
        for j in range(len(prior)):
            raw_out[j,i]=np.exp(arr[j,i]-row_max)*prior[j]
    out=raw_out/np.sum(raw_out,axis=0)
    return out
        
#===============================================================================
# scales utilities of each alternative
#===============================================================================
def normalize_raw_like(arr):
    size=arr.shape[1]
    types=arr.shape[0]
    raw_out=np.zeros((types,size),dtype=np.float64)
    out=np.zeros((types,size),dtype=np.float64)
    for i in range(size):
        row_max=np.max(arr[:,i])
        for j in range(types):
            raw_out[j,i]=arr[j,i]-row_max
    return raw_out


def GenerateEmaxLaborFunctions(sectors,horizon,horizon_later,skilled_wage_covar,
    gamma_p,beta,skilled_wage_coeffs,flows,wageshocks,major,
    grade,choose,quality,switching_costs_skilled, wageshocks_later,
    zero_exp_penalty):
    skilled_flows_penalized = flows[0:sectors]
    time_zero_flows_penalized = (flows[(sectors+3):(2*sectors+3)])


    output=EmaxLaborFunctionsJIT(horizon,horizon_later-1,gamma_p,beta,
        np.array(skilled_wage_coeffs,dtype=np.float64),major,grade,
        skilled_flows_penalized,wageshocks,choose,quality,
        time_zero_flows_penalized,switching_costs_skilled,zero_exp_penalty,
        wageshocks_later)
    output.solveLabor()
    return output.EmaxList


def main():

    sectors=3 # occupational sectors
    LaborGradeRange=np.linspace(2,4,11) # discretization of GPA

    DFDataFull=pd.read_csv(os.path.abspath(os.curdir)+'/sim_data.csv')
    DFDataFinal=DFDataFull.sort_values(by=['id','time'])

    DFDataFinal['tuition']=round(DFDataFinal['tuition'],3)
    SATTuition=list(set(list(DFDataFinal[
        ['A_N','A_S','SAT_M','SAT_V','hs_GPA','quality','tuition',
        'univ_type_num']].itertuples(index=False,name=None))))

    DFDataFinal['ed_emax_mapping']=education_emax_mapping(
        np.array(DFDataFinal.A_N),np.array(DFDataFinal.A_S),
        np.array(DFDataFinal.SAT_M),np.array(DFDataFinal.SAT_V),
        np.array(DFDataFinal.hs_GPA),np.array(DFDataFinal.quality),
        np.array(DFDataFinal.tuition),np.array(DFDataFinal.univ_type_num),
        SATTuition)


    DFData=DFDataFinal.set_index('id')

    # Fixed Parameters
    grad_horizon=20 # number of years after grad over which individual can work
    final_size=int((grad_horizon+2)*(grad_horizon+1)*grad_horizon/6) 
    normReps=3 # number of times to approximate normal integrals
    simReps=9

    x=[None]*77
    x[   0] =  -3.57463145387e-01
    x[   1] =   2.91977296659e-03
    x[   2] =   1.95316216605e-01
    x[   3] =   9.25860946596e-02
    x[   4] =   8.62884440731e-02
    x[   5] =   2.92030386564e-02
    x[   6] =  -2.91496807177e-03
    x[   7] =   1.42318700232e-01
    x[   8] =   1.71465816752e-01
    x[   9] =  -7.18086990825e-02
    x[  10] =   7.60432176422e-02
    x[  11] =   2.11765561474e-02
    x[  12] =   4.38204792908e-02
    x[  13] =   1.20394116017e-01
    x[  14] =   3.51686757779e-02
    x[  15] =  -3.36630490648e-03
    x[  16] =   1.65158701538e-01
    x[  17] =   1.44959412552e-01
    x[  18] =  -2.08515787160e-01
    x[  19] =   3.12896606294e-02
    x[  20] =   3.60700910377e-02
    x[  21] =   9.61613435526e-02
    x[  22] =   1.00016800946e-01
    x[  23] =   1.07201214624e-01
    x[  24] =  -6.66985383651e-04
    x[  25] =   8.08772463941e-02
    x[  26] =   7.83390848292e-02
    # variance
    x[   27] =   1.20298718487e-01
    x[   28] =   1.84871679894e-01
    x[   29] =   2.04041389338e-01

    x[  30] =   0.045
    x[      31] =   2.23952504637e+00
    x[      32] =   1.89728166714e-01
    x[      33] =  -1.25794969577e-02
    x[      34] =   7.75091193775e-02
    x[      35] =   2.26338674507e-01


    # endowments
    x[  36] =   3.05293283161e+00
    x[  37] =   2.62441196225e+00
    x[  38] =   2.46033967260e+00
    x[  39] =   2.56368098198e+00
    x[  40] =   2.72121594147e+00
    x[  41] =   2.40046279662e+00
    x[  42] =   2.61531385785e+00
    x[  43] =   2.35571558220e+00
    x[  44] =   2.84266634225e+00

    x[  45] =   None
    x[  46] =   None
    x[  47] =   None

    # unskilled penalized
    x[  48] =   2.26338674507e-01-1.74504648900e+00-1.06002212772e-01

    # preferences
    x[  49] =  -9.59383751286e-01
    x[  50] =  -1.18126176863e-00
    x[  51] =  -5.00411469168e-01
    x[  52] =  -7.73609805648e-01
    x[  53] =  -1.09947141311e-00
    x[  54] =  -5.33518970995e-01
    x[  55] =  -1.59673817156e-00
    x[  56] =  -8.01187982756e-01
    x[  57] =  -8.05363312411e-01

    x[  58] =  None
    x[  59] =  None
    x[  60] =  None

    # age return
    x[  61] =  0
    x[  62] =  0
    x[  63] =  0

    x[64] = 0 # unskilled age return 

    # time zero penalty
    x[  65] =  -1.00163200467e-00
    x[  66] =  -9.65601821815e-01
    x[  67] =  -3.17868117763e-01

    # zero experience additional nonpecuniary cost
    x[  68] =  -8.48824381131e-01
    x[  69] =  -8.15931334463e-01
    x[  70] =   0.00000000000e+00
    # unskilled switching cost (not used atm)
    x[71] = -1

    # unskilled year four intercept
    x[  72] =   2.32692315928e+00

    # switching costs
    x[  73] =   2.67037140390e-00
    x[  74] =   1.72109708887e-00
    x[  75] =   1.96173635149e-00

    # year four unskilled flow utility (penalized)
    x[76] = -2.78146963456e-02-2.70327516104-1.06002212772e-01

    xLabor = x.copy()


    # ability is normalized to 0 for type 1

    # xEduc[0:7] STEM params: SAT_M, SAT_V, HS_GPA, year 1-4
    # xEduc[7:14] nonSTEM params
    # xEduc[14:16] STEM vars quality 0
    # xEduc[16:18] nonSTEM vars quality 0
    # xEduc[18:20] STEM vars quality 1
    # xEduc[20:22] nonSTEM vars quality 1
    # xEduc[22:24] prefs type 1
    # xEduc[24:26] prefs type 2
    # xEduc[26:28] prefs type 3
    # xEduc[28:30] type 2 ability intercept 
    # xEduc[30:32] type 3 ability intercept
    # xEduc[32:36] high quality STEM intercepts
    # xEduc[36:40] high quality nonSTEM intercepts
    # xEduc[40:42] switching cost to STEM, nonSTEM
    # xEduc[42:44] private non-rel STEM/non-STEM shifter
    # xEduc[44:46] private rel STEM/non-STEM shifter
    # xEduc[46:48] for-profit STEM/non-STEM shifter
    # xEduc[48] is the payoff to graduating
    X=[None]*49
    X[0]=  1.000e-01
    X[1]=  1.832e-01
    X[2]=  3.720e-01
    X[3]=  9.744e-02
    X[4]=  2.766e-01
    X[5]=  3.681e-01
    X[6]=  5.680e-01
    X[7]=  5.205e-02
    X[8]=  1.074e-01
    X[9]=  4.178e-01
    X[10]=  6.6813e-01
    X[11]=  6.3102e-01
    X[12]=  7.0338e-01
    X[13]=  8.8836e-01
    X[14]=  5.26273933209e-01
    X[15]=  3.23304097751e-01
    X[16]=  4.67322142646e-01
    X[17]=  3.09228549750e-01
    X[18]=  4.21979725688e-01
    X[19]=  1.66911362212e-01
    X[20]=  1.18688629073e-01
    X[21]=  5.23589564731e-01
    X[22]=  1.34233321411e+00
    X[23]=  1.29899913004e+00
    X[24]=  1.19357198084e+00
    X[25]=  1.64026427950e+00
    X[26]=  8.33983203649e-01
    X[27]=  8.34875776639e-01
    X[28]= -3.27021154148e-01
    X[29]=  1.31210372774e-01
    X[30]= -9.23269494714e-02
    X[31]= -1.89372159214e-01
    X[32]=  1.55289127232e-01
    X[33]=  1.48446665696e-01
    X[34]=  3.91901877961e-01
    X[35]=  3.64680503363e-01
    X[36]=  7.83851719939e-01
    X[37]=  8.67293248077e-01
    X[38]=  6.37921879489e-01
    X[39]=  8.07808386509e-01
    X[40]=  6.72267877497e-01
    X[41]=  1.90896066934e+00
    X[42]=  2.85064990522e-01
    X[43]=  5.01107287546e-01
    X[44]=  2.81513397647e-02
    X[45]=  1.08530022547e+00
    X[46]= -5.52138028336e-01
    X[47]=  9.64522578512e-02
    X[48]=  2.51913840318e+00

    xEduc=X.copy()

    # discount factor
    beta=0.9

    # split data into skilled and unskilled
    DFDataSkilled=DFData[DFData.numeric_state==7]
    DFDataUnskilled=DFData[DFData.numeric_state==8]
    DFDataEducation=DFData[(DFData.numeric_state<=6)]


    college_values = list(set(list(DFDataSkilled[
        ['quality','dSTEM','tGPA']].itertuples(index=False,name=None))))


    DFDataSkilled['skilled_emax_mapping_abridged']=skilled_emax_mapping(
        np.array(DFDataSkilled.dSTEM), np.array(DFDataSkilled.quality),
        np.array(DFDataSkilled.numeric_state),
        np.array(DFDataSkilled.tGPA),college_values)



    num_endowment_types = 3

    # Solve the labor market model to get the payoff values for college/dropout

    # Parse labor parameters
    gamma_p=xLabor[30]
    skilled_wage_covar=np.array([[xLabor[27],0,0],[0,xLabor[28],0],
        [0,0,xLabor[29]]])

    zero_exp_penalty = np.array(xLabor[68:71],dtype=np.float64)

    # remember that flows_skilled_by_type includes the unskilled and then 
    # time zero flows


    switching_costs_skilled = np.array(xLabor[73:76],dtype=np.float64)

    flows_skilled_by_type_penalized = [np.zeros(2*sectors+3,
        dtype=np.float64) for i in range(num_endowment_types)]
    for i in range(num_endowment_types):
        flows_skilled_by_type_penalized[i][0:3]=(xLabor[(49+3*i):(52+3*i)]-
            switching_costs_skilled)
        flows_skilled_by_type_penalized[i][(sectors+3):(2*sectors+3)]=(
            [xLabor[49+3*i]+xLabor[65],xLabor[50+3*i]+xLabor[66],
            xLabor[51+3*i]+xLabor[67]]-switching_costs_skilled)



    # iterate over both endowment and preference types
    flowsUnskilled=np.array([0,0,0,xLabor[48],0,0],dtype=np.float64)
    wage_coeffs_full_by_type=[np.array([[0]+xLabor[0:9]+[xLabor[61]],
    [0]+xLabor[9:18]+[xLabor[62]],[0]+xLabor[18:27]+[xLabor[63]],
    [xLabor[31],0,0,0,xLabor[32],0,0,xLabor[33],xLabor[34],0,0]],dtype=np.float64) 
    for i in range(3)]
    for i in range(3):
        for j in range(3):
            wage_coeffs_full_by_type[i][j][0]=xLabor[36+3*i+j]


    # dummy variables
    grade_params=np.array([[0]*9,[0]*9],dtype=np.float64)
    unskilled_var=np.array([[xLabor[35]]],dtype=np.float64)
    ability=np.zeros(2)
    LaborEmax={}
    LaborGradeRange=np.linspace(2,4,11)
    normReps=3
    choose = simplexmap.pascal(grad_horizon+4,sectors)

    zscores=scipy.stats.norm.ppf(np.array(range(1,normReps+1))/(normReps+1))
    base_draws=np.matrix.transpose(np.matrix(list(
        itertools.product(zscores,repeat=(sectors)))))
    lmat=np.linalg.cholesky(skilled_wage_covar)
    wageshocks=np.array(np.transpose(np.matmul(lmat,base_draws)))

    normReps_later = 3
    zscores_later=scipy.stats.norm.ppf(np.array(range(1,normReps_later+1))/
        (normReps_later+1))
    base_draws_later=np.matrix.transpose(np.matrix(list(
        itertools.product(zscores_later,repeat=(sectors)))))
    lmat=np.linalg.cholesky(skilled_wage_covar)
    wageshocks_later=np.array(np.transpose(np.matmul(lmat,base_draws_later)))

    # heterogeneity in STEM and non-STEM payouts by type, but no type het in
    # unskilled payouts
    LaborFinal=np.linspace(200,400,21,dtype=np.int64)

    STEM_payouts=np.zeros([num_endowment_types,2,len(LaborFinal)],
        dtype=np.float64)
    nonSTEM_payouts=np.zeros([num_endowment_types,2,len(LaborFinal)],
        dtype=np.float64)
    dropout_payouts=np.zeros([4,2],dtype=np.float64)

    horizon=30
    horizon_later=grad_horizon

    # # Labor Emax is keyed by endowment, quality, dSTEM, grade
    for endowment_type in range(num_endowment_types):
        for quality in range(2):
            for dSTEM in range(2):
                for grade in LaborGradeRange:
                    LaborEmax[(endowment_type,quality,dSTEM,int(100*grade))]=\
                    GenerateEmaxLaborFunctions(sectors,horizon,horizon_later,
                        skilled_wage_covar,gamma_p,beta,
                        wage_coeffs_full_by_type[endowment_type][:-1],
                        flows_skilled_by_type_penalized[endowment_type],
                        wageshocks,dSTEM,grade,choose,quality,
                        np.array(switching_costs_skilled,dtype=np.float64),
                        wageshocks_later,zero_exp_penalty)


    # Fully solve the unskilled labor market, over the 4 dropout times
    flowUnskilled = xLabor[48]
    unskilled_reps=20
    unskilled_wage_shocks=np.array(np.transpose(np.matrix(scipy.stats.norm.ppf(
        np.array(range(1,unskilled_reps+1))/
        (unskilled_reps+1)))))*unskilled_var[0][0]**0.5

    unskilled_coeffs = wage_coeffs_full_by_type[0][-1]
    unskilled_coeffs[-1]=0 # age is set to zero (best fit)
    unskilled_Emax=[None]*4


    # take dropout payouts as given

    dropout_payouts[0,0]=17.95723211
    dropout_payouts[0,1]=19.99124993
    dropout_payouts[1,0]=17.88247618
    dropout_payouts[1,1]=19.91649401
    dropout_payouts[2,0]=17.79941404
    dropout_payouts[2,1]=19.83343187
    dropout_payouts[3,0]=15.73479004
    dropout_payouts[3,1]=18.99858802

    unskilled_Emax=tuple(unskilled_Emax)

    # Parse the payouts from the Emax scripts
    STEM_payouts_raw=np.zeros([num_endowment_types,2,11],dtype=np.float64)
    nonSTEM_payouts_raw=np.zeros([num_endowment_types,2,11],dtype=np.float64)

    for endowment_type in range(num_endowment_types):
        for quality in range(2):
            for idx,grade in enumerate(LaborGradeRange):
                STEM_payouts_raw[endowment_type,quality,idx]=(
                    LaborEmax[(endowment_type,quality,1,int(100*grade))][0,0,0])
                nonSTEM_payouts_raw[endowment_type,quality,idx]=(
                    LaborEmax[(endowment_type,quality,0,int(100*grade))][0,0,0])            

    LaborGradeInt=np.array([int(x) for x in LaborGradeRange*100])

    # Interpolate values to 0.1 GPA points for Education Emax functions

    for endowment_type in range(num_endowment_types):
        for quality in range(2):
            STEM_payouts[endowment_type,quality,:]=scipy.interpolate.interp1d(
                LaborGradeInt,STEM_payouts_raw[endowment_type,quality,:],
                kind='cubic')(LaborFinal)
            nonSTEM_payouts[endowment_type,quality,:]=(
                scipy.interpolate.interp1d(LaborGradeInt,
                    nonSTEM_payouts_raw[endowment_type,quality,:],kind='cubic')(
                    LaborFinal))



    # pick the 0th type, unskilled wage coeffs are repeated anyway
    # type 0, last value, first entry in that array
    unskilled_mean=wage_coeffs_full_by_type[0][-1][0]
    unskilled_meanvar=np.array((unskilled_mean,unskilled_var[0][0]),
        dtype=np.float64)


    (meanterm,covar,skilled_shocks,hp_wage_shocks)=(
        fs.MVNposterior(skilled_wage_covar,4))
    skilled_shocks_list=[x for x in skilled_shocks]
    skilled_wage_shocks=tuple(skilled_shocks_list)


    unskilledWageShocks=(np.transpose(scipy.stats.norm.ppf(
        (np.array(range(simReps))+1)/(simReps+1))*(unskilled_var[0][0])**0.5))
    firstUnskilledDraws=np.exp(unskilled_coeffs[0]+unskilledWageShocks)

    endowment_dist_old=np.array([0.275,0.464,0.261],dtype=np.float64)

    # Calculate posterior probabilities for labor:

    # x1=xLabor
    num_individuals=len(DFDataSkilled.groupby('id'))
    plike=np.zeros([num_endowment_types,len(DFDataSkilled)])
    individual_like=np.zeros([num_endowment_types,num_individuals])
    posteriors_endowment=np.zeros([len(DFDataSkilled),num_endowment_types])
    individual_id=np.unique(DFDataSkilled.index.values)


    # for each type, calculate loglikelihood

    for endowment_type in range(num_endowment_types):

        wage_coeffs_full=wage_coeffs_full_by_type[endowment_type]

        # individual conditional likelihood of each observation
        plike[endowment_type,:]=np.sum(fs.LogLikeSkilled(DFDataSkilled,
            SATTuition,college_values,grad_horizon,sectors,gamma_p,beta,ability,
            flows_skilled_by_type_penalized[endowment_type],
            wage_coeffs_full,skilled_wage_covar,
            unskilled_var,grade_params,normReps,
            simReps,LaborGradeRange,final_size,np.array(switching_costs_skilled,
                dtype=np.float64),zero_exp_penalty,3,horizon,True),axis=1)

        # calculate unconditional probability of individual i being of each type
        # i.e. Bayes' formula

        individual_like[endowment_type]=(DFDataSkilled.assign(
            loglike=plike[endowment_type]).groupby('id').\
        apply(lambda x: np.sum(x.loglike)))




    # # aggregate loglikelihood, using underflow prevention
    posteriors_by_id=normalize_posterior(individual_like,endowment_dist_old)
    # calculate posterior probability of being pref type i
    for i in range(num_endowment_types):
        posterior_frame=pd.DataFrame({'id':individual_id,
            'data':posteriors_by_id[i]})
        posterior_frame=posterior_frame.set_index('id')
        posteriors_endowment[:,i]=np.array(DFDataSkilled.merge(posterior_frame,
            left_index=True,right_index=True).data)


    skilled_like = normalize_raw_like(individual_like)
    # get likelihood contributions for skilled individuals
    skilled_like_by_id = pd.DataFrame({'id':individual_id,
            'type1':skilled_like[0,:],
            'type2':skilled_like[1,:],
            'type3':skilled_like[2,:]})

  # scaling
    xEd = xEduc.copy()
    for i in [0,1,2,7,8,9]:
        xEd[i]=xEd[i]/100

    grade_params_by_quality = np.array([[xEd[0:7]+xEd[14:16],
        xEd[7:14]+xEd[16:18]],[xEd[0:3]+xEd[32:36]+xEd[18:20],xEd[7:10]+
        xEd[36:40]+xEd[20:22]]],dtype=np.float64)

    year_four_intercept = xLabor[72]

    num_quantiles=20

    norm_quantiles=scipy.stats.norm.ppf(
        np.array(range(1,num_quantiles))/num_quantiles)

    # subset for only educational decisions
    DFDataEducation = DFData[(DFData.numeric_state<7)]
     
    school_likes=np.zeros(
        [num_endowment_types,DFDataEducation.shape[0],2],dtype=np.float64)
    school_likes_row=np.zeros([num_endowment_types,DFDataEducation.shape[0]],
        dtype=np.float64)

    ability_by_type = [np.zeros(2,dtype=np.float64) 
    for x in range(num_endowment_types)]
    for x in range(1,num_endowment_types):
        ability_by_type[x]=np.array(xEd[(26+2*x):(28+2*x)],dtype=np.float64)

    ed_switching_costs = np.array([xEd[40],xEd[41]],dtype=np.float64)
    univ_type_shifters = np.array(xEd[42:48],dtype=np.float64)

    # grad_payoff = xEd[48]
    grad_payoff = xEd[48]

    for unobs_type in range(num_endowment_types):

        STEM_payouts_by_quality = STEM_payouts[unobs_type]
        nonSTEM_payouts_by_quality = nonSTEM_payouts[unobs_type]

        # first year out of college nonpecuniary cost
        # needs to be flow utility - switching cost + zero exp penalty
        flow_unskilled_penalized = xLabor[48]
        year_four_flow_penalized = xLabor[76]

        flowsFull = np.array([0,0,0,flow_unskilled_penalized,
            xEd[22+unobs_type*2],xEd[23+unobs_type*2]],dtype=np.float64)

        school_likes[unobs_type]=fs.LogLikeEducation(
            DFDataEducation,SATTuition,grad_horizon,
            sectors,beta,ability_by_type[unobs_type],
            flowsFull,unskilled_var,grade_params_by_quality,
            normReps,simReps,LaborGradeRange,final_size,dropout_payouts,
            STEM_payouts_by_quality, nonSTEM_payouts_by_quality,
            gamma_p, unskilled_meanvar, norm_quantiles,
            wage_coeffs_full_by_type[0][:-1],
            wage_coeffs_full_by_type[0][-1],
            skilled_wage_covar,LaborGradeInt,
            choose,year_four_intercept,year_four_flow_penalized,
            ed_switching_costs,univ_type_shifters,grad_payoff,True)
        school_likes_row = np.sum(school_likes,axis=2)



    num_students=len(DFDataEducation.groupby('id'))

    individual_school_like=np.zeros([num_endowment_types,num_students])
    for unobs_type in range(num_endowment_types):
        individual_school_like[unobs_type]=(DFDataEducation.assign(
        loglike=school_likes_row[unobs_type]).groupby('id').\
        apply(lambda x: np.sum(x.loglike)))

    # 3 types!
    school_like_by_id = pd.DataFrame({'id':np.unique(
        DFDataEducation.index.values),
        'type1':individual_school_like[0,:],
        'type2':individual_school_like[1,:],
        'type3':individual_school_like[2,:]})

    school_like_by_id=school_like_by_id.set_index('id')
    skilled_like_by_id=skilled_like_by_id.set_index('id')

    # combine type specific likelihood for skilled and education observations
    posteriors_total = skilled_like_by_id.add(school_like_by_id,fill_value=0)

    # normalize with posterior
    posteriors_total_arr=np.array([posteriors_total.type1,
        posteriors_total.type2,posteriors_total.type3])

    posteriors_by_id=normalize_posterior(posteriors_total_arr,
        endowment_dist_old)

    endowment_dist_new=np.mean(posteriors_by_id,axis=1)



    posteriors_endowment=np.zeros([len(DFDataSkilled),num_endowment_types])

    # calculate posterior probability of being pref type i (for LABOR calculation)
    for i in range(num_endowment_types):
        posterior_frame=pd.DataFrame({'id':np.unique(DFData.index.values),
            'data':posteriors_by_id[i]})
        posterior_frame=posterior_frame.set_index('id')
        posteriors_endowment[:,i]=np.array(DFDataSkilled.merge(posterior_frame,
            left_index=True,right_index=True).data)

    skilled_subset_by_id = DFDataSkilled.reset_index().groupby('id').first()
    posteriors_skilled_subset = np.zeros([len(skilled_subset_by_id),
        num_endowment_types])
    for i in range(num_endowment_types):
        posterior_frame=pd.DataFrame({'id':np.unique(DFData.index.values),
            'data':posteriors_by_id[i]})
        posterior_frame=posterior_frame.set_index('id')
        posteriors_skilled_subset[:,i]=np.array(skilled_subset_by_id.merge(
            posterior_frame,
            left_index=True,right_index=True).data)

    num_types = 3
    horizon=20

    return LaborLikeEM(xLabor,DFDataSkilled,SATTuition,college_values,
        grad_horizon,num_types,sectors,gamma_p,beta,ability,normReps,simReps,
        LaborGradeRange,final_size,horizon,posteriors_endowment)
