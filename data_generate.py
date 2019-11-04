# generates bootstrap estimates
# output is a data frame

import numpy as np
import pandas as pd
import itertools
import scipy.stats
import scipy.interpolate
import os
import random
import math
import simplexmap

from EmaxLaborFunctionsJITUnskilled import EmaxLaborFunctionsJITUnskilled
from EmaxLaborFunctionsJIT import EmaxLaborFunctionsJIT
from ForwardSimLabor import LaborSolve, ForwardSimulateEducation

import FSLikelihoodJITfinal as fs

# from DataPreProcessApprox import *


# number of observations
num_individuals = 306

# import transcript with type probabilities
hs_transcript = pd.read_csv(os.path.abspath(os.curdir)+
    '/year_one_transcript.csv')

bootstrap_iterations = 2
# draw individuals and their type
total_rows = num_individuals*bootstrap_iterations
random_rows = random.choices(range(num_individuals),k=total_rows)

df_base = hs_transcript.iloc[random_rows]
type_vector=np.zeros(total_rows,dtype=np.int64)

hs_transcript['tuition']=round(hs_transcript['tuition'],3)
SATTuition=list(set(list(hs_transcript[
    ['A_N','A_S','SAT_M','SAT_V','hs_GPA','quality','tuition','univ_type_num']
    ].itertuples(index=False,name=None))))


def education_emax_mapping(A_N,A_S,SAT_M,SAT_V,hs_GPA,quality,tuition,univ_type,
    SATTuition):
    out=np.zeros(len(A_N),dtype=np.int64)
    for x in range(len(A_N)):
        key=tuple([A_N[x],A_S[x],SAT_M[x],SAT_V[x],hs_GPA[x],quality[x],
            tuition[x],univ_type[x]])
        out[x]=SATTuition.index(key)
    return out


hs_transcript['ed_emax_mapping']=education_emax_mapping(
    np.array(hs_transcript.A_N),np.array(hs_transcript.A_S),
    np.array(hs_transcript.SAT_M),np.array(hs_transcript.SAT_V),
    np.array(hs_transcript.hs_GPA),np.array(hs_transcript.quality),
    np.array(hs_transcript.tuition),np.array(hs_transcript.univ_type_num),
    SATTuition)



# variable initializations
num_endowment_types = 3
sectors=3
grad_horizon=30 # number of years after grad over which individual can work
final_size=int((grad_horizon+2)*(grad_horizon+1)*grad_horizon/6) # experience allocations
num_types = 3


reps_per_person = 30
stop_time = 15 # when to stop generating output for the model. 

beta=0.90 # discount factor
normReps=4 # perform sectors^normReps draws to integrate
LaborGradeRange=np.linspace(2,4,11)
simReps=9
horizon_later = 15 
horizon = 10


# assign types

for x in range(total_rows):
    df_row = df_base.iloc[x]
    type_vector[x]=scipy.stats.rv_discrete(values=(np.arange(num_endowment_types),
    [df_row.type1,df_row.type2,df_row.type3])).rvs(size=1)

df = df_base.assign(unobs_type=type_vector)


# Labor Parameters

x=[None]*81
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
x[  48] =   3.074e-01-1.74504648900e+00-1.06002212772e-01

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


x[  58] =  -10.00000000000
x[  59] =  -10.00000000000
x[  60] =  -10.00000000000

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
x[  72] =   2.0252

# switching costs
x[  73] =   2.67037140390e-00
x[  74] =   1.72109708887e-00
x[  75] =   1.96173635149e-00


# year four unskilled flow utility
x[76] = -2.78146963456e-02-2.70327516104-1.06002212772e-01

# unskilled switching cost, years 3-4
x[77] = 1.74504648900
# unskilled switching cost, year 4
x[78] = 2.70327516104
# unskilled zero experience penalty, same for all years
x[79] = -1.06002212772e-01

# year 4 one unit of experience term
x[80] = .4973
xLabor = x.copy()


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
X[0]=  1.00026490154e-01
X[1]=  1.83258907025e-01
X[2]=  3.72008885365e-01
X[3]=  9.74487029675e-02
X[4]=  2.76622777762e-01
X[5]=  3.68126207676e-01
X[6]=  5.68023505068e-01
X[7]=  5.20518836818e-02
X[8]=  1.07495457989e-01
X[9]=  4.17848178298e-01
X[10]=  6.68134933365e-01
X[11]=  6.31022596078e-01
X[12]=  7.03387301461e-01
X[13]=  8.88369374044e-01
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

xEduc = X.copy()
for x in [0,1,2,7,8,9]:
    xEduc[x]=xEduc[x]/100

    


def GenerateEmaxLaborFunctions(horizon, horizon_later,skilled_wage_covar,
    gamma_p,beta,skilled_wage_coeffs,flows_penalized,wageshocks,major,
    grade,choose,quality,switching_costs_skilled, wageshocks_later,
    zero_exp_penalty):
    skilled_flows_penalized = flows_penalized[0:sectors]
    time_zero_flows_penalized = (flows_penalized[(sectors+3):(2*sectors+3)])


    output=EmaxLaborFunctionsJIT(horizon,horizon_later-1,gamma_p,beta,
        np.array(skilled_wage_coeffs,dtype=np.float64),major,grade,
        skilled_flows_penalized,wageshocks,choose,quality,
        time_zero_flows_penalized,switching_costs_skilled,zero_exp_penalty,
        wageshocks_later)
    output.solveLabor()
    return output.EmaxList


# education parsing function
def GenerateEmaxEducationFunctions(unobs_type, quality, dropout_payouts,
    STEM_payouts_dict,nonSTEM_payouts_dict,grade_params_by_quality,gamma_p,beta,
    ed_flows_penalized_by_type,ability_by_type,exog_chars,unskilled_meanvar,
    norm_quantiles,year_four_intercept,year_four_flow_penalized,yearly_tuition,
    ed_switching_costs,univ_type_shifters,univ_type_num,grad_payoff):
    ability=ability_by_type[unobs_type]
    grade_params = grade_params_by_quality[quality]
    tuition = np.array([yearly_tuition,yearly_tuition,yearly_tuition,
        yearly_tuition],type=np.float64)

    STEM_payouts_array=STEM_payouts_dict[(unobs_type,quality)]
    nonSTEM_payouts_array=nonSTEM_payouts_dict[(unobs_type,quality)]
    flows_penalized = ed_flows_penalized_by_type[unobs_type].\
    astype(np.float64).copy()

    if univ_type_num>0:
        flows_penalized[0]=(flows_penalized[0]+
            univ_type_shifters[2*univ_type_num-2])
        flows_penalized[1]=(flows_penalized[1]+
            univ_type_shifters[2*univ_type_num-1])

    solution = EmaxEducationJIT(dropout_payouts,STEM_payouts_array,
        nonSTEM_payouts_array, grade_params, gamma_p, beta, flows_penalized,
        tuition,exog_chars,ability, unskilled_meanvar, norm_quantiles,
        year_four_intercept,year_four_flow_penalized,
        ed_switching_costs,grad_payoff)
    solution.solve()
    return solution.EmaxEducationValues
    
    
    
    
    
grade_params_by_quality = np.array([[xEduc[0:7]+xEduc[14:16],
    xEduc[7:14]+xEduc[16:18]],[xEduc[0:3]+xEduc[32:36]+xEduc[18:20],
    xEduc[7:10]+xEduc[36:40]+xEduc[20:22]]],
    dtype=np.float64)
num_quantiles=20

norm_quantiles=scipy.stats.norm.ppf(
    np.array(range(1,num_quantiles))/num_quantiles)
ability_by_type = [np.zeros(2,dtype=np.float64)
 for x in range(num_endowment_types)]
for x in range(1,num_endowment_types):
    ability_by_type[x]=np.array(xEduc[(26+2*x):(28+2*x)],dtype=np.float64)


ed_switching_costs = np.array([xEduc[40],xEduc[41]],dtype=np.float64)
univ_type_shifters = np.array(xEduc[42:48],dtype=np.float64)
grad_payoff = xEduc[48]

# Parse labor parameters
gamma_p=xLabor[30]


zero_exp_penalty = np.array(xLabor[68:71],dtype=np.float64)


# iterate over both endowment and preference types
flowsUnskilled=np.array([0,0,0,xLabor[48],0,0],dtype=np.float64) 
wage_coeffs_full_by_type=[np.array([[0]+xLabor[0:9]+[xLabor[61]],
[0]+xLabor[9:18]+[xLabor[62]],[0]+xLabor[18:27]+[xLabor[63]],
[xLabor[31],0,0,0,xLabor[32],0,0,xLabor[33],xLabor[34],0,0]],dtype=np.float64) 
for i in range(4)]
for i in range(4):
    for j in range(3):
        wage_coeffs_full_by_type[i][j][0]=xLabor[36+3*i+j]
skilled_wage_covar=np.array([[xLabor[27],0,0],[0,xLabor[28],0],[0,0,xLabor[29]]])

year_four_intercept = xLabor[72]
year_four_flow_penalized = xLabor[76]
year_four_one_exp = xLabor[80]

LaborEmax={}
choose = simplexmap.pascal(grad_horizon+4,sectors)


zscores=scipy.stats.norm.ppf(np.array(range(1,normReps+1))/(normReps+1))
base_draws=np.matrix.transpose(np.matrix(list(
    itertools.product(zscores,repeat=(sectors)))))
lmat=np.linalg.cholesky(skilled_wage_covar)
wageshocks=np.array(np.transpose(np.matmul(lmat,base_draws)))

normReps_later = 2
zscores_later=scipy.stats.norm.ppf(np.array(range(1,normReps_later+1))/
    (normReps_later+1))
base_draws_later=np.matrix.transpose(np.matrix(list(
    itertools.product(zscores_later,repeat=(sectors)))))
lmat=np.linalg.cholesky(skilled_wage_covar)
wageshocks_later=np.array(np.transpose(np.matmul(lmat,base_draws_later)))



switching_costs_skilled = np.array(xLabor[73:76],dtype=np.float64)

flows_skilled_by_type_penalized = [np.zeros(2*sectors+3,
    dtype=np.float64) for i in range(num_endowment_types)]
for i in range(num_endowment_types):
    flows_skilled_by_type_penalized[i][0:3]=(xLabor[(49+3*i):(52+3*i)]-
        switching_costs_skilled)
    flows_skilled_by_type_penalized[i][(sectors+3):(2*sectors+3)]=(
        [xLabor[49+3*i]+xLabor[65],xLabor[50+3*i]+xLabor[66],
        xLabor[51+3*i]+xLabor[67]]-switching_costs_skilled)



# heterogeneity in STEM and non-STEM payouts by type, but no type het in
# unskilled payouts
LaborFinal=np.linspace(200,400,21,dtype=np.int64)

STEM_payouts=np.zeros([num_endowment_types,2,len(LaborFinal)],dtype=np.float64)
nonSTEM_payouts=np.zeros([num_endowment_types,2,len(LaborFinal)],
    dtype=np.float64)


for endowment_type in range(num_endowment_types):
    for quality in range(2):
        for dSTEM in range(2):
            for grade in LaborGradeRange:
                LaborEmax[(endowment_type,quality,dSTEM,int(100*grade))]=\
                GenerateEmaxLaborFunctions(horizon,horizon_later,
                    skilled_wage_covar,gamma_p,beta,
                    wage_coeffs_full_by_type[endowment_type][:-1],
                    flows_skilled_by_type_penalized[endowment_type],
                    wageshocks,dSTEM,grade,choose,quality,
                    np.array(switching_costs_skilled,dtype=np.float64),
                    wageshocks_later,zero_exp_penalty)

# Fully solve the unskilled labor market, over the 4 dropout times
flowUnskilled = xLabor[48]-xLabor
unskilled_reps=20

unskilled_var=np.array([[xLabor[35]]],dtype=np.float64)
unskilled_wage_shocks=np.array(np.transpose(np.matrix(scipy.stats.norm.ppf(
    np.array(range(1,unskilled_reps+1))/
    (unskilled_reps+1)))))*unskilled_var[0][0]**0.5

unskilled_wage_coeffs = wage_coeffs_full_by_type[0][-1]

unskilled_switching_cost = np.array([xLabor[77]],dtype=np.float64)
year_four_switching_cost = np.array([xLabor[78]],dtype=np.float64)
unskilled_zero_exp_penalty = np.array([xLabor[79]],dtype=np.float64)

# Solve the recursive problem
dropout_payouts=np.zeros((4,2),dtype=np.float64)

unskilled_Emax=[None]*4
for drop_time in range(4):
    if drop_time==3:
        unskilled_coeffs_final=unskilled_wage_coeffs.copy()
        unskilled_coeffs_final[0][0]=year_four_intercept
        unskilled_coeffs_final[0][8]=year_four_one_exp

        unskilled=EmaxLaborFunctionsJITUnskilled(grad_horizon+4-drop_time-1,
            gamma_p,beta,np.array([unskilled_coeffs_final]),0,0,
            np.array([year_four_flow_penalized]),unskilled_wage_shocks,choose,
            np.array([year_four_switching_cost],dtype=np.float64),
            unskilled_zero_exp_penalty,0)
    else:
        unskilled=EmaxLaborFunctionsJITUnskilled(grad_horizon+4-drop_time-1,
            gamma_p,beta,np.array([unskilled_wage_coeffs]),0,0,
            np.array([flowUnskilled]),unskilled_wage_shocks,choose,
            np.array([unskilled_switching_cost],dtype=np.float64),
            unskilled_zero_exp_penalty,dtype=np.float64),0)
    unskilled.solveLabor()
    dropout_payouts[drop_time,0]=unskilled.EmaxList[1,0,1]
    dropout_payouts[drop_time,1]=unskilled.EmaxList[1,1,0]
    unskilled_Emax[drop_time]=unskilled.EmaxList

unskilled_Emax=tuple(unskilled_Emax)
# dropout_payouts[0,0]=17.95723211
# dropout_payouts[0,1]=19.99124993
# dropout_payouts[1,0]=17.88247618
# dropout_payouts[1,1]=19.91649401
# dropout_payouts[2,0]=17.79941404
# dropout_payouts[2,1]=19.83343187
# dropout_payouts[3,0]=15.73479004
# dropout_payouts[3,1]=18.99858802


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
        nonSTEM_payouts[endowment_type,quality,:]=scipy.interpolate.interp1d(
            LaborGradeInt,nonSTEM_payouts_raw[endowment_type,quality,:],
            kind='cubic')(LaborFinal)


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

ShockPref=np.random.gumbel(size=(df.shape[0],grad_horizon+4,max([sectors+1,4])))

# wage shocks
ShockSkilledWage=np.random.multivariate_normal([0]*sectors,
    cov=skilled_wage_covar,size=(df.shape[0],grad_horizon))


ShocksDict={'pref':ShockPref,'skilled':ShockSkilledWage}


# unskilled payouts
# unskilled_reps=20
# unskilled_wage_shocks=(unskilled_var**0.5*
# np.array(np.transpose(np.matrix(scipy.stats.norm.ppf(
#     np.array(range(1,unskilled_reps+1))/(unskilled_reps+1))))))
# unskilled_meanvar=np.array([wage_coeffs_full_by_type[0][-1][0],unskilled_var],
#     dtype=np.float64)
# num_quantiles=20
# norm_quantiles=scipy.stats.norm.ppf(
#     np.array(range(1,num_quantiles))/num_quantiles)

# dropout_payouts[0,0]=17.95723211
# dropout_payouts[0,1]=19.99124993
# dropout_payouts[1,0]=17.88247618
# dropout_payouts[1,1]=19.91649401
# dropout_payouts[2,0]=17.79941404
# dropout_payouts[2,1]=19.83343187
# dropout_payouts[3,0]=15.73479004
# dropout_payouts[3,1]=18.99858802



# Solve the type-specific education problem for each unobserved type. 
LaborFinal=np.linspace(200,400,21,dtype=np.int64)


STEM_payouts_dict={}
nonSTEM_payouts_dict={}
for unobs_type in range(num_types):
    for quality in range(2):

        STEM_payouts_dict[(unobs_type,quality)]=(
            STEM_payouts[(unobs_type,quality)])
        nonSTEM_payouts_dict[(unobs_type,quality)]=(
            nonSTEM_payouts[(unobs_type,quality)])


ed_flows_penalized_by_type = np.array([[xEduc[22]-ed_switching_costs[0],
    xEduc[23]-ed_switching_costs[1],xLabor[48]],[
    xEduc[24]-ed_switching_costs[0],xEduc[25]-ed_switching_costs[1],xLabor[48]],
    [xEduc[26]-ed_switching_costs[0],xEduc[27]-ed_switching_costs[1],
    xLabor[48]]])

grad_payoff = xEduc[48]

ed_Emax = {}


for x in SATTuition:
    SAT_M = x[2]
    SAT_V = x[3]
    hs_GPA = x[4]
    quality = x[5]
    yearly_tuition=x[6]
    univ_type_num = x[7]
    exog_chars=np.array([SAT_M,SAT_V,hs_GPA],dtype=np.float64)
    for unobs_type in range(num_types):
        ed_Emax[(unobs_type,quality,SAT_M,SAT_V,hs_GPA,yearly_tuition)]=(
            GenerateEmaxEducationFunctions(unobs_type,
            quality, dropout_payouts,STEM_payouts_dict,nonSTEM_payouts_dict,
            grade_params_by_quality,gamma_p,beta,ed_flows_penalized_by_type,
            ability_by_type,exog_chars,unskilled_meanvar,norm_quantiles,
            year_four_intercept,year_four_flow_penalized,yearly_tuition,
            ed_switching_costs,univ_type_shifters,univ_type_num,grad_payoff))



# START SIMULATING

# grade shocks
num_shocks=reps_per_person*df.shape[0]
ShockGradeRaw=np.random.normal(0,1,(num_shocks,4))

# preference shocks
ShockPref=np.random.gumbel(size=(num_shocks,grad_horizon+4,max([sectors+1,4])))

# wage shocks
ShockSkilledWage=np.random.multivariate_normal([0]*sectors,
    cov=skilled_wage_covar,size=(num_shocks,grad_horizon))
ShockUnskilledWage=(np.random.normal(0,1,(num_shocks,grad_horizon+4))*
    unskilled_meanvar[1]**0.5)

ShocksDict={'grade':ShockGradeRaw,'pref':ShockPref,'skilled':ShockSkilledWage,
'unskilled':ShockUnskilledWage}


# Run for each row of DFChars, then aggregate the results
num_simulations = df.shape[0]
finaloutput=[None]*num_simulations

labor_dict = []

finaloutput=[None]*num_simulations
for x in range(num_simulations):
    finaloutput[x]=ForwardSimulateEducation(df,x,ed_Emax,
        dropout_payouts,unskilled_meanvar,grade_params_by_quality,gamma_p,
        beta,ed_flows_penalized_by_type,ability_by_type,norm_quantiles,
        ShocksDict,STEM_payouts_dict,nonSTEM_payouts_dict,year_four_intercept,
        year_four_flow_penalized,zero_exp_penalty,wage_coeffs_full_by_type,
        ed_switching_costs,univ_type_shifters,grad_payoff,
        flows_skilled_by_type_penalized,switching_costs_skilled)


largeDF_no_index=df.reset_index()
flatdict=[]

id_big = []
unobs_type_big = []
choice_big = []
outcome_big = []
shock_big = []
time_big = []
type_big = []
sim_big = []
tdropout_big = []
hpskilled_big = []
skilled1_big = []
skilled2_big = []
skilled3_big = []

for x in range(num_simulations):
    num_obs = len(finaloutput[x])+1
    id_list = [largeDF_no_index.iloc[x].id]*num_obs
    unobs_type_list = [largeDF_no_index.iloc[x].unobs_type]*num_obs
    choice_list = [largeDF_no_index.iloc[x].choice]
    outcome_list = [largeDF_no_index.iloc[x].outcome]
    sim_list = [x]*num_obs
    shock_list = [0]
    time_list = [1]
    type_list = ['grade']
    skilled1_list = [0]
    skilled2_list = [0]
    skilled3_list = [0]
    hpskilled_list = [0]
    for y in finaloutput[x].keys():
        time_list.append(y)
        choice_list.append(finaloutput[x][y]['choice'])
        if 'grade' in finaloutput[x][y]:
            type_list.append('grade')
            shock_list.append(finaloutput[x][y]['shock'])
            outcome_list.append(finaloutput[x][y]['grade'])
            hpskilled_list.append(0)
            skilled1_list.append(0)
            skilled2_list.append(0)
            skilled3_list.append(0)

        elif 'lwage' in finaloutput[x][y]:
            type_list.append('lwage')
            shock_list.append(finaloutput[x][y]['shock'])
            outcome_list.append(finaloutput[x][y]['lwage'])
            if finaloutput[x][y]['choice']=='unskilled':
                hpskilled_list.append(0)
                skilled1_list.append(0)
                skilled2_list.append(0)
                skilled3_list.append(0)
            else:
                hpskilled_list.append(finaloutput[x][y]['hp'])
                skilled1_list.append(finaloutput[x][y]['skilled1'])
                skilled2_list.append(finaloutput[x][y]['skilled2'])
                skilled3_list.append(finaloutput[x][y]['skilled3'])

        else:
            type_list.append('hp')
            shock_list.append(0)
            outcome_list.append(0)
            hpskilled_list.append(0)
            skilled1_list.append(0)
            skilled2_list.append(0)
            skilled3_list.append(0)

    sim_big.extend(sim_list)
    id_big.extend(id_list)
    unobs_type_big.extend(unobs_type_list)
    choice_big.extend(choice_list)
    outcome_big.extend(outcome_list)
    shock_big.extend(shock_list)
    time_big.extend(time_list)
    type_big.extend(type_list)
    hpskilled_big.extend(hpskilled_list)
    skilled1_big.extend(skilled1_list)
    skilled2_big.extend(skilled2_list)
    skilled3_big.extend(skilled3_list)


df_educ_out=pd.DataFrame({'id':id_big,'unobs_type':unobs_type_big,'time':time_big,
    'choice':choice_big,'outcome':outcome_big,'choice_type':type_big,
    'shock':shock_big,'hpskilled':hpskilled_big,'skilled1':skilled1_big,
    'skilled2':skilled2_big,'skilled3':skilled3_big,'sim_num':sim_big})
df_educ_out=df_educ_out.set_index('id')

# Rejoin results to HS GPA and SAT Scores
df_chars=df.groupby('id').first()[['SAT_M','SAT_V','hs_GPA','quality','tuition','univ_type']]
df_educ_out_inter=df_educ_out.merge(df_chars,how='left',on='id')


def terminalmajorgpa(x):
    length=x.shape[0]
    if length<4:
        return pd.DataFrame({'tmajor':['dropout'],\
            'terminalGPA':[0]})
    if x.choice.iloc[3]=='STEM':
        return pd.DataFrame({'tmajor':['STEM'],\
            'terminalGPA':[np.mean(x.outcome.iloc[0:4])]})
    elif x.choice.iloc[3]=='nonSTEM':
        return pd.DataFrame({'tmajor':['nonSTEM'],\
            'terminalGPA':[np.mean(x.outcome.iloc[0:4])]})
    else:
        return pd.DataFrame({'tmajor':['dropout'],\
            'terminalGPA':[0]})

terminal =df_educ_out_inter.groupby('sim_num').apply(terminalmajorgpa).reset_index()
df_educ_out_final = df_educ_out_inter.reset_index().merge(terminal,how='left',on='sim_num')




# solve labor market 
df_labor = pd.DataFrame(labor_dict)
ShockPref_labor=np.random.gumbel(size=(df_labor.shape[0],grad_horizon+4,max([sectors+1,4])))

# wage shocks
ShockSkilledWage_labor=np.random.multivariate_normal([0]*sectors,
    cov=skilled_wage_covar,size=(df_labor.shape[0],grad_horizon))


ShocksDict_labor={'pref':ShockPref_labor,'skilled':ShockSkilledWage_labor}

labor_output={}
for x in range(df_labor.shape[0]):
    major = df_labor.dSTEM.iloc[x]
    grade = df_labor.tGPA.iloc[x]
    quality = df_labor.quality.iloc[x]
    unobs_type = df_labor.unobs_type.iloc[x]
    out = {}
    flows_skilled_penalized_val = (flows_skilled_by_type_penalized[unobs_type][0:sectors])
    time_zero_flows_penalized_val = (flows_skilled_by_type_penalized[unobs_type][(sectors+3):
        (2*sectors+3)])

    LaborSolve(x,5,grad_horizon+4,stop_time,major,grade,(0,0,0,0),
        LaborEmax[(unobs_type,quality,major,grade)],
        ShocksDict_labor,out,wage_coeffs_full_by_type[unobs_type],
        flows_skilled_penalized_val,gamma_p,beta,choose,quality,
        time_zero_flows_penalized_val,5,zero_exp_penalty,
        switching_costs_skilled,None)
    labor_output[x] = out


# now for all the output, solve for 

flatdict_labor=[]
# Turn this output into a dataframe
for x in labor_output.keys():
    for y in labor_output[x].keys():
        if 'grade' in labor_output[x][y]:
            flatdict_labor.append({'id':x,'time':y,'major':labor_output.dSTEM.iloc[x],
                'gpa':df_labor.tGPA.iloc[x],
                'choice':labor_output[x][y]['choice'],'type':'grade',\
                'outcome':labor_output[x][y]['grade'],\
                'shock':labor_output[x][y]['shock']})
        elif 'lwage' in labor_output[x][y]:
            flatdict_labor.append({'id':x,'time':y,'major':df_labor.dSTEM.iloc[x],
                'gpa':df_labor.tGPA.iloc[x],
                'choice':labor_output[x][y]['choice'],'type':'lwage',\
                'outcome':labor_output[x][y]['lwage'],\
                'shock':labor_output[x][y]['shock']})
        else:
            flatdict_labor.append({'id':x,'time':y,'major':df_labor.dSTEM.iloc[x],
                'gpa':df_labor.tGPA.iloc[x],
                'choice':labor_output[x][y]['choice'],'type':'hp',\
                'outcome':0,'shock':0})

results_labor=pd.DataFrame(flatdict_labor)
results_labor.sort_values(['id','time'])



# hard coded length
sim_nums = np.zeros(len(labor_dict),dtype=np.int64)
cf_nums = np.zeros(len(labor_dict),dtype=np.int64)

for x in range(len(labor_dict)):
    sim_nums[x]=labor_dict[x]['sim_num']
    cf_nums[x] = sim_nums[x] // (num_individuals)

df_sim_nums=pd.DataFrame({'sim_num':sim_nums.repeat(stop_time-4, axis=0),
    'counterfactual_num':cf_nums.repeat(stop_time-4,axis=0)})


df_labor_out=pd.concat([results_labor.reset_index(),df_sim_nums],axis=1)


fullCategories=['STEM','nonSTEM','unskilled','hp']+\
['skilled'+str(x) for x in range(1,sectors+1)]
expCategories=['unskilled']+['skilled'+str(x) for x in range(1,sectors+1)]


df_labor_outShort=df_labor_out.set_index('id')
df_labor_outShort['choicecat']=df_labor_outShort.choice.astype(pd.api.types.CategoricalDtype(
        categories=fullCategories))
def AddExpCols(x):
    return x.shift().fillna(0).cumsum().astype(int)
v = pd.get_dummies(df_labor_outShort.choicecat).groupby(level=0).apply(AddExpCols)



# Pre Process


df_educ_rbind = df_educ_out_final[['sim_num','time','SAT_M','SAT_V','choice',
'outcome','choice_type','tuition','quality','tmajor','terminalGPA','hs_GPA',
'univ_type']]

df_educ_rbind['type']=df_educ_rbind['choice_type']
df_educ_rbind=df_educ_rbind.drop(columns='choice_type')


df_labor_rbind=pd.merge(df_labor_out.set_index('sim_num'),
    df_educ_rbind.groupby('sim_num').first()[[
    'SAT_M','SAT_V','tuition','quality','tmajor','terminalGPA','hs_GPA',
    'univ_type']],left_on='sim_num',
         right_on='sim_num').reset_index().drop(
         columns=['id','counterfactual_num'])[['sim_num','time','SAT_M','SAT_V','choice',
'outcome','tuition','quality','tmajor','terminalGPA','hs_GPA',
'univ_type','type']]

DFData=pd.concat([df_educ_rbind,df_labor_rbind])
DFData=DFData.rename(columns={'sim_num':'id'})
DFData=DFData.set_index(['id','time'])

DFData['A_N']=0
DFData['A_S']=0
DFData['augmented']=0

fullCategories=['STEM','nonSTEM','unskilled','hp']+\
['skilled'+str(x) for x in range(1,sectors+1)]
expCategories=['unskilled','hp']+['skilled'+str(x) for x in range(1,sectors+1)]
DFData['choicecat']=DFData.choice.astype(pd.api.types.CategoricalDtype(
    categories=fullCategories))

def AddExpCols(x):
    return x.shift().fillna(0).cumsum().astype(int)

v = pd.get_dummies(DFData.choicecat).groupby(level=0).apply(AddExpCols)
DFDataExp=pd.concat([DFData,v[expCategories]], 1).reset_index()
DFDataExp.set_index(['id','time'])

v2=DFDataExp.groupby('id').apply(terminalmajorgpa).reset_index().\
drop('level_1',axis=1)
DFDataExp=pd.merge(DFDataExp.drop(columns=['terminalGPA','tmajor']),v2,
    left_on='id',right_on='id')


v3=DFDataExp.groupby('id').apply(AddEducationState).reset_index().\
drop('level_1',axis=1)
DFDataFinal=pd.merge(DFDataExp,v3,left_on=['id','time'],
    right_on=['id','time']).reset_index()


raw_sector=DFDataFinal.choice.str.slice(-1)
for x in range(len(raw_sector)):
    if not raw_sector[x].isdigit():
        raw_sector[x]=None
    else:
        raw_sector[x]=int(raw_sector[x])-1

DFDataFinal['skilledsector']=raw_sector

# adds dropout times
DFDataFinal['tdropout']=DFDataFinal.id.map(
    DFDataFinal[DFDataFinal.choice.isin(['unskilled','hp'])].\
    drop_duplicates(['id'],keep='first').set_index('id').time).\
apply(lambda x: (int(x) if x<5 else -1))

# for students that fail out of college, need to remap their dropout time
DFDataFinal.loc[(DFDataFinal['tdropout']==-1) &
(DFDataFinal['terminalGPA']<200),'tdropout']=4

cumulativeGPAcol=DFDataFinal.groupby('id').apply(cumulativeGPA).\
reset_index().drop('level_1',axis=1)
DFDataFinal=pd.merge(DFDataFinal,cumulativeGPAcol,left_on=['id','time'],
    right_on=['id','time']).reset_index()

# Round all true terminal GPAs to the values used in the the discretization
# of the grade space in labor market outcomes
DFDataFinal['tGPA']=DFDataFinal.terminalGPA.apply(lambda x: int(\
    takeClosest(x,LaborGradeRange*100)))

# dropouts get assigned a terminal GPA of 0
DFDataFinal.loc[DFDataFinal['tdropout']!=-1,'tGPA']=0
DFDataFinal['univ_type_num']=(
    univ_type_numeric(np.array(DFDataFinal.univ_type)))

SATTuition=list(set(list(DFDataFinal[
    ['A_N','A_S','SAT_M','SAT_V','hs_GPA','quality','tuition',
    'univ_type_num']].itertuples(index=False,name=None))))

DFDataFinal['numeric_state']=numeric_state(DFDataFinal)


def gen_dSTEM(y):
    length=y.shape[0]
    out=np.zeros(length,dtype=np.int64)
    for x in range(length):
        var=y.tmajor.iloc[x]
        if var=='STEM':
            out[x]=1
    return out

DFDataFinal['dSTEM']=gen_dSTEM(DFDataFinal)
college_values = list(set(list(DFDataFinal[DFDataFinal.numeric_state==7][
    ['quality','dSTEM','tGPA']].itertuples(index=False,name=None))))

DFDataFinal['col_type']=col_type(DFDataFinal)
DFDataFinal['numeric_choice']=numeric_choice(DFDataFinal)

DFDataFinal['ed_emax_mapping']=education_emax_mapping(
    np.array(DFDataFinal.A_N),np.array(DFDataFinal.A_S),
    np.array(DFDataFinal.SAT_M),np.array(DFDataFinal.SAT_V),
    np.array(DFDataFinal.hs_GPA),np.array(DFDataFinal.quality),
    np.array(DFDataFinal.tuition),np.array(DFDataFinal.univ_type_num),
    SATTuition)
DFDataFinal['skilled_emax_mapping_abridged']=skilled_emax_mapping2(
    np.array(DFDataFinal.dSTEM), np.array(DFDataFinal.quality),
    np.array(DFDataFinal.numeric_state),
    np.array(DFDataFinal.tGPA),college_values)

DFDataFinal['unskilled_emax_mapping']=unskilled_emax_mapping(
    np.array(DFDataFinal.tdropout))
DFDataFinal['lastchoiceraw'] = DFDataFinal.groupby('id')['numeric_choice'].\
transform(lambda x:x.shift())
DFDataFinal['lastchoice'] = 0
DFDataFinal['lastchoice']=DFDataFinal.lastchoice.astype(np.int64)

def intconvert(x):
    if np.isnan(x):
        return -1
    return x

DFDataFinal['lastchoice']=np.vectorize(intconvert)(
    DFDataFinal.lastchoiceraw)

# hard coded for 3 sectors!
# x = 0,1,2 for sectors 1,2,3; 3 = hp, 4= unskilled
def skilled_convert(x):
    if x>=4:
        return x-4
    elif x==2:
        return 3
    elif x==3:
        return 4
    return -1
DFDataFinal['prior_skilled']=np.vectorize(skilled_convert)(
    DFDataFinal.lastchoice)

DFDataFinal.to_csv(os.path.abspath(os.curdir)+'/sim_data.csv')