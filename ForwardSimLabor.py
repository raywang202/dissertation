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
import FSLikelihoodJITfinal as fs

#===============================================================================
# Calculates the log-wage over all sectors, given skilled wage coefficients,
# a skilled experience vector (including HP), major, GPA, school quality.
# Constraint is that quadratic in experience is such that if experience is past
# the 'peak', the return to experience is just the max of the quadratic
#===============================================================================

def ElogWage(skilled_wage_coeffs,experience,dSTEM,GPA,quality):
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

#===============================================================================
# Solves the model for the skilled labor market, taking in the various labor
# market parameters and 
#===============================================================================

def LaborSolve(individual_number, current_time, end_time, stop_time, major,
    grade, current_endowment, Emax, ShocksDict, out, wage_coeffs_full,
    flowsSkilled_penalized, gamma_p, beta, choose, quality,
    time_zero_flows_penalized, zero_time, zero_exp_penalty,
    skilled_switching_costs, prior_choice):

    skilled_wage_coeffs = wage_coeffs_full[:-1]

    flows_final = flowsSkilled_penalized.copy()

    # hard coded for 3 sectors
    if current_endowment[0]==0:
        flows_final[0]=flows_final[0]+zero_exp_penalty[0]
    if current_endowment[1]==0:
        flows_final[1]=flows_final[1]+zero_exp_penalty[1]
    if current_endowment[2]==0:
        flows_final[2]=flows_final[2]+zero_exp_penalty[2]

    if prior_choice=='skilled1':
        flows_final[0]=flows_final[0]+skilled_switching_costs[0]
    if prior_choice=='skilled2':
        flows_final[1]=flows_final[1]+skilled_switching_costs[1]
    if prior_choice=='skilled3':
        flows_final[2]=flows_final[2]+skilled_switching_costs[2]

    if current_time>stop_time:
        return

    if current_time==end_time:
        totalexp=np.sum(current_endowment)
        totalhp=current_endowment[-1]
        expvector=list(current_endowment[:-1])
        lwage={}
        choiceSet={}
        shock={}
        NextExp={}
        base_wage = ElogWage(skilled_wage_coeffs,current_endowment,
            major,grade,quality)

        for x in range(len(expvector)):
            CurrentExp=list(expvector)
            laborexp=CurrentExp[x]
            shock['skilled'+str(x+1)]=ShocksDict['skilled'][\
            individual_number][current_time-5][x]
            lwage['skilled'+str(x+1)]=base_wage[x]+shock['skilled'+str(x+1)]
            payout=(gamma_p*np.exp(lwage['skilled'+str(x+1)])+
                ShocksDict['pref'][individual_number][
                current_time-1][x]+flows_final[x])
            choiceSet['skilled'+str(x+1)]=payout

        choiceSet['hp']=(ShocksDict['pref'][individual_number][
            current_time-1][-1])
        choice=max(choiceSet,key=choiceSet.get)
        if choice=='hp':
            out[current_time]={'choice':'hp'}
        else:
            out[current_time]={'choice':choice,'lwage':lwage[choice],\
            'shock':shock[choice]}
        out[current_time]['hp']=choiceSet['hp']
        out[current_time]['skilled1']=choiceSet['skilled1']
        out[current_time]['skilled2']=choiceSet['skilled2']
        out[current_time]['skilled3']=choiceSet['skilled3']

        return

    # first year out of work
    elif current_time==zero_time:

        totalexp=np.sum(current_endowment)
        totalhp=current_endowment[-1]
        expvector=list(current_endowment[:-1])
        lwage={}
        choiceSet={}
        shock={}
        NextExp={}
        base_wage = ElogWage(skilled_wage_coeffs,current_endowment,
            major,grade,quality)
        for x in range(len(expvector)):
            CurrentExp=list(expvector)
            laborexp=CurrentExp[x]
            CurrentExp[x]=CurrentExp[x]+1
            NextExp['skilled'+str(x+1)]=tuple(CurrentExp+[totalhp])
            shock['skilled'+str(x+1)]=ShocksDict['skilled'][\
            individual_number][current_time-5][x]
            lwage['skilled'+str(x+1)]=base_wage[x]+shock['skilled'+str(x+1)]
            payout=(gamma_p*np.exp(lwage['skilled'+str(x+1)])+
                ShocksDict['pref'][individual_number][current_time-1][x]+
                time_zero_flows_penalized[x]+zero_exp_penalty[x]+
                beta*Emax[totalexp+1,
                simplexmap.combo_to_array(totalexp+1,
                    NextExp['skilled'+str(x+1)],choose),x])
            choiceSet['skilled'+str(x+1)]=payout

        # 3 is hard-coded. for 3 sectors
        hpNext=tuple(expvector)+(current_endowment[-1]+1,)
        choiceSet['hp']=(ShocksDict['pref'][individual_number][
            current_time-1][-1]+beta*Emax[totalexp+1,
            simplexmap.combo_to_array(totalexp+1,hpNext,choose),3])

        choice=max(choiceSet,key=choiceSet.get)

        if choice=='hp':
            out[current_time]={'choice':'hp'}
            out[current_time]['hp']=choiceSet['hp']
            out[current_time]['skilled1']=choiceSet['skilled1']
            out[current_time]['skilled2']=choiceSet['skilled2']
            out[current_time]['skilled3']=choiceSet['skilled3']
            LaborSolve(individual_number,current_time+1,end_time,stop_time,major,
                grade,hpNext,Emax,ShocksDict,out,wage_coeffs_full,
                flowsSkilled_penalized,gamma_p,beta,choose,quality,
                time_zero_flows_penalized,zero_time,
                zero_exp_penalty,skilled_switching_costs,'hp')
        else:
            out[current_time]={'choice':choice,'lwage':lwage[choice],\
            'shock':shock[choice]}
            out[current_time]['hp']=choiceSet['hp']
            out[current_time]['skilled1']=choiceSet['skilled1']
            out[current_time]['skilled2']=choiceSet['skilled2']
            out[current_time]['skilled3']=choiceSet['skilled3']

            LaborSolve(individual_number,current_time+1,end_time,stop_time,major,
                grade,NextExp[choice],Emax,ShocksDict,out,wage_coeffs_full,
                flowsSkilled_penalized,gamma_p,beta,choose,quality,
                time_zero_flows_penalized,zero_time,
                zero_exp_penalty,skilled_switching_costs,choice)
    # skilled sector
    else:
        totalexp=np.sum(current_endowment)
        totalhp=current_endowment[-1]
        expvector=list(current_endowment[:-1])
        lwage={}
        choiceSet={}
        shock={}
        NextExp={}
        base_wage = ElogWage(skilled_wage_coeffs,current_endowment,
            major,grade,quality)
        for x in range(len(expvector)):
            CurrentExp=list(expvector)
            laborexp=CurrentExp[x]
            CurrentExp[x]=CurrentExp[x]+1
            NextExp['skilled'+str(x+1)]=tuple(CurrentExp+[totalhp])
            shock['skilled'+str(x+1)]=ShocksDict['skilled'][\
            individual_number][current_time-5][x]
            lwage['skilled'+str(x+1)]=base_wage[x]+shock['skilled'+str(x+1)]
            payout=(gamma_p*np.exp(lwage['skilled'+str(x+1)])+
                ShocksDict['pref'][individual_number][current_time-1][x]+
                flows_final[x]+beta*Emax[totalexp+1,
                simplexmap.combo_to_array(totalexp+1,
                    NextExp['skilled'+str(x+1)],choose),x])
            choiceSet['skilled'+str(x+1)]=payout

        hpNext=tuple(expvector)+(current_endowment[-1]+1,)
        choiceSet['hp']=(ShocksDict['pref'][individual_number][
            current_time-1][-1]+beta*Emax[totalexp+1,
            simplexmap.combo_to_array(totalexp+1,hpNext,choose),3])

        choice=max(choiceSet,key=choiceSet.get)


        if choice=='hp':
            out[current_time]={'choice':'hp'}
            out[current_time]['hp']=choiceSet['hp']
            out[current_time]['skilled1']=choiceSet['skilled1']
            out[current_time]['skilled2']=choiceSet['skilled2']
            out[current_time]['skilled3']=choiceSet['skilled3']
            LaborSolve(individual_number,current_time+1,end_time,stop_time,major,
                grade,hpNext,Emax,ShocksDict,out,wage_coeffs_full,
                flowsSkilled_penalized,gamma_p,beta,choose,quality,
                time_zero_flows_penalized,zero_time,
                zero_exp_penalty,skilled_switching_costs,'hp')

        else:
            out[current_time]={'choice':choice,'lwage':lwage[choice],\
            'shock':shock[choice]}
            out[current_time]['hp']=choiceSet['hp']
            out[current_time]['skilled1']=choiceSet['skilled1']
            out[current_time]['skilled2']=choiceSet['skilled2']
            out[current_time]['skilled3']=choiceSet['skilled3']

            LaborSolve(individual_number,current_time+1,end_time,stop_time,major,
                grade,NextExp[choice],Emax,ShocksDict,out,wage_coeffs_full,
                flowsSkilled_penalized,gamma_p,beta,choose,quality,
                time_zero_flows_penalized,zero_time,
                zero_exp_penalty,skilled_switching_costs,choice)


def round_to(n, precision):
    correction = 0.5 if n >= 0 else -0.5
    return int( n/precision+correction ) * precision


# censored as well
def round_to_5(n):
    if n>=400:
        return 400
    if n<=0:
        return 0
    return round_to(n, 5)

def round_to_10(n):
    if n>=400:
        return 400
    if n<=0:
        return 0
    return round_to(n,10)

# Simulates the educational decisions of individuals, conditional on their
# quality, incoming grade, and type
# unskilled params is the mean and variance of earnings in unskilled sector
# uncomment code on 'LaborSolve' to turn on/off labor market simulations

def ForwardSimulateEducation(DFChars,individual_number,ed_Emax,dropout_payouts,
    unskilled_params,grade_params_by_quality,gamma_p,beta,
    ed_flows_penalized_by_type,ability_by_type,normquantiles,ShocksDict,
    STEM_payouts_dict,nonSTEM_payouts_dict,year_four_intercept,
    year_four_flow_penalized,zero_exp_penalty,wage_coeffs_full_by_type,
    ed_switching_costs,univ_type_shifters,grad_payoff,
    flows_skilled_by_type_penalized,switching_costs_skilled):
    # pull in the individual's characteristics
    SAT_M=int(DFChars.iloc[individual_number]['SAT_M'])
    SAT_V=int(DFChars.iloc[individual_number]['SAT_V'])
    hs_GPA = int(DFChars.iloc[individual_number]['hs_GPA'])
    t1grade=DFChars.iloc[individual_number]['outcome']

    t1choice = DFChars.iloc[individual_number]['choice']

    quality=int(DFChars.iloc[individual_number]['quality'])
    unobs_type=int(DFChars.iloc[individual_number]['unobs_type'])
    (A_S,A_N)=ability_by_type[unobs_type]
    grade_params= grade_params_by_quality[quality]
    tuition = round(DFChars.iloc[individual_number]['tuition'],3)

    univ_type_num = DFChars.iloc[individual_number]['univ_type_num']


    STEM_payouts = STEM_payouts_dict[(unobs_type,quality)]
    nonSTEM_payouts = nonSTEM_payouts_dict[(unobs_type,quality)]
    STEMsd_by_time=[grade_params[0][7],grade_params[0][8]]
    nonSTEMsd_by_time=[grade_params[1][7],grade_params[1][8]]
    EducationEmax=ed_Emax[(unobs_type,quality,SAT_M,SAT_V,hs_GPA,tuition)]
    (flowSTEM,flownonSTEM,flowUnskilled)=ed_flows_penalized_by_type[unobs_type]

    if univ_type_num>0:
        flowSTEM=flowSTEM+univ_type_shifters[2*univ_type_num-2]
        flownonSTEM=flownonSTEM+univ_type_shifters[2*univ_type_num-1]

    wage_coeffs_full = wage_coeffs_full_by_type[unobs_type]
    sectors = 3
    skilled_flows = flows_skilled_by_type_penalized[unobs_type][0:sectors]
    time_zero_flows = flows_skilled_by_type_penalized[unobs_type][(sectors+3):
    (2*sectors+3)]

    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return array[idx]
    def gpa_to_index(gpa):
        return int(gpa/5)

    def tgpa_to_index(gpa):
        return int((gpa-200)/10)
    # Now that all expectations are generated, let's solve it.

    # returns vector of future grade outcomes, rounded to nearest 0.05
    # currentGPA is to 3 decimal points (e.g. 300)
    # major=1 if STEM
    # may want some throw/catch for currentGPA being ridiculous
    def future_grade(grade_params,current_year,currentGPA,major,SAT_M,SAT_V,
        hs_GPA,A_S,A_N,normquantiles):
        if major==1:
            if current_year<=2:
                sigma = 100*grade_params[0][7]**0.5
            else:
                sigma = 100*grade_params[0][8]**0.5 
            egrade=(SAT_M*grade_params[0][0]+SAT_V*grade_params[0][1]+
                hs_GPA*grade_params[0][2]+grade_params[0][current_year+2]+A_S)
        else:
            if current_year<=2:
                sigma = 100*grade_params[1][7]**0.5
            else:
                sigma = 100*grade_params[1][8]**0.5 
            egrade=(SAT_M*grade_params[1][0]+SAT_V*grade_params[1][1]+
                hs_GPA*grade_params[1][2]+grade_params[1][current_year+2]+A_N)

        # generate next grade, rounded to nearest 5, and top/bottom capped
        # at 0, 400 GPA
        semGrades=(normquantiles*sigma+100*egrade)
        for x in range(len(semGrades)):
            if semGrades[x]>400:
                semGrades[x]=400
            elif semGrades[x]<0:
                semGrades[x]=0

        nextGrades=[round_to_5((currentGPA*(current_year-1)+x)/current_year) 
        for x in semGrades]
        return nextGrades

    def Egrade(grade_params,current_year,major,SAT_M,SAT_V,hs_GPA,A_S,A_N):
        if major==1:
            return (SAT_M*grade_params[0][0]+SAT_V*grade_params[0][1]+
                hs_GPA*grade_params[0][2]+grade_params[0][current_year+2]+A_S)
        else:
            return (SAT_V*grade_params[1][0]+SAT_V*grade_params[1][1]+
                hs_GPA*grade_params[1][2]+grade_params[1][current_year+2]+A_N)



    output={}

    # t=2
    # STEM picked in period 1
    if t1choice=='STEM':
        t2STEM=(beta*np.mean([EducationEmax[2,gpa_to_index(x),0]
            for x in future_grade(grade_params,2,t1grade,1,SAT_M,SAT_V,hs_GPA,
                A_S,A_N,normquantiles)])+
        ShocksDict['pref'][individual_number][1][0]+flowSTEM+
        ed_switching_costs[0]-gamma_p*tuition)

        t2nonSTEM=(beta*np.mean([EducationEmax[2,gpa_to_index(x),1]
            for x in future_grade(grade_params,2,t1grade,0,SAT_M,SAT_V,hs_GPA,
                A_S,A_N,normquantiles)])+
        ShocksDict['pref'][individual_number][1][1]+flownonSTEM-gamma_p*tuition)

    else:
        t2STEM=(beta*np.mean([EducationEmax[2,gpa_to_index(x),0]
            for x in future_grade(grade_params,2,t1grade,1,SAT_M,SAT_V,hs_GPA,
                A_S,A_N,normquantiles)])+
        ShocksDict['pref'][individual_number][1][0]+flowSTEM-gamma_p*tuition)

        t2nonSTEM=(beta*np.mean([EducationEmax[3,gpa_to_index(x),1]
            for x in future_grade(grade_params,2,t1grade,0,SAT_M,SAT_V,hs_GPA,
                A_S,A_N,normquantiles)])+ ed_switching_costs[1] +
        ShocksDict['pref'][individual_number][1][1]+flownonSTEM-gamma_p*tuition)

    t2work=(beta*dropout_payouts[1,1]+
        gamma_p*np.exp(unskilled_params[0]+
            ShocksDict['unskilled'][individual_number][1])+
        ShocksDict['pref'][individual_number][1][2]+flowUnskilled)
    
    t2hp=(beta*dropout_payouts[1,0]+
        ShocksDict['pref'][individual_number][1][3])


    # pick the best choice out of T=2
    t2choiceSet={'STEM':t2STEM,'nonSTEM':t2nonSTEM,'unskilled':t2work,
    'hp':t2hp}
    t2choice=max(t2choiceSet, key=t2choiceSet.get)

    # write down the observation
    if t2choice=='hp':
        output[2]={'choice':'hp'}
        return output
    elif t2choice=='STEM':
        t2grade=round_to_5(100*(Egrade(grade_params,2,1,SAT_M,SAT_V,hs_GPA,
            A_S,A_N)+ShockGradeRaw[individual_number][1]*STEMsd_by_time[0]))
        output[2]={'choice':'STEM','grade':t2grade,
        'shock':ShockGradeRaw[individual_number][1]*STEMsd_by_time[0]}
    elif t2choice=='nonSTEM':
        t2grade=round_to_5(100*(Egrade(grade_params,2,0,SAT_M,SAT_V,hs_GPA,
            A_S,A_N)+ShockGradeRaw[individual_number][1]*nonSTEMsd_by_time[0]))
        output[2]={'choice':'nonSTEM','grade':t2grade,
        'shock':ShockGradeRaw[individual_number][1]*nonSTEMsd_by_time[0]}
    else:
        output[2]={'choice':'unskilled','lwage':unskilled_params[0]+
        ShocksDict['unskilled'][individual_number][1],
        'shock':ShocksDict['unskilled'][individual_number][1]}
        return output

    # LOM to t3
    # meet STEM major requirement
    if t2choice=='STEM' or t1choice=='STEM':

        # prior choice t=2 STEM
        if t2choice=='STEM':
            t3STEM=(beta*np.mean([EducationEmax[4,gpa_to_index(x),0]
            for x in future_grade(grade_params,3,(t1grade+t2grade)/2,1,SAT_M,
                SAT_V,hs_GPA,A_S,A_N,normquantiles)])+ed_switching_costs[0]+
            ShocksDict['pref'][individual_number][2][0]+flowSTEM-
            gamma_p*tuition)

            t3nonSTEM=(beta*np.mean([EducationEmax[5,gpa_to_index(x),1]
                for x in future_grade(grade_params,3,(t1grade+t2grade)/2,0,
                    SAT_M,SAT_V,hs_GPA,A_S,A_N,normquantiles)])+
            ShocksDict['pref'][individual_number][2][1]+
            flownonSTEM-gamma_p*tuition)

        # prior choice t=2 nonSTEM
        else:
            t3STEM=(beta*np.mean([EducationEmax[4,gpa_to_index(x),0] 
                for x in future_grade(grade_params,3,(t1grade+t2grade)/2,1,
                    SAT_M,SAT_V,hs_GPA,A_S,A_N,normquantiles)])+
            ShocksDict['pref'][individual_number][2][0]+
            flowSTEM-gamma_p*tuition)

            t3nonSTEM=(beta*np.mean([EducationEmax[5,gpa_to_index(x),1] 
                for x in future_grade(grade_params,3,(t1grade+t2grade)/2,0,
                    SAT_M,SAT_V,hs_GPA,A_S,A_N,normquantiles)])+
            ed_switching_costs[1]+ShocksDict['pref'][individual_number][2][1]+
            flownonSTEM-gamma_p*tuition)

    # do not meet STEM major requirement (chose nonSTEM at t=2)
    else:
        t3nonSTEM=(beta*np.mean([EducationEmax[5,gpa_to_index(x),1]
            for x in future_grade(grade_params,3,(t1grade+t2grade)/2,0,SAT_M,
                SAT_V,hs_GPA,A_S,A_N,normquantiles)])+
        ed_switching_costs[1]+ShocksDict['pref'][individual_number][2][1]+
        flownonSTEM-gamma_p*tuition)

    t3work=(beta*dropout_payouts[2,1]+
        gamma_p*np.exp(unskilled_params[0]+
            ShocksDict['unskilled'][individual_number][2])+
        ShocksDict['pref'][individual_number][2][2]+flowUnskilled)
    
    t3hp=(beta*dropout_payouts[2,0]+
        ShocksDict['pref'][individual_number][2][3])

    # pick the best choice out of T=3
    if t2choice=='STEM' or t1choice=='STEM':
        t3choiceSet={'STEM':t3STEM,'nonSTEM':t3nonSTEM,'unskilled':t3work,
    'hp':t3hp}
    else:
        t3choiceSet={'nonSTEM':t3nonSTEM,'unskilled':t3work,'hp':t3hp}


    t3choice=max(t3choiceSet, key=t3choiceSet.get)
    # write down the observation
    if t3choice=='hp':
        output[3]={'choice':'hp'}
        return output
    elif t3choice=='STEM':
        t3grade=round_to_5(100*(Egrade(grade_params,3,1,SAT_M,SAT_V,hs_GPA,
            A_S,A_N)+ShockGradeRaw[individual_number][2]*STEMsd_by_time[1]))
        output[3]={'choice':'STEM','grade':t3grade,
        'shock':ShockGradeRaw[individual_number][2]*STEMsd_by_time[1]}
    elif t3choice=='nonSTEM':
        t3grade=round_to_5(100*(Egrade(grade_params,3,0,SAT_M,SAT_V,hs_GPA,
            A_S,A_N)+ShockGradeRaw[individual_number][2]*nonSTEMsd_by_time[1]))
        output[3]={'choice':'nonSTEM','grade':t3grade,
        'shock':ShockGradeRaw[individual_number][2]*nonSTEMsd_by_time[1]}
    else:
        output[3]={'choice':'unskilled','lwage':unskilled_params[0]+
        ShocksDict['unskilled'][individual_number][2],
        'shock':ShocksDict['unskilled'][individual_number][2]}
        return output

    # LOM to T=4

    if t3choice=='STEM':
        t4grades=future_grade(grade_params,4,(t1grade+t2grade+t3grade)/3,
            1,SAT_M,SAT_V,hs_GPA,A_S,A_N,normquantiles)
        payout=np.zeros(len(t4grades))
        for x in range(len(t4grades)):
            if t4grades[x]<200:
                payout[x]=dropout_payouts[3,0]
            else:
                payout[x]=(
                    STEM_payouts[round_to_10(tgpa_to_index(t4grades[x]))]+
                    grad_payoff)
        t4STEM=(beta*np.mean(payout)+
            ShocksDict['pref'][individual_number][3][0]+flowSTEM-gamma_p*tuition)
    else:
        t4grades=future_grade(grade_params,4,(t1grade+t2grade+t3grade)/3,
            0,SAT_M,SAT_V,hs_GPA,A_S,A_N,normquantiles)
        payout=np.zeros(len(t4grades))
        for x in range(len(t4grades)):
            if t4grades[x]<200:
                payout[x]=dropout_payouts[3,0]
            else:
                payout[x]=(nonSTEM_payouts[round_to_10(tgpa_to_index(
                    t4grades[x]))]+grad_payoff)
        t4nonSTEM=(beta*np.mean(payout)+
            ShocksDict['pref'][individual_number][3][1]+flownonSTEM-gamma_p*tuition)      

    # YEAR FOUR
    t4work=(beta*dropout_payouts[3,1]+gamma_p*np.exp(
        year_four_intercept+ShocksDict['unskilled'][individual_number][3])+
    ShocksDict['pref'][individual_number][3][2]+year_four_flow_penalized)

   
    t4hp=(beta*dropout_payouts[3,0]+
        ShocksDict['pref'][individual_number][3][3])

    # pick the best choice out of T=4
    if t3choice=='STEM':
        t4choiceSet={'STEM':t4STEM,'unskilled':t4work,'hp':t4hp}
    else:
        t4choiceSet={'nonSTEM':t4nonSTEM,'unskilled':t4work,'hp':t4hp}

    t4choice=max(t4choiceSet, key=t4choiceSet.get)
    # write down the observation and terminate the model
    if t4choice=='hp':
        output[4]={'choice':'hp'}

        return output

    # if STEM or nonSTEM are chosen, round the grade to the nearest available
    # GPA (see LaborGradeRange) and then solve the dynamic program
    elif t4choice=='STEM':
        t4grade=round_to_5(100*(Egrade(grade_params,4,1,SAT_M,SAT_V,hs_GPA,
            A_S,A_N)+ShockGradeRaw[individual_number][3]*STEMsd_by_time[1]))
        output[4]={'choice':'STEM','grade':t4grade,
        'shock':ShockGradeRaw[individual_number][3]*STEMsd_by_time[1]}
        finalGPA=round_to_5((t1grade+t2grade+t3grade+t4grade)/4)

        if finalGPA>=200:
            if finalGPA>=400:
                grade=400
            else:
                grade=round_to(finalGPA,20)
                new_output={'sim_num':individual_number,'quality':quality,'tGPA':grade,
                'dSTEM':1,'unobs_type':unobs_type}

                labor_dict.append(new_output)

        return output

    elif t4choice=='nonSTEM':
        t4grade=round_to_5(100*(Egrade(grade_params,4,0,SAT_M,SAT_V,hs_GPA,
            A_S,A_N)+ShockGradeRaw[individual_number][3]*nonSTEMsd_by_time[1]))
        output[4]={'choice':'nonSTEM','grade':t4grade,
        'shock':ShockGradeRaw[individual_number][3]*nonSTEMsd_by_time[1]}
        finalGPA=round_to_5((t1grade+t2grade+t3grade+t4grade)/4)

        if finalGPA>=200:
            if finalGPA>=400:
                grade=400
            else:
                grade=round_to(finalGPA,20)
                new_output={'sim_num':individual_number,'quality':quality,'tGPA':grade,
                'dSTEM':0,'unobs_type':unobs_type}

                labor_dict.append(new_output)

        return output
    else:
        output[4]={'choice':'unskilled','lwage':unskilled_params[0]+
        ShocksDict['unskilled'][individual_number][3],
        'shock':ShocksDict['unskilled'][individual_number][3]}

        return output
