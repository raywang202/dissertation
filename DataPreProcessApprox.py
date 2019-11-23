import sys
sys.path.append('/nas/longleaf/home/raywang/.local/lib/python3.5/site-packages')

import pandas as pd
import os

# Additional step for local import

import numpy as np

from bisect import bisect_left
from numba import jit

def cumulativeGPA(x):
    if x.augmented.iloc[0]==1:
        out=np.zeros(x.shape[0])
        return pd.DataFrame({'time':list(range(5,x.shape[0]+5)),
            'cumulativeGPA':out})
    out=np.zeros(x.shape[0])
    if x.choice.iloc[0] not in ['STEM','nonSTEM']:
        return pd.DataFrame({'cumulativeGPA':out})
    time=1
    while x.type.iloc[time-1]=='grade':
        out[time-1]=np.mean(x.outcome.iloc[0:time])
        time=time+1
        if time>x.shape[0]:
            break
    out[time-1:(x.shape[0]+1)]=out[time-2]
    return pd.DataFrame({'time':list(range(1,x.shape[0]+1)),
        'cumulativeGPA':out})



# bisect function lol
def takeClosest(myNumber, myList):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before

def AddEducationState(x):
        length=x.shape[0]
        out=[None]*x.shape[0]
        if x.augmented.iloc[0]==1:
            out=['skilled']*length
            return pd.DataFrame({'time':list(range(5,length+5)),'state':out})
        out[0]='t1'

        if length<2:
            return pd.DataFrame({'time':list(range(1,length+1)),'state':out})
        if x.choice.iloc[0]=='STEM':
            out[1]='t1STEM'
        elif x.choice.iloc[0]=='nonSTEM':
            out[1]='t1nonSTEM'
        else:
            out[1:length]=['unskilled']*(length-1)
            return pd.DataFrame({'time':list(range(1,length+1)),'state':out})
        if length<3:
            return pd.DataFrame({'time':list(range(1,length+1)),'state':out})
        if x.choice.iloc[1] in ['unskilled','hp']:
            out[2:length]=['unskilled']*(length-2)
            return pd.DataFrame({'time':list(range(1,length+1)),'state':out})
        elif x.choice.iloc[0]=='nonSTEM' and x.choice.iloc[1]=='nonSTEM':
            out[2]='STEMReqNotMet'
        else:
            out[2]='STEMReqMet'
        if length<4:
            return pd.DataFrame({'time':list(range(1,length+1)),'state':out})

        if x.choice.iloc[2] in ['unskilled','hp']:
            out[3:length]=['unskilled']*(length-3)
            return pd.DataFrame({'time':list(range(1,length+1)),'state':out})
        elif x.choice.iloc[2]=='STEM':
            out[3]='t3STEM'
        else:
            out[3]='t3nonSTEM'
        if length<5:
            return pd.DataFrame({'time':list(range(1,length+1)),'state':out})
        if x.choice.iloc[3] in ['unskilled','hp']:
            out[4:length]=['unskilled']*(length-4)
        elif x.terminalGPA.iloc[3]>200:
            out[4:length]=['skilled']*(length-4)
        else:
            out[4:length]=['unskilled']*(length-4)
        return pd.DataFrame({'time':list(range(1,length+1)),'state':out})

# terminal GPA is defined to be zero for dropouts
def terminalmajorgpa(x):
    if x.augmented.iloc[0]==1:
        return pd.DataFrame({'tmajor':[x.tmajor.iloc[0]],\
            'terminalGPA':[x.terminalGPA.iloc[0]]})
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

def numeric_state(y):
    length=y.shape[0]
    out=np.zeros(length,dtype=np.int64)
    for x in range(length):
        var=y.state.iloc[x]
        if var=='t1STEM':
            out[x]=0
        elif var=='t1nonSTEM':
            out[x]=1
        elif var=='STEMReqMet':
            out[x]=2
        elif var=='STEMReqNotMet':
            out[x]=3
        elif var=='t3STEM':
            out[x]=4
        elif var=='t3nonSTEM':
            out[x]=5
        elif var=='t1':
            out[x]=6
        elif var=='skilled':
            out[x]=7
        elif var=='unskilled':
            out[x]=8
    return out

def numeric_choice(y):
    length=y.shape[0]
    out=np.zeros(length,dtype=np.int64)
    for x in range(length):
        var=y.choice.iloc[x]
        if var=='STEM':
            out[x]=0
        elif var=='nonSTEM':
            out[x]=1
        elif var=='hp':
            out[x]=2
        elif var=='unskilled':
            out[x]=3
        else: out[x]=int(var[-1])+3
    return out

# coltype = 0 if HP, 1 if grade, 2 if logwage
def col_type(y):
    length=y.shape[0]
    out=np.zeros(length,dtype=np.int64)
    for x in range(length):
        var=y.type.iloc[x]
        if var=='hp':
            out[x]=0
        elif var=='grade':
            out[x]=1
        else:
            out[x]=2
    return out

def dSTEM(y):
    length=y.shape[0]
    out=np.zeros(length,dtype=np.int64)
    for x in range(length):
        var=y.tmajor.iloc[x]
        if var=='STEM':
            out[x]=1
    return out


def education_emax_mapping(A_N,A_S,SAT_M,SAT_V,hs_GPA,quality,tuition,univ_type,
    SATTuition):
    out=np.zeros(len(A_N),dtype=np.int64)
    for x in range(len(A_N)):
        key=tuple([A_N[x],A_S[x],SAT_M[x],SAT_V[x],hs_GPA[x],quality[x],
            tuition[x],univ_type[x]])
        out[x]=SATTuition.index(key)
    return out

# returns index of skilled emax array
# HARD CODED
def skilled_emax_mapping2(dSTEM,quality,state,tGPA,college_values):
    out=np.zeros(len(dSTEM),dtype=np.int64)

    for x in range(len(dSTEM)):
        if state[x]==7:
            key=tuple([quality[x],dSTEM[x],tGPA[x]])
            out[x]=college_values.index(key)   
        else:
            out[x]=-1
    return out

@jit(nopython=True)
def unskilled_emax_mapping(tdropout):
    out=np.zeros(len(tdropout),dtype=np.int64)
    for i in range(len(tdropout)):
        if tdropout[i]!=-1:
            out[i]=tdropout[i]-1
    return out

# if univ_type = public: 0, priv_nonrel : 1, priv_rel : 2, forprofit : 3
def univ_type_numeric(y):
    length=y.shape[0]
    out=np.zeros(length,dtype=np.int64)
    for x in range(length):
        var=y[x]
        if var=='public':
            out[x]=0
        elif var=='priv_nonrel':
            out[x]=1
        elif var=='priv_rel':
            out[x]=2
        else:
            out[x]=3
    return out

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


def AddExpCols(x):
    return x.shift().fillna(0).cumsum().astype(int)

def intconvert(x):
    if np.isnan(x):
        return -1
    return x

def PreProcess(fileLocation,sectors,LaborGradeRange):

    # Appends experience columns for each sector to the dataset

    # Adds education state space to data as well

    # df is one individual
    # sectors = number of skilled sectors
    # read in 'fileLocation' from MaximumLikelihoodFulSol.py

    #DFData=pd.read_csv(os.path.abspath(os.curdir)+'/sample_observations.csv')

    DFData=pd.read_csv(os.path.abspath(os.curdir)+fileLocation)
    DFData=DFData[['id','time','augmented','male','SAT_M','SAT_V','choice','outcome','type','A_N',
    'A_S','tuition','quality','tmajor','terminalGPA','hs_GPA','year','univ_type']]
    DFData=DFData.set_index(['id','time'])

    fullCategories=['STEM','nonSTEM','unskilled','hp']+\
    ['skilled'+str(x) for x in range(1,sectors+1)]
    expCategories=['unskilled','hp']+['skilled'+str(x) for x in range(1,sectors+1)]
    DFData['choicecat']=DFData.choice.astype(pd.api.types.CategoricalDtype(
        categories=fullCategories))
    


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

    # cumulative GPA function (calculated from outcome column, so minor rounding
    # errors)
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

    DFDataFinal['dSTEM']=dSTEM(DFDataFinal)
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



    DFDataFinal['lastchoice']=np.vectorize(intconvert)(
        DFDataFinal.lastchoiceraw)


    DFDataFinal['prior_skilled']=np.vectorize(skilled_convert)(
        DFDataFinal.lastchoice)
    return [DFDataFinal,SATTuition]