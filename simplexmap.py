from numba import jit
import numpy as np
import scipy.special


# note that sectors is number of sectors in data, i.e. does
# not include HP
# @jit(nopython=True)
# def pascal(time,sectors):
#     out=np.zeros((time+sectors+1,time+sectors+1),dtype=np.int64)
#     for x in range(1,time+sectors+1):
#         for y in range(1,x+1):
#             product=int(1)
#             for i in range(y+1,x+1):
#                 product=product*i
#             for i in range(1,x-y+1):
#                 product=product/i
#             out[x,y]=int(product)
#     out[:,0]=1
#     return out

def pascal(time,sectors):
    out=np.zeros((time+sectors+1,time+sectors+1),dtype=np.int64)
    for x in range(1,time+sectors+1):
        for y in range(1,x+1):
            out[x,y]=scipy.special.comb(x, y, exact=True,
                repetition=False)
    out[:,0]=1
    return out

# convert tuple (including extra HP sector) to array
@jit(nopython=True)
def combo_to_array(time,endowment,pascal):
    counter=0
    sectors=len(endowment)-1
    cumulative=0
    for idx,val in enumerate(endowment[:-1]):
        for i in range(val):
            counter+=pascal[time+sectors-1-i-cumulative-idx][sectors-idx-1]
        cumulative+=val
    return int(counter)

# convert array back to tuple (including extra HP sector)
# sectors does NOT count HP
@jit(nopython=True)
def array_to_combo(array_value,time,sectors,pascal):
    out=np.zeros(sectors+1,dtype=np.int64)
    sector_index=0
    cumulative=0
    sector_value=0

    while array_value>0:

        if array_value>= (pascal[time+sectors-1-cumulative-
            sector_value-sector_index][sectors-1-sector_index]):
            
            array_value-=(pascal[time+sectors-1-cumulative-
            sector_value-sector_index][sectors-1-sector_index])
            sector_value+=1

        else:
            out[sector_index]=sector_value
            cumulative+=sector_value
            sector_index+=1
            sector_value=0
    out[sector_index]=sector_value
    out[-1]=time-sector_value-cumulative
    return out

def main():
    # case for 3 sectors + HP, 5 time units to allocate
    sectors=3
    time=5
    choose=pascal(time,sectors)
    print('(4,1,0,0) maps to '+str(combo_to_array(time,(4,1,0,0),choose)))
    print('54 maps to '+str(array_to_combo(54,5,3,choose)))
if __name__ == '__main__':
    main()