#!/usr/bin/env python3

##read files into pandas: atomic, lines, abundandes, model

import numpy as np
import pandas as pd
from pandas import DataFrame as df
import numba

atomic_txt = "atomic.dat"
abundances_txt = "abundances.dat"
lines_txt = "lines.dat"
model_txt = "kurucz_model.dat"

lines = pd.read_csv(lines_txt, delimiter='\t',header=1)
abundances = pd.read_csv(abundances_txt, delimiter='\t',header=1)
model =  pd.read_csv(model_txt, delimiter='\t',header=0)

raw_array = np.array([])
with open(atomic_txt) as f:
    for i,x in enumerate(f):
        x = x.replace("\n","")
        x = x.split("\t")
        raw_array = np.append(raw_array,x)

L = len(raw_array)
new_array = np.reshape(raw_array, (L//5,5))

ele = new_array[:,1]
data = new_array[:,2]
coefs1 = new_array[:,3]
coefs2 = new_array[:,4]

N = len(new_array)
params = np.empty((N,8))
for i in range(N):
    params[i,1] = float(data[i][1:6])
    params[i,2] = float(data[i][7:12])
    params[i,3] = float(data[i][13:18]) 
    params[i,4] = float(data[i][19:24])
    params[i,5] = float(data[i][25:30])
    params[i,5] = float(data[i][31:36])
    params[i,6] = float(data[i][37:42])
    params[i,7] = float(data[i][43:48]) 

atomic = pd.concat([df(ele),df(params)/1000,df(coefs1),df(coefs2)],axis=1)
atomic.columns = ['ele','mass','e_atom','e_ion','e_ion2','p_atom','p_ion','p_ion2','p_ion3','coef1','coef2']
