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
raw_array = np.reshape(raw_array, (L//5,5))

ele = raw_array[:,1]
data = raw_array[:,2]
coefs1 = raw_array[:,3]
coefs2 = raw_array[:,4]

params = np.empty((L,8))
for i in range(L):
    params[i,1] = float(data[i][1:6]) / 1000
    params[i,2] = float(data[i][7:12]) / 1000
    params[i,3] = float(data[i][13:18]) / 1000
    params[i,4] = float(data[i][19:24]) / 1000
    params[i,5] = float(data[i][25:30]) / 1000
    params[i,5] = float(data[i][31:36]) / 1000
    params[i,6] = float(data[i][37:42]) / 1000
    params[i,7] = float(data[i][43:48]) / 1000

atomic = pd.concat([df(ele),df(params),df(coefs1),df(coefs2)],axis=1)
atomic.columns = ['ele','mass','e_atom','e_ion','e_ion2','p_atom','p_ion','p_ion2','p_ion3','coef1','coef2']
