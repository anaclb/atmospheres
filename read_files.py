#!/usr/bin/env python3

##read files into pandas

import numpy as np
import pandas as pd
import numba


directory = "/home/bolacha/University/6th semester/Processos Radiativos/trabalho"
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

N = L//5
mass = np.empty(N)
e_atom, e_ion, e_ion2 = np.copy(mass), np.copy(mass), np.copy(mass)
p_atom, p_ion, p_ion2, p_ion3 = np.copy(mass),np.copy(mass),np.copy(mass),np.copy(mass)
for i in range(N):
    mass[i] = float(data[i][1:6]) / 1000
    e_atom[i] = float(data[i][7:12]) / 1000
    e_ion[i] = float(data[i][13:18]) / 1000
    e_ion2[i] = float(data[i][19:24]) / 1000
    p_atom[i] = float(data[i][25:30]) / 1000
    p_ion[i] = float(data[i][31:36]) / 1000
    p_ion2[i] = float(data[i][37:42]) / 1000
    p_ion3[i] = float(data[i][43:48]) / 1000
    
atomic = pd.DataFrame(np.transpose(np.reshape(np.concatenate([ele,mass,e_atom,e_ion,e_ion2,p_atom,p_ion,p_ion2,p_ion3,coefs1,coefs2]),(11,N))))
atomic.columns = ['ele','mass','e_atom','e_ion','e_ion2','p_atom','p_ion','p_ion2','p_ion3','coef1','coef2']

