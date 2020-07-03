#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.integrate import quad, simps
from read_files import pd_atomic, pd_model, pd_abundances, pd_lines, split_coefs
import numba

#model:  1a col: Teff/logg, >=4 são anãs. cols em cgs
############### constantes ###############
h = 4.135667696*10**(-15) #[eV*s]
c = 3*10**8 #[m/s]
k = 8.62*10**-5 #[eV/K]
e = 1.692*(10)**(-19)
m_e = 9.10938356*10**(-34) #[g]

logn_H, logN_H, logn_e = 12, 24, 12
T_layer, T_cont = 5000, 5800 #[K]
a, e_mic = 0.01, 2000 #[ad],[m/s]


def B_lmb(lmb,T):
    return (2*h*c**2) / (lmb**5 * (np.exp((h*c)/(lmb*k*T))-1))

def intensity(tau,lmb,T_cont=T_cont,T_layer=T_layer):
    return B_lmb(lmb,T_cont) * np.exp(-tau) + B_lmb(lmb,T_layer) * (1-np.exp(-tau))

##finds indices for the elements which will be tested
def find_index(all_ele,test_ele):
    indices = np.array([])
    for ele in test_ele:
        indices = np.append(indices,np.where(all_ele==ele)).astype(int)
    return indices

###########calculating abundance
elements = ['Na','K','Fe','Si','Mg','Ca']
ab_indices = find_index(pd_abundances.el,elements)
logn_A = np.array(pd_abundances.logna[ab_indices])
atomic_test = pd_atomic.iloc[find_index(pd_atomic.ele,elements)]

def f_logN_A(logn_H, logN_H):
    logN_A = logn_A-logn_H+logN_H
    return logN_A

###########ionization fraction
def particc(g0s, coefs, T_layer):
    U = np.zeros(len(g0s))
    for i in range(5):
        U += np.exp (coefs[:,i]*(np.log(5040/T_layer))**i)
    return U + g0s

def saha(logn_e,T_layer,atomic_test=atomic_test):
    raw_coef1, raw_coef2 = np.array(atomic_test.coef1), np.array(atomic_test.coef2)
    coef1, coef2 = split_coefs(raw_coef1), split_coefs(raw_coef2)
    U_AI = particc(g0_atom, coef1, T_layer)
    U_AII = particc(g0_ion, coef2, T_layer)
    log_frac = np.log10(2*U_AII/U_AI) + 2.5*np.log10(T_layer) - e_ion1*(5040/T_layer) - logn_e - np.log10(k*T_layer) - 0.48
    return log_frac, U_AI, U_AII

def fracc(logn_e,T_layer,logN_H,logn_H):
    logN_A = f_logN_A(logn_H, logN_H)
    log_frac, U_AI, U_AII = saha(logn_e,T_layer)
    n_AII = 1  / (1 + 10**(1/log_frac))
    n_AI = 1 - n_AII
    N_AI, N_AII = 10**logN_A * n_AI, 10**logN_A * n_AII
    return N_AI, N_AII, U_AI, U_AII

##organizar dados
def pandas_data(logn_e, T_layer,logN_H,logn_H):
    N_AI, N_AII, U_AI, U_AII = fracc(logn_e,T_layer,logN_H,logn_H)
    concat = np.concatenate([np.array(atomic_test.ele),N_AII,U_AI,U_AII,e_ion1])
    concat = np.reshape(concat,(5,6))
    data = pd.DataFrame(concat).transpose()
    data.columns = ['ele','N_AII','U_AI','U_AII','e_ion1']
    return data


#########riscas em analise##########

#########numero de particulas absorsores
def particles(logn_e,T_layer,logN_H,logn_H):
    data = pandas_data(logn_e,T_layer,logN_H,logn_H)
    logN_Aif = np.zeros(len(pd_lines.ele))
    Us = np.copy(logN_Aif)
    logN_As = np.copy(logN_Aif)
    mass = np.copy(logN_Aif)
    names = []

    for i in range(len(pd_lines.ele)):
        N_AI, N_AII, U_AI, U_AII = fracc(logn_e,T_layer,logN_H,logn_H)
        logN_As[i] = np.log10(N_AII[np.where(data.ele==np.array(pd_lines.ele)[i])])
        names = np.append(names, pd_lines.ele[i])
        mass[i] = np.array(pd_atomic.mass)[np.where(data.ele==np.array(pd_lines.ele)[i])]

        if pd_lines.ion[i] == 1:
            Us[i] = U_AI[np.where(data.ele==np.array(pd_lines.ele)[i])]
        else:
            Us[i] = U_AII[np.where(data.ele==pd_lines.ele[i])]
    logN_AIf = np.array(pd_lines.loggf)+logN_As-np.log10(Us)-np.array(pd_lines.exc)*(5040/T_layer)
    return logN_AIf, mass

##opacidade, perfil de voigt
lambdas_teste = pd_lines["lambda"]

def voigt(a,e_mic,logn_e,T_layer,logN_H,logn_H):
    logN_AIf, mass = particles(logn_e,T_layer,logN_H,logn_H)
    L, N = 1000, len(lambdas_teste)
    integral = np.zeros((L,N))
    linspaces = np.copy(integral)
    dlmb = [2.1,1.5,.8,.2,.2,.15,.3,.2,.1] #customizing where each line starts/finishes

    for i, lmb_teste in enumerate(lambdas_teste):
        lambda_D = (lmb_teste / c) * np.sqrt(2*k*T_layer/mass[i] + e_mic**2)
        linspace_lmbs = np.linspace(lmb_teste-dlmb[i],lmb_teste+dlmb[i],L)
        linspaces[:,i] = linspace_lmbs
        vs = (linspace_lmbs - lmb_teste) / lambda_D
        for j, v in enumerate(vs):
            integrand = 2*quad(lambda y: np.exp(-y**2)/((v-y)**2+a**2),0,np.inf)[0]
            integral[j,i] = integrand
    H = a/np.pi * integral
    psi = H / (np.sqrt(np.pi) * lambda_D)
    return psi, linspaces, logN_AIf

def espessura(a,e_mic,logn_e,T_layer,logN_H,logn_H):
    psis, lambdas, logN_AIf = voigt(a,e_mic,logn_e,T_layer,logN_H,logn_H)
    tau = (((10 ** logN_AIf) * np.pi * e**2) / (m_e * c**2)) * lambdas**2 * psis
    return tau, lambdas

def eq_width(a,e_mic,T_cont,logn_e,T_layer,logN_H, logn_H):
    tau, lambdas = espessura(a, e_mic,logn_e,T_layer,logN_H,logn_H)
    Is = intensity(tau, lambdas)
    eq_w = np.zeros(len(lambdas_teste))
    for i in range(len(eq_w)):
        integrand = 1 - Is[:,i] /B_lmb(lambdas[:,i],T_cont)
        eq_w[i] = simps(integrand,lambdas[:,i],3000)
    return eq_w*1000, Is, lambdas #miliangstrom

e_ion1 = np.array(atomic_test.e_atom)
g0_atom, g0_ion = np.array(atomic_test.p_atom), np.array(atomic_test.p_ion)



