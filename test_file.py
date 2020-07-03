#!/usr/bin/env python3
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import general_functions as gf
import matplotlib.pyplot as plt
import numba
import seaborn as sns
sns.set_context("paper")
#lines: angstrom, eV
#model:  1a col: Teff/logg, >=4 são anãs. cols em cgs

logn_H, logN_H, logn_e = 12, 24, 12
T_layer, T_cont = 5000, 5800 #[K]
a, e_mic = 0.01, 2000 #[ad],[m/s]

#####eqw calc using a=0.01 e=2km/s

#eqw_test,Is_test,lambdas_test = gf.eq_width(a,e_mic,T_cont,logn_e,T_layer,logN_H, logn_H)
#pd_eqwidth = pd.concat([gf.pd_lines.ele,gf.pd_lines['lambda'],gf.pd_lines.ion, pd.DataFrame(eqw_test)],axis=1)
#pd_eqwidth.columns = ["ele","lmb","ion","eq_width(mA)"]
#print("a = {}".format(a), "e_mic = {} m/s".format(e_mic))
#print(pd_eqwidth)

#plotting all lines with those parameters, individually
#for i in range(9):
#    plt.plot(lambdas_test[:,i],Is_test[:,i]/np.max(Is_test[:,i]))
#    plt.title("{} {}, {} A".format(gf.pd_lines.ele[i],gf.pd_lines.ion[i],gf.pd_lines['lambda'][i]))
#    plt.legend()
#    plt.ylabel("Intensity (normalized)")
#    plt.xlabel(r"$\lambda [\AA]$")
#    plt.show()

#plotting all, together
#fig, ax = plt.subplots(3,3, figsize=(13, 20))
#for i in range(3):
#    plot1 = ax[i,0].plot(lambdas_test[:,i], Is_test[:,i]/np.max(Is_test[:,i]))
#    ax[i,0].set_title("{} {}".format(gf.pd_lines.ele[i],gf.pd_lines.ion[i]),fontsize=25)
#    plt.setp(ax, xticks=[], yticks=[])
#    plot2 = ax[i,1].plot(lambdas_test[:,3+i], Is_test[:,3+i]/np.max(Is_test[:,3+i]))
#    ax[i,1].set_title("{} {}".format(gf.pd_lines.ele[3+i],gf.pd_lines.ion[3+i]),fontsize=25)
#    plt.setp(ax, xticks=[], yticks=[])
#    plot3 = ax[i,2].plot(lambdas_test[:,6+i], Is_test[:,6+i]/np.max(Is_test[:,6+i]))
#    ax[i,2].set_title("{} {}".format(gf.pd_lines.ele[6+i],gf.pd_lines.ion[6+i]),fontsize=25)
#    plt.setp(ax, xticks=[], yticks=[])
#plt.savefig("line-all.pdf",dpi=1000,bbox_inches='tight')
#plt.show()

####varying a tests -- Si II

#for a in [0.003,0.01,0.3,0.8]:
#    eq_width_test, Is_test, lambdas_test = gf.eq_width(a,e_mic,T_cont,logn_e,T_layer,logN_H, logn_H)
#    plt.plot(lambdas_test[:,8],Is_test[:,8]/np.max(Is_test[:,8]), label = "a = {}".format(a))
##     plt.title("Na I")
#    plt.ylabel("Intensity (normalized)")
#    plt.xlabel(r"$\lambda [\AA]$")
#    plt.legend(loc='best',fontsize='medium')
#    plt.xlim(6346.95,6347.1)
#    eqw_test,Is_test,lambdas_test = gf.eq_width(a,e_mic,T_cont,logn_e,T_layer,logN_H, logn_H)
#    pd_eqwidth = pd.concat([gf.pd_lines.ele,gf.pd_lines['lambda'],gf.pd_lines.ion, pd.DataFrame(eq_width_test)],axis=1)
#    pd_eqwidth.columns = ["ele","lmb","ion","eq_width(mA)"]
#    print("a = {}".format(a), "e_mic = {} m/s".format(e_mic))
#    print(pd_eqwidth)
#plt.savefig("SiII-a.pdf", dpi=800, bbox_inches="tight")
#plt.show()

#########varying e_mic tests --- Si II

#for e_mic in [200,2000,20000]:
#    eq_width_test, Is_test, lambdas_test = gf.eq_width(a,e_mic,T_cont,logn_e,T_layer,logN_H, logn_H)
#    plt.plot(lambdas_test[:,8],Is_test[:,8]/np.max(Is_test[:,8]), label = r"$\epsilon = {} km/s$".format(e_mic/1000))
  #  plt.title("Na I")
#    plt.ylabel("Intensity (normalized)")
#    plt.xlabel(r"$\lambda [\AA]$")
#    plt.legend(loc='best',fontsize='medium')
#    plt.xlim(6346.95,6347.1)
#    eqw_test,Is_test,lambdas_test = gf.eq_width(a,e_mic,T_cont,logn_e,T_layer,logN_H, logn_H)
#    pd_eqwidth = pd.concat([gf.pd_lines.ele,gf.pd_lines['lambda'],gf.pd_lines.ion, pd.DataFrame(eq_width_test)],axis=1)
#    pd_eqwidth.columns = ["ele","lmb","ion","eq_width(mA)"]
#    print("a = {}".format(a), "e_mic = {} m/s".format(e_mic))
#    print(pd_eqwidth)
#plt.savefig("SiII-e_mic.pdf", dpi=800, bbox_inches="tight")
#plt.show()
#print(pd_model[3])
#print(pd_model.iloc[3])
#print(pd_model['T'])
#T_conts = np.array([5740, 4000, 4000, 5750,8000])plt.xlim()


####varying stellar parameters, fixing a=0.01, e_mic=2 km/s

T_layers = np.array(gf.pd_model['T'])
star = np.array(gf.pd_model.star)
T_conts=np.zeros(8)
star[0]='5800'
for i in range(8):
    T_conts[i] = star[i][0:4]

logn_es = np.log10(np.array(gf.pd_model.Xne))
logN_Hs = np.log10(np.array(gf.pd_model.NH))
logn_Hs = np.log10(np.array(gf.pd_model.nH))

for i in range(0,8):
    eq_width_test, Is_test, lambdas_test = gf.eq_width(a,e_mic,T_conts[i],logn_es[i],T_layers[i],logN_Hs[i], logn_Hs[i])
    plt.plot(lambdas_test[:,2],Is_test[:,2]/np.max(Is_test[:,2]), label = r"star = {}".format(gf.pd_model.star[i]))
    plt.xlabel(r"$\lambda [\AA]$")
    plt.ylabel("Intensity (normalized)")
    plt.legend(loc='lower right',fontsize='medium')
    plt.xlim(6161.9,6162.2)
plt.savefig("CaI-stars.pdf", dpi=800, bbox_inches="tight")
plt.show()
