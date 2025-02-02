#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd


#%%

def f(t, u, N, w0, gamma, g):
    Sz, Sx, Sy, x1, x2 = u
   
    Lr = N**0.5*g / (Sx*Sx + Sy*Sy) * (Sx*x1 - Sy*x2)
    Li = N**0.5*g / (Sx*Sx + Sy*Sy) * (Sx*x2 + Sy*x1) - N*gamma/2
   
    return [2*Li * (Sx*Sx + Sy*Sy),
        (-w0*Sy + 2*Lr*Sy*Sz - 2*Li*Sx*Sz),
        (w0*Sx - 2*Lr*Sx*Sz - 2*Li*Sy*Sz),
        (-N**0.5*g*Sy + w0*x2),
        (-N**0.5*g*Sx - w0*x1)]

N = 1e3
w0 = 1e2
gamma = 1e-3
g = 1   
tc = 2/(gamma*N)
t0 = tc*np.log(N)

u0 = [0, 0.5, 0, 0, 0]

t = np.linspace(0, 3, 100000)

sol = solve_ivp(f, [t[0], t[-1]], u0, t_eval=t, args=(N, w0, gamma, g))
sol1 = solve_ivp(f, [t[0], t[-1]], u0, t_eval=t, args=(N, w0, gamma, g==0))
sol2 = solve_ivp(f, [t[0], t[-1]], u0, t_eval=t, args=(N:=1, w0, gamma, g:=1))
Sz = sol.y[0]
Sx = sol.y[1]
Sy = sol.y[2]
x1 = sol.y[3]
x2 = sol.y[4]
v = [Sz, Sx, Sy, x1, x2]
A = f(t, v, N, w0, gamma, g)[0]
B = f(t, v, N, w0, gamma, g)[1]
C = f(t, v, N, w0, gamma, g)[2]
D = f(t, v, N, w0, gamma, g)[3]
E = f(t, v, N, w0, gamma, g)[4]



#expressão de acordo com o que calculei
Lr = N**0.5*g*(Sx*x1 - Sy*x2) / (Sx*Sx + Sy*Sy) 
Li = N**0.5*g* (Sx*x2 + Sy*x1) / (Sx*Sx + Sy*Sy) - N*gamma/2
Lrd=g*N**0.5 / (Sx*Sx + Sy*Sy)*((B*x1 + Sx*D - C*x2 - Sy*E) + (Sx*x1 - Sy*x2)*2*A*Sz / (Sx*Sx + Sy*Sy))
Ia = -N*(A + 2/w0*(Lrd*(Sx*Sx + Sy*Sy) - 2*Lr*A*Sz))
If = 2*N*(g*N**0.5*(Sy*x1 + x2*Sx) - (g/w0)*N**0.5*(B*x1 + Sx*D - C*x2 - Sy*E))
Tot = Ia + If
plt.subplots(1, 1, figsize=(8,4))


plt.plot(sol.t, Ia, label = r"Ia/w0")
plt.xlabel('gt')
plt.xlim(0, 0.5)
plt.legend(loc='upper right')


plt.plot(sol.t, If, label = r"If/w0")
plt.xlabel('gt')
plt.xlim(0, 0.5)
plt.legend(loc='upper right')






#plt.plot(sol.t,Tot)
#plt.plot('gt')
#plt.xlim(0,1)




F =0.5 + sol.y[0]

G =0.5 + sol1.y[0]

H = 0.5 + sol2.y[0]
plt.subplots(1, 1, figsize=(8,4))

plt.plot(sol.t, F, color = 'g', label = r"$g\neq 0$")
plt.xlabel(r"$T=t/\tau_c$")
plt.xlim(0, 3)
plt.legend(loc='best')

plt.plot(sol.t, G, color = 'b', label = "$g=0$")
plt.xlabel(r"$T=t/\tau_c$")
plt.ylabel(r"$\langle\sigma_+\sigma_-\rangle$")
plt.xlim(0, 3)
plt.legend(loc='best')

plt.plot(sol.t, H, color = 'r', label = "$N=1$")
plt.xlabel(r"$T=t/\tau_c$")
plt.ylabel(r"$\langle\sigma_+\sigma_-\rangle$")
plt.xlim(0, 3)
plt.legend(loc='best')

plt.show()
#%%

cols = ["t", "Sz", "Sx", "Sy", "x1", "x2", "Ia", "If"]

dat = np.concatenate((sol.t.reshape((1,-1)), sol.y, Ia.reshape((1,-1)), If.reshape((1,-1)))).T

df = pd.DataFrame(dat, columns=cols)

df.to_csv("output_ode_leandro.csv")
