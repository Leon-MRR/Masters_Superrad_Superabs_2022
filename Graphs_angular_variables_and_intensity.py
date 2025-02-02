import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp




#%%
def f(t, u, N, w0, g, gamma):
    alpha, theta, I = u

    tau = 2 / (np.sqrt(N)*g)
    phi = tau*w0*t

    return [-1j*(w0*tau*alpha + np.sin(theta)*np.exp(-1j*phi)),
        np.sqrt(N)*gamma*np.sin(theta) / g - 4*np.sqrt(np.imag(alpha)**2 + np.real(alpha)**2),
        tau*(alpha*np.exp(1j*phi) + np.conjugate(alpha)*np.exp(-1j*phi))/ 4]

N = 1e6 #número de átomos
gamma = 1e2
w0 = 1e10#omega 0
g = 1e5
t = np.linspace(0, 2, 100000)

u0 = [0,np.pi/2,0]

sol = solve_ivp(f, [t[0], t[-1]], u0, t_eval=t, args=(N, w0, g, gamma))

plt.subplots(1, 3, figsize=(10,7), sharex=True)
plt.subplots_adjust(wspace=0.3)

plt.subplot(1, 3, 1)
plt.plot(sol.t, sol.y[0], color='g')
plt.xlabel(r"$t/\tau$")
plt.grid()
plt.xlim(0,2)
plt.title(r"$\alpha(t)$") 

plt.subplot(1, 3, 2)
plt.plot(sol.t, sol.y[1], color = 'r')
plt.xlabel(r"$t/\tau$")
plt.grid()
plt.xlim(0,2)
plt.title(r"$\theta(t)$")

plt.subplot(1, 3, 3)
plt.plot(sol.t, sol.y[2], color = 'b')
plt.xlabel(r"$t/\tau$")
plt.grid()
plt.xlim(0,2)
plt.title(r"$I_+(t)$")


plt.show()
# %%