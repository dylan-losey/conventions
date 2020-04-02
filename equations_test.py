import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import copy
import math


# 1DoF system
# Abar = A + B * (F - K)
# Bbar = A + B * F
# e(t) = exp(Abar * t) * e(0) - Bbar / Abar * (exp(Abar * t) - 1) * s_star


# the matrix system falls apart because we have to take the derivative of
# a matrix exponential wrt that matrix

# the scalar system reaches a closed form derivative for dJ(t) / dF, but
# it is really ugly and needs an equation solver.


# time settings
timestep = 0.001
T = 5.0
n_steps, time = int(T / timestep) + 1, []
for idx in range(n_steps):
    time.append(idx * timestep)

# convention settings
F = 0.0
K = 1.0

# dynamic settings
A = 0
B = 1.0
Abar = A + B * (F - K)
Bbar = A + B * F

# cost settings
Q = 10.0
R = 1.0

# task settings
s_0 = 0.0
s_star = 1.0
e_0 = s_star - s_0


def dynamics(s):
    z = K * (s_star - s)
    sdot = (A + B * F) * s + B * z
    return s + sdot * timestep

def rollout():
    xi, s = [], s_0
    for idx in range(n_steps):
        xi.append(s)
        s = dynamics(s)
    return xi

def EoM(Abar, Bbar, t):
    return math.exp(Abar * t) * e_0 - Bbar / Abar * (math.exp(Abar * t) - 1) * s_star

def cost(e, t):
    return (Q + K**2 * R) * e**2

def numerical_der(t):
    eta = 0.01
    Fp = F + eta
    Fn = F - eta
    Abarp, Bbarp = A + B * (Fp - K), A + B * Fp
    Abarn, Bbarn = A + B * (Fn - K), A + B * Fn
    cp = cost(EoM(Abarp, Bbarp, t), t)
    cn = cost(EoM(Abarn, Bbarn, t), t)
    return (cp - cn) * 0.5 / eta

def closed_der(Abar, Bbar, t):
    c1 = 2.0 * (Q + K**2 * R)
    e = math.exp(Abar*t)*e_0 - Bbar/Abar*(math.exp(Abar*t) - 1)*s_star
    d1 = B*t*math.exp(Abar*t) * (e_0 - Bbar/Abar*s_star)
    d2 = B**2*K*s_star/Abar**2 * (math.exp(Abar*t) - 1)
    return c1 * e * (d1 + d2)



xi = rollout()
plt.plot(time, xi)

xip = []
for t in time:
    e = EoM(Abar, Bbar, t)
    xip.append(s_star - e)

plt.plot(time, xip, '--')
plt.show()

print(numerical_der(4.0))
print(closed_der(Abar, Bbar, 4.0))
