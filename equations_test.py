import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import copy
import math

# Abar = A + B @ (F - K)
# Bbar = A + B @ F
# e(t) = exp(Abar * t) * e(0) - (exp(Abar * t) - eye(2)) * Abar^{-1} * Bbar * s_star


# s = [x, \dot{x}, x* - x, \dot{x}* - \dot{x}]
# Abar = A + B * (F*C1 + K*C2)
# s(t) = exp(Abar * t) * s(0)

# Q = C2' * q * C2
# R = C2'*K'* r *K*C2
# M = Q + R
# J = \int s(t)'*M*s(t) dt
# J = \int s(0)' * exp(Abar * t)' * M * exp(Abar * t) * s(0)


# run into problems differentiating J wrt F, mostly bc we have to differentiate
# wrt a matrix...


# time settings
timestep = 0.001
T = 2.0
n_steps = int(T / timestep) + 1
time = []
for idx in range(n_steps):
    time.append(idx * timestep)

# convention settings
F = np.array([[0.0, 0.0]])
K = np.array([[1.0, 0.1]])

# dynamic settings
mass = 1.0
damper = 1.0
A = np.array([[0, 1, 0, 0],[0, -damper/mass, 0, 0],[0, -1, 0, 0],[0, damper/mass, 0, 0]])
B = np.array([[0], [1/mass], [0], [-1/mass]])
C1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
C2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
Abar = A + B @ (F@C1 + K@C2)

# cost settings
q = np.array([[10.0, 0], [0, 1.0]])
r = 1.0
Q = np.transpose(C2) @ q @ C2
R = np.transpose(C2) @ np.transpose(K) * r * K @ C2
M = Q + R

# task settings
s_0 = np.array([[0], [0], [1.0], [0.0]])


def dynamics(s):
    sdot = Abar @ s
    return s + sdot * timestep

def rollout():
    xi, s = [], copy.deepcopy(s_0)
    for idx in range(n_steps):
        xi.append([s[0,0],s[1,0],s[2,0],s[3,0]])
        s = dynamics(s)
    return xi

def EoM(Abar, t):
    return la.expm(Abar * t) @ s_0

def cost(Abar):
    J = 0
    for t in time:
        s = EoM(Abar, t)
        J += float(np.transpose(s) @ M @ s * timestep)
    return J

def numerical_der():
    eta = 0.01
    F1p = F + np.array([[eta, 0.0]])
    F1n = F + np.array([[-eta, 0.0]])
    F2p = F + np.array([[0.0, eta]])
    F2n = F + np.array([[0.0, -eta]])
    Abar1p = A + B @ (F1p@C1 + K@C2)
    Abar1n = A + B @ (F1n@C1 + K@C2)
    Abar2p = A + B @ (F2p@C1 + K@C2)
    Abar2n = A + B @ (F2n@C1 + K@C2)
    c1p = cost(Abar1p)
    c1n = cost(Abar1n)
    c2p = cost(Abar2p)
    c2n = cost(Abar2n)
    return np.array([c1p - c1n, c2p - c2n]) * 0.5 / eta

print(numerical_der())

xi = rollout()
plt.plot(time, xi)

xip = []
for t in time:
    s = EoM(Abar, t)
    xip.append([s[0,0],s[1,0],s[2,0],s[3,0]])

plt.plot(time, xip, '--')
plt.show()
