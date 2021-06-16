import numpy as np
import matplotlib.pyplot as plt
import sys

# bayesian inference with boltzmann rational model
def bayes(s, a, A, G, beta=20.0):
    P = []
    for g in G:
        num = np.exp(-beta * np.linalg.norm(g - (s + a)))
        den = 0
        for ap in A:
            den += np.exp(-beta * np.linalg.norm(g - (s + ap)))
        P.append(num / den)
    P = np.asarray(P)
    return P / sum(P)

# choose your goals
g1 = np.array([+0.7, +0.3])
g2 = np.array([+0.7, -0.3])
g3 = np.array([+0, -0.7])
G = [g1, g2]

# pick the true goal
gstar_idx = 0
gstar = G[gstar_idx]

# discretize the space of actions
n_actions = 31
r = 0.05
A = []
angles = np.linspace(0, 2*np.pi, n_actions)
for angle in angles:
    action = [r * np.cos(angle), r * np.sin(angle)]
    A.append(action)
A = np.asarray(A)

# go through the states and find the action that maximizes belief in the true goal
# subject to the constraint of being within epsilon of the optimal Q-value
sx = np.linspace(-1.0, 1.0, 11)
sy = np.linspace(-1.0, 1.0, 11)
epsilon = float(sys.argv[1])
fig, ax = plt.subplots()
for x in sx:
    for y in sy:
        s = np.array([x, y])
        Q = {}
        Qmax = -np.Inf
        for a in A:
            Q[str(a)] = np.linalg.norm(gstar - s) - np.linalg.norm(gstar - (s + a))
            if Q[str(a)] > Qmax:
                Qmax = Q[str(a)]
        value = -np.Inf
        astar = None
        for a in A:
            likelihood = bayes(s, a, A, G)
            if likelihood[gstar_idx] > value and Qmax - Q[str(a)] < epsilon:
                astar = np.copy(a)
                value = likelihood[gstar_idx]
        ax.arrow(x, y, astar[0], astar[1], head_width=0.1, color='b')

# plot a trajectory through this vector field
s = np.array([-0.8, 0.0])
S = []
while np.linalg.norm(gstar - s) > r:
    S.append(np.copy(s))
    Q = {}
    Qmax = -np.Inf
    for a in A:
        Q[str(a)] = np.linalg.norm(gstar - s) - np.linalg.norm(gstar - (s + a))
        if Q[str(a)] > Qmax:
            Qmax = Q[str(a)]
    value = -np.Inf
    astar = None
    for a in A:
        likelihood = bayes(s, a, A, G)
        if likelihood[gstar_idx] > value and Qmax - Q[str(a)] < epsilon:
            astar = np.copy(a)
            value = likelihood[gstar_idx]
    s += astar
S = np.asarray(S)
plt.plot(S[:,0], S[:,1], 'g-', linewidth=5)

for g in G:
    ax.plot(g[0], g[1], 'ro', markersize=10)
ax.axis([-1.1, 1.1, -1.1, 1.1])
ax.axis('equal')
plt.show()