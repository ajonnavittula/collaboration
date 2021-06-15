import numpy as np
import matplotlib.pyplot as plt

# bayesian inference with boltzmann rational model
def bayes(s, a, A, G):
    beta = 1.0
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
g1 = np.array([+1, +0.3])
g2 = np.array([+1, -0.3])
g3 = np.array([+0, -1.0])
G = [g1, g2, g3]

# discretize the space of actions
n_actions = 31
r = 0.05
A = []
angles = np.linspace(0, 2*np.pi, n_actions)
for angle in angles:
    action = [r * np.cos(angle), r * np.sin(angle)]
    A.append(action)
A = np.asarray(A)

# go through the states and find the action that maximizes belief in the first goal
sx = np.linspace(-1, 1.0, 11)
sy = np.linspace(-1, 1.0, 11)
ASTAR = []
for x in sx:
    for y in sy:
        s = np.array([x, y])
        max_b = 0.0
        astar = None
        for a in A:
            b = bayes(s, a, A, G)
            if b[0] >= max_b:
                astar = np.copy(a)
                max_b = b[0]
        plt.arrow(x, y, astar[0], astar[1], head_width=0.1)
for g in G:
    plt.plot(g[0], g[1], 'ro')
plt.show()