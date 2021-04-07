import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
# Bandit environment
class Environment(object):
    def __init__(self, total, good):
        """
        A simple multi-arm bandit ennvironment for conventions. Creates an environment with 
        given number of arms and good arms. Returns reward = 1.0 if good arm pulled and -1
        otherwise.

        total: Total number of arms
        good: Number of good arms
        """
        self.total = total
        self.good = good
        self.context = None
        self.good_arms = None

    def reset(self):
        self.context = np.ones(self.total) * -1
        self.good_arms = random.sample(range(self.total - 1), self.good)
        self.context[self.good_arms] = 1
        return self.context

    def step(self, arm):
        return self.context[arm]


class Policy(nn.Module):
    def __init__(self, n_arms):
        super(Policy, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(n_arms, 10),
            nn.Tanh(),
            nn.Linear(10, 5),
            nn.Tanh(),
            nn.Linear(5, 4)
            ).to(device)

        self.dec = nn.Sequential(
            nn.Linear(1, 4),
            nn.Tanh(),
            nn.Linear(4, n_arms)
            ).to(device)

    def forward(self, x):
        x = self.enc(x)
        # x = self.dec(x)
        return F.softmax(x, dim=0)

def main():
    n_arms = 4
    n_good = 1
    policy = Policy(n_arms)
    savename = "models/bandit_1"

    optimizer = optim.SGD(policy.parameters(), lr=1e-2)

    env = Environment(n_arms, n_good)

    epochs = 100
    for e in range(epochs):
        loss_epoch = 0
        for i in range(100):
            state = env.reset()
            # print(env.good_arms)
            state = torch.tensor(state, device=device)
            probs = policy(state)
            m = Categorical(probs)
            action = m.sample()
            reward = env.step(action.item())
            loss = -m.log_prob(action) * reward
            loss_epoch += loss

        optimizer.zero_grad()
        # loss_epoch = torch.tensor(loss_epoch, device=device, requires_grad=True).sum()
        loss_epoch.backward()
        optimizer.step()
        print(e, loss_epoch.item())
    torch.save(policy.state_dict(), savename)
    
    policy.eval

    test_inp = torch.tensor([-1., -1., -1., 1.], device=device)
    probs = policy.forward(test_inp)
    print(probs)


if __name__ == "__main__":
    main()