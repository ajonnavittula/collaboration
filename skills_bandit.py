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
    def __init__(self, total=10):
        """
        A simple multi-arm bandit environment for conventions. Creates an environment with 
        given number of arms and good arms. Returns reward = 1.0 if good arm pulled and -1
        otherwise.

        total: Total number of arms
        good: Number of good arms
        """
        self.total = total
        self.context = None

    def reset(self):
        self.context = np.arange(1, self.total+1)
        np.random.shuffle(self.context)
        # self.good_arms = random.sample(range(self.total - 1), self.good)
        # self.context[self.good_arms] = 1
        return self.context

    def step(self, arm):
        return self.context[arm]


class Policy(nn.Module):
    def __init__(self, n_arms):
        super(Policy, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(n_arms+1, 10),
            nn.Tanh(),
            nn.Linear(10, 5),
            nn.Tanh(),
            nn.Linear(5, 1)
            ).to(device)

        self.dec = nn.Sequential(
            nn.Linear(1, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
            ).to(device)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return F.softmax(x, dim=0)

def main():
    n_arms = 10
    policy = Policy(n_arms)
    savename = "models/skills_bandit"

    optimizer = optim.SGD(policy.parameters(), lr=1e-2)

    env = Environment(n_arms)

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