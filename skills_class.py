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
        # np.random.shuffle(self.context)
        self.skill = np.random.choice(self.context)
        idx = np.where(self.context==self.skill)
        self.idx = idx[0]
        # print(self.skill)
        # self.good_arms = random.sample(range(self.total - 1), self.good)
        # self.context[self.good_arms] = 1
        return np.append(self.context, self.skill).astype(float)

    def step(self, arm):
        # print("context: ",self.context)
        # print(arm)
        # print(self.skill)
        idx = np.where(self.context==self.skill)
        if  idx[0] == arm:
            return self.context[arm]
        else:
            return -1


class Policy(nn.Module):
    def __init__(self, n_arms):
        super(Policy, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(n_arms+1, 20),
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
            ).to(device)

        self.dec = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, n_arms)
            ).to(device)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x
        # return F.softmax(x, dim=0)

    def loss(self, output, target):
        return self.loss_func(output, target)


def main():
    n_arms = 10
    policy = Policy(n_arms)
    savename = "models/skills_class"

    optimizer = optim.SGD(policy.parameters(), lr=1e-5)

    env = Environment(total=n_arms)

    epochs = 250
    for e in range(epochs):
        loss_epoch = 0
        state = env.reset()
        # print(env.good_arms)
        state = torch.tensor(state, device=device)
        probs = policy(state)
        m = Categorical(probs)
        for i in range(2000):
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

    correct = 0
    for i in range(100):
        state = env.reset()
        # print(env.good_arms)
        state = torch.tensor(state, device=device)
        probs = policy(state)
        action = torch.argmax(probs)
        reward = env.step(action.item())
        if reward > 0:
            correct+=1

    print(correct)



if __name__ == "__main__":
    main()