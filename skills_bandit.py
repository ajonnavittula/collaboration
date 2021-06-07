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

# collect dataset
class MotionData(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        snippet = torch.FloatTensor(item[0]).to(device)
        action = torch.FloatTensor(item[1]).to(device)
        return (snippet, action)


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
        return np.append(self.context, self.idx)

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
            nn.Linear(n_arms+1, 10),
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
        return F.softmax(x, dim=0)


def main():
    n_arms = 10
    policy = Policy(n_arms)
    savename = "models/skills_bandit"

    env = Environment(total=n_arms)

    n_history = 1000
    history = []
    for i in range(n_history):
        state = env.reset()
        history.append([state[:-1].tolist(), state[-1].tolist()])
    # print(history)

    EPOCH = 2000
    BATCH_SIZE_TRAIN = 200
    LR = 0.01
    LR_STEP_SIZE = 1400
    LR_GAMMA = 0.1

    optimizer = optim.SGD(policy.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    train_data = MotionData(history)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    for epoch in range(EPOCH):
        
        x = next(iter(train_set))
        c = torch.cat((x[0], x[1]), 1)
        probs = policy(c)
        m = Categorical(probs)
        action = m.sample()

        if action.item() == x[1].item():
            print("matched")
        # optimizer.zero_grad()
        #     reward = env.step(action.item())
        #     loss = -m.log_prob(action) * reward
        #     loss.backward()
        #     optimizer.step()
        # scheduler.step()
        # print(epoch, loss.item())
        # torch.save(policy.state_dict(), savename)
    
    # epochs = 250
    # for e in range(epochs):
    #     loss_epoch = 0
    #     for i in range(200):
    #         state = env.reset()
    #         # print(env.good_arms)
    #         state = torch.tensor(state, device=device)
    #         probs = policy(state)
    #         m = Categorical(probs)
    #         action = m.sample()
    #         reward = env.step(action.item())
    #         loss = -m.log_prob(action) * reward
    #         loss_epoch += loss

    #     optimizer.zero_grad()
    #     # loss_epoch = torch.tensor(loss_epoch, device=device, requires_grad=True).sum()
    #     loss_epoch.backward()
    #     optimizer.step()
    #     print(e, loss_epoch.item())
    # torch.save(policy.state_dict(), savename)
    
    # policy.eval

    # correct = 0
    # for i in range(100):
    #     state = env.reset()
    #     # print(env.good_arms)
    #     state = torch.tensor(state, device=device)
    #     probs = policy(state)
    #     action = torch.argmax(probs)
    #     reward = env.step(action.item())
    #     if reward > 0:
    #         correct+=1

    # print(correct)



if __name__ == "__main__":
    main()