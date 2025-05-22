# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class CNN_QNet(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_actions=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)  # output: 16x32x32
        self.pool = nn.MaxPool2d(2, 2)  # output: 16x16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # output: 32x16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch, 16, 16, 16)
        x = F.relu(self.conv2(x))             # -> (batch, 32, 16, 16)
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        os.makedirs(model_folder_path, exist_ok=True)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert to torch tensors
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)

        if len(state.shape) == 3:  # single sample
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Normalize state
        state /= 1.0
        next_state /= 1.0

        # Predicted Q values
        pred = self.model(state)
        target = pred.clone()

        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new += self.gamma * torch.max(self.model(next_state[i].unsqueeze(0)))
            target[i][action[i]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
