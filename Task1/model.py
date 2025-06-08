# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


unique_color_num = 6

class CNN_QNet(nn.Module):
    model_folder_path = './Task1/model'

    def __init__(self, input_shape=(5, 32, 32), num_actions=3):
        super().__init__()
        c, h, w = input_shape  # c=channels, h=height, w=width

        self.conv1 = nn.Conv2d(c, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Compute the size after conv/pool layers
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            dummy = self.pool2(F.relu(self.bn2(self.conv2(dummy))))
            dummy = F.relu(self.conv3(dummy))
            flatten_size = dummy.view(1, -1).size(1)

        self.fc1 = nn.Linear(flatten_size, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def save(self, file_name='model.pth'):
        os.makedirs(CNN_QNet.model_folder_path, exist_ok=True)
        file_name = os.path.join(CNN_QNet.model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    @staticmethod
    def load(env, file_name='model.pth'):
        file_name = os.path.join(CNN_QNet.model_folder_path, file_name)
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Model file {file_name} does not exist.")
        model = CNN_QNet(input_shape=(unique_color_num, env.height + 2 * env.border, env.width + 2 * env.border), num_actions=3)
        model.load_state_dict(torch.load(file_name))
        return model



class QTrainer:
    def __init__(self, model, lr, gamma,target_model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.target_model = target_model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert to torch tensors
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)


        if len(state.shape) == 3:  # single sample
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Predicted Q values
        pred = self.model(state)
        target = pred.clone().detach()

        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                with torch.no_grad():
                    Q_new += self.gamma * torch.max(self.model(next_state[i].unsqueeze(0))).item()
            target[i][action[i]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item() #return the loss value

