import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
from collections import deque

import numpy as np

save_filename = "dqn_weights.pth"

# Actions possibles
ACTIONS = [
    (0, 0, 0),  # Ne rien faire
    (1, 0, 0),  # Accélérer
    (0, 1, 0),  # Freiner
    (0, 0, 1),  # Tourner droite
    (0, 0, -1),  # Tourner gauche
    (1, 0, 1),  # Accélérer + tourner droite
    (1, 0, -1),  # Accélérer + tourner gauche
    (0, 1, 1),  # Freiner + tourner droite
    (0, 1, -1),  # Freiner + tourner gauche
]

class DQN(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=len(ACTIONS)):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNAgent:
    def __init__(self, state_size=8, action_size=len(ACTIONS), lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.batch_size = 32
        self.target_update = 100
        self.step_count = 0

        # Réseaux de neurones
        self.q_network = DQN(state_size, 128, action_size)
        self.target_network = DQN(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Copier les poids initiaux
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        # Conversion plus efficace pour éviter l'avertissement
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, save_filename = "dqn_weights_default.pth"):
        torch.save(self.q_network.state_dict(), save_filename)
        print(f"Poids sauvegardés dans {save_filename}")

    def load(self, save_filename = "dqn_weights_default.pth"):
        self.q_network.load_state_dict(torch.load(save_filename))
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"Poids chargés depuis {save_filename}")


