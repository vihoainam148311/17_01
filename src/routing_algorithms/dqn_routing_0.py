import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def get_realtime_uav_state(simulator, uav_id):
    drone = simulator.get_drone_by_id(uav_id)
    if drone is None:
        return None

    return np.array([
        drone.coords[0],  # X tọa độ
        drone.coords[1],  # Y tọa độ
        drone.residual_energy / drone.initial_energy,  # % năng lượng còn lại
        np.mean(drone.neighbor_table[:, 12]),  # Chất lượng liên kết trung bình
        np.mean(drone.neighbor_table[:, 8]),  # Độ trễ trung bình
        drone.speed  # Tốc độ UAV
    ])

class DQN_Routing(BASE_routing):
    def __init__(self, drone, simulator):
        super().__init__(drone, simulator)
        self.state_size = 6  # x, y, energy, LQ, delay, speed
        self.action_size = len(self.simulator.drones)  # Số UAV có thể chọn
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss()

    def get_state(self):
        return get_realtime_uav_state(self.simulator, self.drone.identifier)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            if next_state is None:
                continue

            target = reward
            if not done:
                best_action = torch.argmax(self.model(torch.FloatTensor(next_state))).item()
                target = reward + self.gamma * self.target_model(torch.FloatTensor(next_state))[best_action]

            predicted = self.model(torch.FloatTensor(state))
            if action < len(predicted):
                predicted[action] = target
            loss = self.loss_function(predicted, self.model(torch.FloatTensor(state)))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def select_action(self, state, opt_neighbors):
        if not opt_neighbors:
            print("Không có UAV relay hợp lệ! Không có hàng xóm khả dụng.")
            return None

        if np.random.rand() <= self.epsilon:
            return random.choice(range(len(opt_neighbors)))

        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_values = self.model(state_tensor)
            sorted_indices = torch.argsort(action_values, descending=True)

            for idx in sorted_indices:
                if idx.item() < len(opt_neighbors):
                    return idx.item()
        print("Lỗi: Không có hành động hợp lệ!")
        return None

    def relay_selection(self, opt_neighbors, data):
        state = self.get_state()
        if state is None or not opt_neighbors:
            print("Không có UAV relay hợp lệ!")
            return None

        action = self.select_action(state, opt_neighbors)
        if action is None or action >= len(opt_neighbors):
            print("Lỗi: Action ngoài phạm vi! Chọn UAV ngẫu nhiên từ danh sách.")
            return random.choice(opt_neighbors) if opt_neighbors else None

        chosen_uav = opt_neighbors[action]
        #print(f"UAV relay được chọn: {chosen_uav.identifier}")
        return chosen_uav

    def feedback(self, outcome, id_j, Q_value_best_action):
        """
        Cập nhật giá trị Q dựa trên phản hồi từ môi trường.
        """
        alpha = self.drone.neighbor_table[id_j, 10]
        gamma = self.drone.neighbor_table[id_j, 7]
        Q_value_i_j = self.drone.neighbor_table[id_j, 9]

        if outcome == 1:  # Gói tin đến đích
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (5)  # maxReward
        elif outcome == 0:  # Gói tin chưa đến đích
            delay = self.drone.neighbor_table[id_j, 8] + self.drone.neighbor_table[id_j, 11]
            reward = self.compute_reward(outcome, delay)
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (reward + gamma * Q_value_best_action - Q_value_i_j)
        else:  # Gói tin mất
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (-5 + gamma * Q_value_best_action - Q_value_i_j)

    def compute_reward(self, outcome, delay):
        """Tính toán phần thưởng dựa trên độ trễ và năng lượng UAV."""
        return 0.7 * np.exp(-delay) + 0.3 * (self.drone.residual_energy / self.drone.initial_energy)
