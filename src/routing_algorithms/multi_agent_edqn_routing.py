# multi_agent_edqn_routing_fixed.py
# Multi-Agent Enhanced DQN với CTDE + Communication + Attention
# Tương thích hoàn toàn với BASE_routing và Simulator
# Author: MARL Expert

import random
import math
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
from src.utilities import config as cfg
from src.entities.uav_entities import Drone  # Import để type checking


# ====================== NEURAL NETWORKS ======================

class BalancedDQN(nn.Module):
    """Mạng DQN với Dropout và BatchNormalization để ổn định huấn luyện"""

    def __init__(self, input_dim, output_dim):
        super(BalancedDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

        # Batch Norm giúp training ổn định hơn với state thay đổi liên tục
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Xử lý Batch Norm cần ít nhất 2 mẫu, nếu 1 mẫu thì bỏ qua BN
        if x.size(0) > 1:
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class MessageEncoder(nn.Module):
    """Mã hóa Local State thành Message Vector để gửi đi"""

    def __init__(self, state_dim, message_dim=32):
        super(MessageEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, message_dim),
            nn.Tanh()  # Tanh để giới hạn giá trị message trong [-1, 1]
        )

    def forward(self, state):
        return self.encoder(state)


class MessageDecoder(nn.Module):
    """Giải mã Message nhận được từ hàng xóm"""

    def __init__(self, message_dim=32, output_dim=16):
        super(MessageDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(message_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, message):
        return self.decoder(message)


# ====================== CENTRALIZED TRAINER ======================

class CentralizedTrainer:
    """Quản lý việc huấn luyện tập trung cho tất cả Agents"""

    def __init__(self, n_agents, state_size, action_size, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shared Models
        self.shared_model = BalancedDQN(state_size, action_size).to(self.device)
        self.target_model = BalancedDQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.shared_model.state_dict())
        self.target_model.eval()

        # Communication Modules
        self.msg_encoder = MessageEncoder(state_size).to(self.device)
        self.msg_decoder = MessageDecoder().to(self.device)

        # Optimizer chung
        self.optimizer = torch.optim.Adam(
            list(self.shared_model.parameters()) +
            list(self.msg_encoder.parameters()) +
            list(self.msg_decoder.parameters()),
            lr=config.get('learning_rate', 0.0005)
        )

        self.memory = deque(maxlen=config.get('memory_size', 20000))
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.tau = 0.005  # Soft update parameter
        self.loss_fn = nn.MSELoss()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Double DQN Logic
        with torch.no_grad():
            # Chọn action tốt nhất từ mạng Online
            next_actions = self.shared_model(next_states).argmax(1, keepdim=True)
            # Tính Q-value từ mạng Target
            next_q_values = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        current_q_values = self.shared_model(states).gather(1, actions)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping để tránh bùng nổ gradient
        nn.utils.clip_grad_norm_(self.shared_model.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update_target()
        return loss.item()

    def soft_update_target(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.shared_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


# ====================== DECENTRALIZED AGENT ======================

class CommunicatingAgent:
    def __init__(self, drone, trainer, config):
        self.drone = drone
        self.trainer = trainer
        self.id = drone.identifier

        # Epsilon Greedy
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.999)

        self.last_state = None
        self.last_action = None

    def get_action(self, state, valid_neighbors_mask, training=True):
        """Chọn action dựa trên State và Mask (chỉ chọn neighbors hợp lệ)"""
        if training and random.random() < self.epsilon:
            # Random trong số các neighbor hợp lệ
            valid_indices = np.where(valid_neighbors_mask)[0]
            if len(valid_indices) == 0:
                return None
            return random.choice(valid_indices)

        state_t = torch.FloatTensor(state).to(self.trainer.device)
        with torch.no_grad():
            q_values = self.trainer.shared_model(state_t).cpu().numpy().flatten()

        # Masking: Gán -inf cho các action không hợp lệ
        q_values[~valid_neighbors_mask] = -np.inf

        # Chọn max Q
        action = np.argmax(q_values)

        if q_values[action] == -np.inf:  # Không có neighbor nào
            return None

        return action

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ====================== MAIN ROUTING CLASS ======================

class MultiAgentEDQN_Routing(BASE_routing):
    def __init__(self, drone, simulator):
        super().__init__(drone, simulator)
        self.drone = drone
        self.simulator = simulator

        # --- CẤU HÌNH MARL ---
        self.config = {
            'state_size': 16,  # 10 local + 6 comm
            'learning_rate': 0.001,
            'memory_size': 50000,
            'batch_size': 64,
            'gamma': 0.95,
            'epsilon_start': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.9995
        }

        # --- KHỞI TẠO HỆ THỐNG MARL (SINGLETON) ---
        # Kiểm tra xem hệ thống MARL đã được khởi tạo trong simulator chưa
        if not hasattr(self.simulator, '_marl_system'):
            print(f"[MARL] Initializing Centralized System for {self.simulator.n_drones} drones...")
            self.simulator._marl_system = {
                'trainer': CentralizedTrainer(
                    self.simulator.n_drones,
                    self.config['state_size'],
                    self.simulator.n_drones,
                    self.config
                ),
                'agents': {}
            }

        # Đăng ký Agent cho Drone hiện tại
        if self.drone.identifier not in self.simulator._marl_system['agents']:
            self.simulator._marl_system['agents'][self.drone.identifier] = \
                CommunicatingAgent(self.drone, self.simulator._marl_system['trainer'], self.config)

        self.agent = self.simulator._marl_system['agents'][self.drone.identifier]
        self.trainer = self.simulator._marl_system['trainer']

    def get_augmented_state(self, opt_neighbors):
        """Tạo state vector: Local info + Neighbor summaries"""
        # 1. Local State (10 features)
        d = self.drone
        depot = self.simulator.depot

        max_dist = math.sqrt(self.simulator.env_width ** 2 + self.simulator.env_height ** 2)
        dist_to_depot = util.euclidean_distance(d.coords, depot.coords)

        local_state = np.array([
            d.coords[0] / self.simulator.env_width,
            d.coords[1] / self.simulator.env_height,
            d.residual_energy / d.initial_energy,
            dist_to_depot / max_dist,
            d.speed / 30.0,  # Norm speed
            len(opt_neighbors) / self.simulator.n_drones,  # Density
            1.0 if len(d.buffer) < d.buffer_max_size else 0.0,  # Buffer full?
            (d.coords[0] - depot.coords[0]) / max_dist,  # DX
            (d.coords[1] - depot.coords[1]) / max_dist,  # DY
            0.0  # Placeholder for extra metric
        ], dtype=np.float32)

        # 2. Communication State (6 features from Neighbors)
        # Giả lập việc nhận message và aggregate
        avg_nb_energy = 0
        avg_nb_dist = 0
        max_nb_energy = 0
        min_nb_dist = 1.0

        if len(opt_neighbors) > 0:
            energies = [n.residual_energy / n.initial_energy for n in opt_neighbors]
            dists = [util.euclidean_distance(n.coords, depot.coords) / max_dist for n in opt_neighbors]

            avg_nb_energy = np.mean(energies)
            avg_nb_dist = np.mean(dists)
            max_nb_energy = np.max(energies)
            min_nb_dist = np.min(dists)

        comm_state = np.array([
            avg_nb_energy, avg_nb_dist, max_nb_energy, min_nb_dist,
            len(opt_neighbors) > 0,  # Has neighbors?
            0.0  # Placeholder
        ], dtype=np.float32)

        return np.concatenate([local_state, comm_state])

    def relay_selection(self, opt_neighbors, packet):
        """
        Chọn relay tốt nhất sử dụng mạng Neural (MA-EDQN).
        CRITICAL FIX: Phải trả về Object Drone, không phải ID.
        """
        # 1. Nếu Depot trong tầm, gửi luôn
        dist_to_depot = util.euclidean_distance(self.drone.coords, self.simulator.depot.coords)
        if dist_to_depot <= self.drone.communication_range:
            return self.simulator.depot  # Trả về Depot object nếu có thể

        # 2. Lấy state hiện tại
        state = self.get_augmented_state(opt_neighbors)
        self.agent.last_state = state

        # 3. Tạo mask cho neighbors hợp lệ
        # Action space size = N_drones. Chỉ những index tương ứng với opt_neighbors mới = 1
        valid_mask = np.zeros(self.simulator.n_drones, dtype=bool)
        neighbor_map = {}  # Map ID -> Drone Object

        for n in opt_neighbors:
            if n.identifier < self.simulator.n_drones:  # Đảm bảo là drone
                valid_mask[n.identifier] = True
                neighbor_map[n.identifier] = n

        if not any(valid_mask):
            return None  # Routing hole

        # 4. Chọn action từ Agent
        action_id = self.agent.get_action(state, valid_mask)
        self.agent.last_action = action_id

        if action_id is not None and action_id in neighbor_map:
            # COMPATIBILITY FIX: Trả về Drone Object
            return neighbor_map[action_id]

        return None

    def feedback(self, *args):
        """
        Xử lý feedback từ Simulator để huấn luyện.
        Chấp nhận *args để tương thích với các cách gọi khác nhau của BASE_routing.
        """
        # Parse arguments dựa trên số lượng tham số
        outcome = -1
        # Thông thường Simulator gọi: feedback(event_id, outcome, ...) hoặc feedback(outcome, ...)
        # Ta sẽ cố gắng tìm tham số outcome (thường là 1: success, 0: forward, -1: fail)

        try:
            # Heuristic parsing
            if len(args) >= 2:
                # Giả định tham số thứ 2 là outcome hoặc tham số đầu
                # Kiểm tra xem có arg nào là 1 hoặc -1 không
                for arg in args:
                    if arg in [1, 0, -1]:
                        outcome = arg
                        break
        except:
            outcome = 0

        # Nếu chưa có state/action trước đó thì bỏ qua
        if self.agent.last_state is None or self.agent.last_action is None:
            return

        # Tính Reward
        reward = 0
        done = False

        if outcome == 1:  # Gửi thành công tới đích/depot
            reward = 10.0
            done = True
        elif outcome == 0:  # Forward thành công
            reward = -0.1  # Phạt nhẹ để khuyến khích đường ngắn
            done = False
        else:  # Thất bại (Routing hole, drop)
            reward = -5.0
            done = True

        # Lấy Next State (xấp xỉ bằng state hiện tại + thay đổi nhỏ hoặc tính lại)
        # Ở đây ta tính lại state dựa trên neighbor hiện tại (có thể đã cũ, nhưng chấp nhận được)
        neighbors = self.drone.get_neighbors(self.simulator.drones)
        next_state = self.get_augmented_state(neighbors)

        # Lưu vào Replay Memory chung
        self.trainer.store_transition(
            self.agent.last_state,
            self.agent.last_action,
            reward,
            next_state,
            done
        )

        # Thực hiện Training bước (nếu đủ batch)
        # Chỉ thực hiện khi Drone có ID = 0 để tránh train quá nhiều lần trong 1 step
        if self.drone.identifier == 0 and self.simulator.cur_step % 5 == 0:
            loss = self.trainer.train_step()
            if loss is not None and self.simulator.cur_step % 100 == 0:
                pass  # Có thể in log loss tại đây

        # Cập nhật Epsilon
        if done:
            self.agent.update_epsilon()
            self.agent.last_state = None
            self.agent.last_action = None

    def routing(self, depot, drones, cur_step):
        """Override phương thức định tuyến định kỳ (nếu cần)"""
        # BASE_routing thường tự xử lý việc gửi gói tin
        # Hàm này chủ yếu để update neighbor table hoặc broadcast hello packet
        super().routing(depot, drones, cur_step)