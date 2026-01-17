import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

# single Agent -> MARL
# Import các module từ hệ thống mô phỏng của bạn
from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
from src.utilities import config

# Cấu hình thiết bị (GPU nếu có, không thì CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# PHẦN 1: CÁC MẠNG NEURAL (Neural Networks)
# ==========================================

class AgentNetwork(nn.Module):
    """
    Mạng cục bộ (Decentralized) chạy trên mỗi Drone.
    Input: Local Observation (Vị trí, năng lượng, khoảng cách đích...)
    Output: Q-values cho từng Drone láng giềng (Action candidates)
    """

    def __init__(self, input_dim, n_actions, hidden_dim=64):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class QMixer(nn.Module):
    """
    Mạng Mixing (Global) đảm bảo tính đơn điệu (Monotonicity).
    Input: Q-values của các Agent + Global State
    Output: Q_total (Joint Value)
    """

    def __init__(self, n_agents, state_dim, embed_dim=32):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetworks để sinh trọng số cho lớp 1 (Weights phải dương)
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim * self.n_agents),
            nn.ReLU()  # Đảm bảo phi tuyến tính
        )
        # Hypernetworks cho bias lớp 1
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # Hypernetworks để sinh trọng số cho lớp 2 (Weights phải dương)
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU()
        )
        # Hypernetworks cho bias lớp 2
        self.hyper_b_final = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        # agent_qs: (batch_size, 1, n_agents) - Trong routing hop-by-hop, ta coi mỗi hop là 1 step của team
        # states: (batch_size, state_dim)

        batch_size = agent_qs.size(0)

        # --- Layer 1 ---
        # Sinh weights từ state và reshape
        w1 = torch.abs(self.hyper_w_1(states))  # Abs để đảm bảo Monotonicity
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)

        b1 = self.hyper_b_1(states)
        b1 = b1.view(batch_size, 1, self.embed_dim)

        # Tính toán hidden: Agent_Q * W1 + B1
        # agent_qs shape: (Batch, 1, Agents)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # --- Layer 2 (Final) ---
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(batch_size, self.embed_dim, 1)

        b_final = self.hyper_b_final(states)
        b_final = b_final.view(batch_size, 1, 1)

        # Tính Q_tot
        y = torch.bmm(hidden, w_final) + b_final

        return y.view(batch_size, 1)


# ==========================================
# PHẦN 2: BỘ NÃO TRUNG TÂM (Centralized Manager)
# ==========================================

Transition = namedtuple('Transition', ('obs', 'state', 'action', 'reward', 'next_obs', 'next_state', 'done', 'mask'))


class QMIX_Brain:
    """
    Quản lý bộ nhớ, networks và quá trình training tập trung.
    Được chia sẻ giữa tất cả các Drone (Singleton-like pattern).
    """

    def __init__(self, n_drones, obs_dim, state_dim):
        self.n_drones = n_drones
        self.batch_size = 64
        self.gamma = 0.99
        self.lr = 0.0005

        # Khởi tạo networks
        self.agent_net = AgentNetwork(obs_dim, n_drones).to(device)
        self.target_agent_net = AgentNetwork(obs_dim, n_drones).to(device)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())

        # Vì ta routing tuần tự (1 agent active tại 1 thời điểm), n_agents input vào mixer thực tế là 1
        # (hoặc ta coi action là one-hot vector cho cả team).
        # Để đơn giản và hiệu quả cho routing, ta coi giá trị Q được chọn là input cho mixer.
        self.mixer = QMixer(1, state_dim).to(device)
        self.target_mixer = QMixer(1, state_dim).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.optimizer = optim.RMSprop(list(self.agent_net.parameters()) + list(self.mixer.parameters()), lr=self.lr)
        self.memory = deque(maxlen=10000)

        self.steps_done = 0
        self.target_update_freq = 200

    def push(self, *args):
        self.memory.append(Transition(*args))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Lấy mẫu ngẫu nhiên từ Replay Buffer
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        # Chuyển đổi sang Tensor
        obs_batch = torch.FloatTensor(np.array(batch.obs)).to(device)
        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(batch.action).view(-1, 1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).view(-1, 1).to(device)
        next_obs_batch = torch.FloatTensor(np.array(batch.next_obs)).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done_batch = torch.FloatTensor(batch.done).view(-1, 1).to(device)
        # Mask dùng để loại bỏ các neighbor không hợp lệ khi tính max Q next
        mask_batch = torch.FloatTensor(np.array(batch.mask)).to(device)

        # --- Tính Q_eval (Hiện tại) ---
        # 1. Agent tính Q cho tất cả actions
        q_acts = self.agent_net(obs_batch)
        # 2. Lấy Q của action đã chọn
        q_chosen = q_acts.gather(1, action_batch)
        # 3. Đưa qua Mixer (Lưu ý reshape để khớp dimension (Batch, 1, 1))
        q_tot_eval = self.mixer(q_chosen.view(-1, 1, 1), state_batch)

        # --- Tính Q_target (Tương lai) ---
        with torch.no_grad():
            # Double DQN: Chọn action tốt nhất bằng mạng Online
            next_q_acts_online = self.agent_net(next_obs_batch)
            # Áp dụng mask: Gán giá trị rất nhỏ cho action không hợp lệ (không phải neighbor)
            next_q_acts_online[mask_batch == 0] = -9999999
            next_best_action = next_q_acts_online.argmax(dim=1, keepdim=True)

            # Lấy giá trị Q từ mạng Target
            next_q_acts_target = self.target_agent_net(next_obs_batch)
            next_q_chosen_target = next_q_acts_target.gather(1, next_best_action)

            # Đưa qua Mixer Target
            q_tot_target = self.target_mixer(next_q_chosen_target.view(-1, 1, 1), next_state_batch)

            # Bellman Target
            y_target = reward_batch + self.gamma * q_tot_target * (1 - done_batch)

        # --- Cập nhật trọng số ---
        loss = F.mse_loss(q_tot_eval, y_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping để tránh bùng nổ gradient
        torch.nn.utils.clip_grad_norm_(self.agent_net.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 10)
        self.optimizer.step()

        # Soft Update Target Networks
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())


# ==========================================
# PHẦN 3: THUẬT TOÁN ROUTING (Tích hợp Simulator)
# ==========================================

class QMIX_Routing(BASE_routing):
    # Biến tĩnh để chia sẻ Brain giữa các instances của Routing (mỗi Drone là 1 instance)
    _brain = None

    def __init__(self, drone, simulator):
        super().__init__(drone, simulator)
        self.n_drones = simulator.n_drones

        # Định nghĩa kích thước Input
        # Obs: [Norm_X, Norm_Y, Norm_Energy, Norm_Dist_Depot, Packet_TTL]
        self.obs_dim = 5
        # Global State: [Drone1_X, Drone1_Y, Drone1_E, ..., Depot_X, Depot_Y]
        self.state_dim = (self.n_drones * 3) + 2

        # Khởi tạo Brain nếu chưa có (chạy 1 lần duy nhất)
        if QMIX_Routing._brain is None:
            QMIX_Routing._brain = QMIX_Brain(self.n_drones, self.obs_dim, self.state_dim)

        self.brain = QMIX_Routing._brain

        # Tham số Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

        # Lưu trạng thái tạm thời để dùng trong hàm feedback
        self.temp_transition = {}

        self.routing_hole_count = 0

    def get_local_obs(self, packet=None):
        """Tạo Local Observation cho Drone hiện tại"""
        d = self.drone
        env_w, env_h = self.simulator.env_width, self.simulator.env_height

        # Normalized Position
        x = d.coords[0] / env_w
        y = d.coords[1] / env_h

        # Normalized Energy
        e = d.residual_energy / d.initial_energy

        # Normalized Distance to Depot
        dist = util.euclidean_distance(d.coords, self.simulator.depot_coordinates)
        max_dist = (env_w ** 2 + env_h ** 2) ** 0.5
        dist_norm = dist / max_dist

        # Packet TTL (nếu không có packet thì mặc định là 1)
        ttl_norm = 1.0
        if packet:
            current_age = self.simulator.cur_step - packet.time_step_creation
            ttl_norm = 1.0 - (current_age / self.simulator.packets_max_ttl)
            ttl_norm = max(0, ttl_norm)

        return np.array([x, y, e, dist_norm, ttl_norm])

    def get_global_state(self):
        """Tạo Global State cho Mixer"""
        state = []
        env_w, env_h = self.simulator.env_width, self.simulator.env_height

        # Thông tin tất cả các drone
        for d in self.simulator.drones:
            state.append(d.coords[0] / env_w)
            state.append(d.coords[1] / env_h)
            state.append(d.residual_energy / d.initial_energy)

        # Thông tin Depot
        state.append(self.simulator.depot_coordinates[0] / env_w)
        state.append(self.simulator.depot_coordinates[1] / env_h)

        return np.array(state)

    def relay_selection(self, opt_neighbors, data):
        """Quyết định chọn Next Hop"""
        packet = data[0] if isinstance(data, tuple) else data

        # 1. Xử lý Routing Hole
        if not opt_neighbors:
            self.routing_hole_count += 1
            return None  # Báo hiệu cho simulator biết là lỗi

        # 2. Chuẩn bị dữ liệu đầu vào
        obs = self.get_local_obs(packet)
        state = self.get_global_state()

        # Tạo Mask: Chỉ cho phép chọn các neighbor hợp lệ
        # mask[i] = 1 nếu drone i là neighbor, ngược lại = 0
        mask = np.zeros(self.n_drones)
        neighbor_map = {}  # Map từ ID -> Object Drone

        for n in opt_neighbors:
            nid = n.identifier
            if nid < self.n_drones:
                mask[nid] = 1
                neighbor_map[nid] = n

        # 3. Chọn Action (Epsilon-Greedy)
        action_idx = -1

        if random.random() < self.epsilon:
            # Random chọn 1 neighbor hợp lệ
            chosen_neighbor = random.choice(opt_neighbors)
            action_idx = chosen_neighbor.identifier
        else:
            # Dùng mạng Neural
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                q_values = self.brain.agent_net(obs_tensor).cpu().numpy()

            # Áp dụng Mask: Gán giá trị cực thấp cho non-neighbors
            q_values[mask == 0] = -np.inf
            action_idx = np.argmax(q_values)

            # Lấy object Drone tương ứng
            if action_idx in neighbor_map:
                chosen_neighbor = neighbor_map[action_idx]
            else:
                # Fallback hiếm gặp
                chosen_neighbor = random.choice(opt_neighbors)
                action_idx = chosen_neighbor.identifier

        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 4. Lưu thông tin tạm thời để dùng cho feedback (Training)
        self.temp_transition[packet] = {
            'obs': obs,
            'state': state,
            'action': action_idx,
            'mask': mask
        }

        return chosen_neighbor

    def feedback(self, outcome, id_j, best_action, link_quality=0):
        """Nhận phản hồi từ môi trường và đẩy vào bộ nhớ"""
        # Do simulator gọi feedback nhưng không truyền packet object vào,
        # ta lấy packet gần nhất vừa xử lý (Giả định xử lý tuần tự)
        if not self.temp_transition:
            return

        # Lấy packet item ra khỏi buffer tạm
        packet, data = self.temp_transition.popitem()

        # 1. Tính Reward
        reward = 0
        done = False

        if outcome == 1:  # Thành công (Đến đích)
            reward = 10.0
            done = True
        elif outcome == 0:  # Chuyển tiếp thành công
            # Reward nhỏ khuyến khích chuyển tiếp + chất lượng link
            reward = 0.5 + link_quality
            done = False
        else:  # Thất bại (Mất gói, TTL hết, Routing hole)
            reward = -5.0
            done = True

        # Phạt nếu năng lượng thấp
        if self.drone.residual_energy / self.drone.initial_energy < 0.2:
            reward -= 2.0

        # 2. Lấy Next State (Trạng thái sau khi action)
        next_obs = self.get_local_obs(packet if not done else None)
        next_state = self.get_global_state()

        # Tạo mask cho next state (Để tính Q_target)
        # Lưu ý: Ta cần ước lượng neighbors tại next step.
        # Vì đây là feedback trễ, ta dùng neighbors hiện tại của self làm xấp xỉ
        # hoặc chấp nhận mask toàn 1 nếu không xác định được.
        # Ở đây dùng neighbors hiện tại của self drone làm xấp xỉ.
        current_neighbors = self.drone.get_neighbors(self.simulator.drones)
        next_mask = np.zeros(self.n_drones)
        for n in current_neighbors:
            if n.identifier < self.n_drones:
                next_mask[n.identifier] = 1

        # 3. Đẩy vào Replay Buffer
        self.brain.push(
            data['obs'],
            data['state'],
            data['action'],
            reward,
            next_obs,
            next_state,
            float(done),
            next_mask  # Lưu mask cho next state
        )

        # 4. Kích hoạt Training
        self.brain.train()

    def get_performance_metrics(self):
        """Trả về metrics để hiển thị"""
        return {
            'epsilon': self.epsilon,
            'routing_hole': self.routing_hole_count,
            'buffer_size': len(self.brain.memory)
        }