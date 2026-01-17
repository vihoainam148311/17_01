# mappo_routing.py
# Multi-Agent Proximal Policy Optimization (MAPPO) cho hop-by-hop UAV routing
# CTDE, parameter sharing, asynchronous (mỗi lần chỉ UAV đang giữ gói quyết định)

import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
from src.utilities import config

# Thiết bị tính toán
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorNet(nn.Module):
    """
    Mạng Actor: nhận observation cục bộ của UAV, xuất logits cho các action (chọn relay).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logits = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (batch, obs_dim)
        return: logits (batch, n_actions)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.fc_logits(x)
        return logits


class CriticNet(nn.Module):
    """
    Mạng Critic tập trung (centralized): nhận global state, xuất V(s).
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch, state_dim)
        return: value (batch, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.fc_value(x)
        return v


Transition = namedtuple(
    "Transition",
    ["obs", "state", "mask", "action", "logp", "value", "reward", "done"],
)


class MAPPOBrain:
    """
    Bộ não MAPPO dùng chung cho tất cả UAV (CTDE, parameter sharing).
    """

    def __init__(self, n_drones: int, obs_dim: int, state_dim: int):
        self.n_drones = n_drones
        self.obs_dim = obs_dim
        self.state_dim = state_dim

        # Hyper-parameters (có thể override bằng config.* nếu có)
        self.gamma = getattr(config, "MAPPO_GAMMA", 0.99)
        self.clip_eps = getattr(config, "MAPPO_CLIP_EPS", 0.2)
        self.lr = getattr(config, "MAPPO_LR", 3e-4)
        self.entropy_coef = getattr(config, "MAPPO_ENTROPY_COEF", 0.01)
        self.value_coef = getattr(config, "MAPPO_VALUE_COEF", 0.5)
        self.max_grad_norm = getattr(config, "MAPPO_MAX_GRAD_NORM", 0.5)
        self.rollout_size = getattr(config, "MAPPO_ROLLOUT_SIZE", 2048)
        self.ppo_epochs = getattr(config, "MAPPO_EPOCHS", 4)
        self.minibatch_size = getattr(config, "MAPPO_MINIBATCH_SIZE", 256)

        # Mạng Actor & Critic
        self.actor = ActorNet(obs_dim, n_drones).to(device)
        self.critic = CriticNet(state_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
        )

        # Bộ đệm on-policy (rollout buffer)
        self.buffer = []

        # Thống kê training
        self.total_steps = 0
        self.update_steps = 0

    # -----------------------------
    # HÀM HÀNH ĐỘNG (ACTOR FORWARD)
    # -----------------------------
    def act(self, obs: np.ndarray, state: np.ndarray, mask: np.ndarray):
        """
        Trả về action, log_prob, value ứng với (obs, state, mask) hiện tại.
        obs: (obs_dim,)
        state: (state_dim,)
        mask: (n_drones,) với 1 = neighbor hợp lệ, 0 = invalid
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

        # Actor: logits cho tất cả UAV
        logits = self.actor(obs_t)  # (1, n_drones)

        # Áp dụng mask: những UAV không phải neighbor không được chọn
        logits = logits.masked_fill(~mask_t, -1e9)

        # Dùng Categorical với logits (an toàn số hơn softmax thủ công)
        dist = Categorical(logits=logits)
        action = dist.sample()               # (1,)
        logp = dist.log_prob(action)         # (1,)

        # Critic: V(s)
        value = self.critic(state_t)         # (1, 1)

        return (
            int(action.item()),
            float(logp.item()),
            float(value.item()),
        )

    # -----------------------------
    # LƯU TRANSITION VÀO BUFFER
    # -----------------------------
    def store_transition(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        mask: np.ndarray,
        action: int,
        logp: float,
        value: float,
        reward: float,
        done: bool,
    ):
        tr = Transition(
            obs=np.array(obs, copy=False),
            state=np.array(state, copy=False),
            mask=np.array(mask, copy=False),
            action=int(action),
            logp=float(logp),
            value=float(value),
            reward=float(reward),
            done=bool(done),
        )
        self.buffer.append(tr)
        self.total_steps += 1

        if len(self.buffer) >= self.rollout_size:
            self.update()

    # -----------------------------
    # PPO UPDATE (ON-POLICY)
    # -----------------------------
    def update(self):
        if not self.buffer:
            return

        self.update_steps += 1

        # Chuyển buffer thành Tensor
        obs = torch.as_tensor(
            np.stack([t.obs for t in self.buffer]), dtype=torch.float32, device=device
        )  # (T, obs_dim)
        states = torch.as_tensor(
            np.stack([t.state for t in self.buffer]), dtype=torch.float32, device=device
        )  # (T, state_dim)
        masks = torch.as_tensor(
            np.stack([t.mask for t in self.buffer]), dtype=torch.bool, device=device
        )  # (T, n_drones)
        actions = torch.as_tensor(
            [t.action for t in self.buffer], dtype=torch.long, device=device
        )  # (T,)
        old_logps = torch.as_tensor(
            [t.logp for t in self.buffer], dtype=torch.float32, device=device
        )  # (T,)
        values = torch.as_tensor(
            [t.value for t in self.buffer], dtype=torch.float32, device=device
        )  # (T,)
        rewards = torch.as_tensor(
            [t.reward for t in self.buffer], dtype=torch.float32, device=device
        )  # (T,)
        dones = torch.as_tensor(
            [t.done for t in self.buffer], dtype=torch.float32, device=device
        )  # (T,)

        T = rewards.size(0)

        # Tính returns bằng Monte Carlo (không dùng GAE cho đơn giản)
        returns = torch.zeros(T, dtype=torch.float32, device=device)
        R = 0.0
        for t in reversed(range(T)):
            if dones[t] > 0.5:
                R = 0.0
            R = rewards[t] + self.gamma * R
            returns[t] = R

        # Advantage = returns - values
        advantages = returns - values
        # Chuẩn hoá advantage để ổn định hơn
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Lặp nhiều epoch PPO
        indices = np.arange(T)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, T, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                if len(mb_idx) == 0:
                    continue

                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=device)

                obs_mb = obs[mb_idx_t]          # (B, obs_dim)
                states_mb = states[mb_idx_t]    # (B, state_dim)
                masks_mb = masks[mb_idx_t]      # (B, n_drones)
                actions_mb = actions[mb_idx_t]  # (B,)
                old_logps_mb = old_logps[mb_idx_t]  # (B,)
                returns_mb = returns[mb_idx_t]  # (B,)
                adv_mb = advantages[mb_idx_t]   # (B,)

                # Tính lại logits, logprob và entropy với policy hiện tại
                logits = self.actor(obs_mb)           # (B, n_drones)
                logits = logits.masked_fill(~masks_mb, -1e9)
                dist = Categorical(logits=logits)

                new_logps = dist.log_prob(actions_mb)  # (B,)
                entropy = dist.entropy().mean()

                # Ratio r_t(θ)
                ratio = torch.exp(new_logps - old_logps_mb)

                # PPO objective
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                values_pred = self.critic(states_mb).squeeze(-1)  # (B,)
                value_loss = F.mse_loss(values_pred, returns_mb)

                # Tổng loss
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Cập nhật
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

        # Xoá buffer sau khi update
        self.buffer.clear()

    @property
    def buffer_size(self) -> int:
        return len(self.buffer)


class MAPPO_Routing(BASE_routing):
    """
    Thuật toán định tuyến dựa trên MAPPO (Multi-Agent PPO, CTDE).
    Mỗi UAV là một agent; tại thời điểm giữ gói tin, agent đó quyết định relay.
    """

    _brain = None  # chia sẻ giữa tất cả instances

    def __init__(self, drone, simulator):
        super().__init__(drone, simulator)

        self.n_drones = simulator.n_drones
        # Obs: [x_norm, y_norm, energy_ratio, dist_to_depot_norm, ttl_norm]
        self.obs_dim = 5
        # State: [x_1, y_1, e_1, ..., x_N, y_N, e_N, depot_x, depot_y]
        self.state_dim = self.n_drones * 3 + 2

        # Khởi tạo brain chung nếu chưa có
        if MAPPO_Routing._brain is None:
            MAPPO_Routing._brain = MAPPOBrain(
                n_drones=self.n_drones, obs_dim=self.obs_dim, state_dim=self.state_dim
            )

        self.brain: MAPPOBrain = MAPPO_Routing._brain

        # Buffer tạm để nối action với feedback (theo packet)
        self.temp_transition = {}

        # Thống kê
        self.routing_hole_count = 0

    # -----------------------------
    # TẠO OBSERVATION & GLOBAL STATE
    # -----------------------------
    def get_local_obs(self, packet=None) -> np.ndarray:
        """
        Observation cục bộ của UAV:
        [x_norm, y_norm, energy_ratio, dist_to_depot_norm, ttl_norm]
        """
        d = self.drone
        env_w, env_h = self.simulator.env_width, self.simulator.env_height

        x_norm = d.coords[0] / env_w
        y_norm = d.coords[1] / env_h
        energy_ratio = d.residual_energy / d.initial_energy

        dist = util.euclidean_distance(d.coords, self.simulator.depot_coordinates)
        max_dist = (env_w ** 2 + env_h ** 2) ** 0.5
        dist_norm = dist / max_dist

        # TTL chuẩn hoá theo tuổi gói (nếu có)
        ttl_norm = 1.0
        if packet is not None:
            current_age = self.simulator.cur_step - packet.time_step_creation
            ttl_norm = 1.0 - (current_age / self.simulator.packets_max_ttl)
            ttl_norm = max(0.0, ttl_norm)

        return np.array([x_norm, y_norm, energy_ratio, dist_norm, ttl_norm], dtype=np.float32)

    def get_global_state(self) -> np.ndarray:
        """
        Global state cho Critic:
        [x_1, y_1, e_1, ..., x_N, y_N, e_N, depot_x, depot_y]
        """
        env_w, env_h = self.simulator.env_width, self.simulator.env_height
        state = []

        for d in self.simulator.drones:
            state.append(d.coords[0] / env_w)
            state.append(d.coords[1] / env_h)
            state.append(d.residual_energy / d.initial_energy)

        depot_x, depot_y = self.simulator.depot_coordinates
        state.append(depot_x / env_w)
        state.append(depot_y / env_h)

        return np.array(state, dtype=np.float32)

    # -----------------------------
    # CHỌN RELAY (NEXT HOP)
    # -----------------------------
    def relay_selection(self, opt_neighbors, data):
        """
        Hàm được BASE_routing.send_packets() gọi.
        opt_neighbors: danh sách Drone láng giềng hợp lệ.
        data: packet tuple hoặc DataPacket (tương tự QMIX_Routing).
        """
        packet = data[0] if isinstance(data, tuple) else data

        # Routing hole
        if not opt_neighbors:
            self.routing_hole_count += 1
            return None

        # Tạo obs, state
        obs = self.get_local_obs(packet)
        state = self.get_global_state()

        # Tạo mask cho n_drones: 1 nếu là neighbor hợp lệ, 0 nếu không
        mask = np.zeros(self.n_drones, dtype=np.bool_)
        neighbor_map = {}
        for n in opt_neighbors:
            nid = n.identifier
            if 0 <= nid < self.n_drones:
                mask[nid] = True
                neighbor_map[nid] = n

        # Nếu vì lý do nào đó không có neighbor hợp lệ sau khi mask
        if not mask.any():
            self.routing_hole_count += 1
            return None

        # Lấy action từ MAPPO brain
        action_idx, logp, value = self.brain.act(obs, state, mask)

        # Map sang Drone object; nếu lỗi, fallback random
        if action_idx in neighbor_map:
            chosen_neighbor = neighbor_map[action_idx]
        else:
            chosen_neighbor = random.choice(opt_neighbors)
            action_idx = chosen_neighbor.identifier

        # Lưu tạm để dùng khi feedback
        self.temp_transition[packet] = {
            "obs": obs,
            "state": state,
            "mask": mask,
            "action": action_idx,
            "logp": logp,
            "value": value,
        }

        return chosen_neighbor

    # -----------------------------
    # NHẬN FEEDBACK TỪ MÔ PHỎNG
    # -----------------------------
    def feedback(self, outcome, id_j, best_action, link_quality=0):
        """
        Simulator gọi khi:
        - outcome = 1: packet đến Depot
        - outcome = 0: forward thành công, packet vẫn trong mạng
        - outcome khác: lỗi (timeout, mất gói, routing hole, ...)
        """
        if not self.temp_transition:
            return

        # Lấy packet và dữ liệu tạm gần nhất (giả định xử lý tuần tự)
        packet, data = self.temp_transition.popitem()

        # 1. Tính reward
        if outcome == 1:
            reward = 10.0
            done = True
        elif outcome == 0:
            reward = 0.5 + float(link_quality)
            done = False
        else:
            reward = -5.0
            done = True

        # Phạt thêm nếu UAV đang gần hết năng lượng
        energy_ratio = self.drone.residual_energy / self.drone.initial_energy
        if energy_ratio < 0.2:
            reward -= 2.0

        # 2. Lấy next_obs, next_state (không bắt buộc cho PPO ở đây, chủ yếu để debug/shaping)
        next_obs = self.get_local_obs(packet if not done else None)
        next_state = self.get_global_state()
        _ = next_obs, next_state  # tránh cảnh báo "unused"

        # 3. Lưu vào buffer MAPPO
        self.brain.store_transition(
            obs=data["obs"],
            state=data["state"],
            mask=data["mask"],
            action=data["action"],
            logp=data["logp"],
            value=data["value"],
            reward=reward,
            done=done,
        )

    # -----------------------------
    # METRICS PHỤC VỤ LOGGING
    # -----------------------------
    def get_performance_metrics(self):
        """
        Hàm tuỳ chọn để simulator gọi và in ra.
        """
        return {
            "buffer_size": self.brain.buffer_size,
            "routing_hole": self.routing_hole_count,
            "total_steps": self.brain.total_steps,
            "update_steps": self.brain.update_steps,
        }
