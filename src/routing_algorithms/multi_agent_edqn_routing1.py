# multi_agent_edqn_routing.py
# Multi-Agent Enhanced DQN với CTDE (Centralized Training Decentralized Execution) + Communication
# Dành cho FANET UAV routing - PEP8, Python 3.7+

import random
import math
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
from src.utilities import config as cfg


# ====================== NEURAL NETWORKS ======================

class BalancedDQN(nn.Module):
    """
    Mạng DQN cân bằng cho single agent.
    Kiến trúc: Input → FC(256) → BN → Dropout → FC(128) → BN → Dropout → FC(64) → Output
    """

    def __init__(self, input_dim, output_dim):
        super(BalancedDQN, self).__init__()
        
        # Validate dimensions
        if input_dim <= 0:
            raise ValueError(f"[ERROR] input_dim must be > 0, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"[ERROR] output_dim must be > 0, got {output_dim}")
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        if x.numel() == 0:
            return torch.zeros(1)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        x = self.fc1(x)
        if batch_size > 1:
            x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x.squeeze(0) if x.shape[0] == 1 else x


class MessageEncoder(nn.Module):
    """Mã hóa state thành message vector để broadcast"""

    def __init__(self, state_dim, message_dim=32):
        super(MessageEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, message_dim),
            nn.Tanh()  # Normalize message
        )

    def forward(self, state):
        return self.encoder(state)


class MessageDecoder(nn.Module):
    """Giải mã message từ neighbors"""

    def __init__(self, message_dim=32, output_dim=16):
        super(MessageDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(message_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, message):
        return self.decoder(message)


class AttentionAggregator(nn.Module):
    """Attention mechanism để aggregate messages từ nhiều neighbors"""

    def __init__(self, message_dim=32):
        super(AttentionAggregator, self).__init__()
        self.query = nn.Linear(message_dim, message_dim)
        self.key = nn.Linear(message_dim, message_dim)
        self.value = nn.Linear(message_dim, message_dim)

    def forward(self, own_message, neighbor_messages):
        """
        own_message: (message_dim,)
        neighbor_messages: list of (message_dim,)
        """
        if len(neighbor_messages) == 0:
            return torch.zeros_like(own_message)

        # Stack neighbor messages
        neighbors = torch.stack(neighbor_messages)  # (N, message_dim)

        # Attention weights
        q = self.query(own_message).unsqueeze(0)  # (1, message_dim)
        k = self.key(neighbors)  # (N, message_dim)
        v = self.value(neighbors)  # (N, message_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.T) / math.sqrt(own_message.shape[0])  # (1, N)
        weights = F.softmax(scores, dim=1)  # (1, N)

        # Weighted sum
        aggregated = torch.matmul(weights, v).squeeze(0)  # (message_dim,)

        return aggregated


# ====================== CENTRALIZED TRAINER ======================

class CentralizedTrainer:
    """
    Centralized Training component:
    - Quản lý shared networks
    - Thu thập experience từ tất cả agents
    - Training với global view
    """

    def __init__(self, n_agents, state_size, action_size, config):
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size

        # Shared networks cho tất cả agents
        self.shared_model = BalancedDQN(state_size, action_size)
        self.shared_target = BalancedDQN(state_size, action_size)
        self.shared_target.load_state_dict(self.shared_model.state_dict())

        # Communication networks
        self.message_encoder = MessageEncoder(state_size, message_dim=32)
        self.message_decoder = MessageDecoder(message_dim=32, output_dim=16)
        self.attention_aggregator = AttentionAggregator(message_dim=32)

        # Optimizer cho tất cả networks
        all_params = (
            list(self.shared_model.parameters()) +
            list(self.message_encoder.parameters()) +
            list(self.message_decoder.parameters()) +
            list(self.attention_aggregator.parameters())
        )
        self.optimizer = torch.optim.Adam(
            all_params,
            lr=config.get('learning_rate', 3e-4),
            weight_decay=1e-5
        )

        self.loss_function = nn.MSELoss()

        # Global replay buffers
        self.global_memory = deque(maxlen=config.get('memory_size', 10000))
        self.global_priority_memory = deque(maxlen=config.get('priority_size', 3000))

        # Training config
        self.gamma = config.get('gamma', 0.98)
        self.batch_size = config.get('batch_size', 128)
        self.target_update_freq = config.get('target_update_freq', 100)

        # Statistics
        self.training_steps = 0
        self.total_loss = 0.0
        self.loss_history = deque(maxlen=1000)

    def collect_experience(self, agent_id, state, action, reward, next_state, done, priority=False):
        """Thu thập experience từ một agent"""
        experience = (agent_id, state, action, reward, next_state, done)

        if priority or abs(reward) > 2.5:
            self.global_priority_memory.append(experience)
        else:
            self.global_memory.append(experience)

    def sample_batch(self):
        """Lấy mẫu balanced batch: 50% priority + 50% normal"""
        total_size = len(self.global_memory) + len(self.global_priority_memory)
        if total_size < self.batch_size:
            return None

        batch = []

        # 50% từ priority memory
        if self.global_priority_memory:
            k = min(self.batch_size // 2, len(self.global_priority_memory))
            batch.extend(random.sample(list(self.global_priority_memory), k))

        # 50% từ normal memory
        remain = self.batch_size - len(batch)
        if remain > 0 and self.global_memory:
            batch.extend(random.sample(list(self.global_memory), min(remain, len(self.global_memory))))

        return batch

    def train_step(self):
        """Một bước training với DDQN"""
        batch = self.sample_batch()
        if batch is None or len(batch) == 0:
            return None

        total_loss = 0.0
        successful_updates = 0

        for agent_id, state, action, reward, next_state, done in batch:
            if next_state is None or len(next_state) == 0:
                continue
            
            if state is None or len(state) == 0:
                continue

            try:
                # Convert to tensors
                state_t = torch.as_tensor(state, dtype=torch.float32)
                next_state_t = torch.as_tensor(next_state, dtype=torch.float32)
                
                # Validate action index
                if action < 0 or action >= self.action_size:
                    continue

                # DDQN: online network chọn action, target network ước lượng value
                target_val = reward
                if not done:
                    with torch.no_grad():
                        # Online chọn best action
                        next_q_online = self.shared_model(next_state_t)
                        if next_q_online.numel() == 0:
                            continue
                        next_action = int(torch.argmax(next_q_online).item())
                        
                        # Validate next_action
                        if next_action < 0 or next_action >= self.action_size:
                            continue

                        # Target ước lượng Q-value
                        next_q_target = self.shared_target(next_state_t)
                        if next_q_target.numel() > next_action:
                            target_val = reward + self.gamma * float(next_q_target[next_action].item())

                # Update Q(s, a)
                current_q = self.shared_model(state_t)
                if current_q.numel() > action:
                    target_q = current_q.clone().detach()
                    target_q[action] = target_val

                    loss = self.loss_function(current_q, target_q)
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 1.0)
                    self.optimizer.step()

                    total_loss += loss.item()
                    successful_updates += 1

            except Exception as e:
                print(f"[WARN] Training step error for agent {agent_id}: {e}")
                continue

        self.training_steps += 1

        # Update target network định kỳ
        if self.training_steps % self.target_update_freq == 0:
            self.shared_target.load_state_dict(self.shared_model.state_dict())

        # Statistics
        if successful_updates > 0:
            avg_loss = total_loss / successful_updates
            self.loss_history.append(avg_loss)
            return avg_loss

        return None

    def get_training_stats(self):
        """Lấy thống kê training"""
        return {
            'training_steps': self.training_steps,
            'avg_loss': float(np.mean(self.loss_history)) if self.loss_history else 0.0,
            'memory_size': len(self.global_memory),
            'priority_memory_size': len(self.global_priority_memory),
        }


# ====================== COMMUNICATION AGENT ======================

class CommunicatingAgent:
    """
    Agent với khả năng communication:
    - Broadcast state/intention
    - Nhận và xử lý messages từ neighbors
    - Quyết định với augmented state
    """

    def __init__(self, drone, simulator, trainer, config):
        self.drone = drone
        self.simulator = simulator
        self.trainer = trainer
        self.config = config

        # Agent info
        self.agent_id = drone.identifier
        self.state_size = config.get('state_size', 10)
        self.action_size = simulator.n_drones

        # Epsilon-greedy
        self.epsilon = config.get('epsilon_start', 0.95)
        self.epsilon_min = config.get('epsilon_min', 0.02)
        self.epsilon_decay = config.get('epsilon_decay', 0.9995)

        # Communication buffers
        self.received_messages = {}  # {sender_id: message_data}
        self.last_broadcast_time = 0
        self.broadcast_interval = config.get('broadcast_interval', 5)

        # Performance tracking
        self.neighbor_history = deque(maxlen=200)
        self.delivery_success_history = deque(maxlen=500)
        self.reward_history = deque(maxlen=2000)
        self.success_count = 0
        self.total_attempts = 0
        self.conservative_mode = False

        # Episode memory
        self.last_state = None
        self.last_action = None
        self.last_augmented_state = None

    # -------------------- STATE REPRESENTATION --------------------

    def get_local_state(self):
        """
        State cục bộ của agent (10 dim):
        [x_norm, y_norm, energy, dist2depot, dx_norm, dy_norm,
         speed_norm, connectivity, avg_nb_energy, best_link_q]
        """
        try:
            drone = self.drone
            if drone is None:
                return np.zeros(self.state_size, dtype=np.float32)

            # Normalize position
            x_norm = max(0.0, min(1.0, drone.coords[0] / max(self.simulator.env_width, 1)))
            y_norm = max(0.0, min(1.0, drone.coords[1] / max(self.simulator.env_height, 1)))

            # Energy
            energy_ratio = max(0.0, min(1.0, drone.residual_energy / max(drone.initial_energy, 1)))

            # Speed
            speed_norm = max(0.0, min(1.0, drone.speed / 20.0))

            # Distance to depot
            depot_distance = util.euclidean_distance(drone.coords, self.simulator.depot_coordinates)
            max_distance = math.sqrt(self.simulator.env_width ** 2 + self.simulator.env_height ** 2)
            distance_ratio = max(0.0, min(1.0, depot_distance / max(max_distance, 1)))

            # Direction to depot
            dx = (self.simulator.depot_coordinates[0] - drone.coords[0]) / max(self.simulator.env_width, 1)
            dy = (self.simulator.depot_coordinates[1] - drone.coords[1]) / max(self.simulator.env_height, 1)

            # Neighbor analysis
            current_step = getattr(self.simulator, 'cur_step', 0)
            valid_neighbors = 0
            avg_neighbor_energy = 0.0
            best_link_quality = 0.0

            if hasattr(drone, 'neighbor_table') and drone.neighbor_table is not None:
                neighbor_energies = []
                link_qualities = []

                n_rows = min(len(drone.neighbor_table), self.simulator.n_drones)
                for i in range(n_rows):
                    arrival_time = drone.neighbor_table[i, 6]
                    if arrival_time > 0 and (current_step - arrival_time) < self.simulator.ExpireTime:
                        valid_neighbors += 1

                        # Neighbor energy
                        if i < len(self.simulator.drones):
                            nb = self.simulator.drones[i]
                            if nb and hasattr(nb, 'residual_energy'):
                                er = nb.residual_energy / max(nb.initial_energy, 1)
                                neighbor_energies.append(er)

                        # Link quality
                        lq = drone.neighbor_table[i, 12]
                        if not np.isnan(lq) and lq >= 0:
                            link_qualities.append(lq)

                avg_neighbor_energy = np.mean(neighbor_energies) if neighbor_energies else 0.0
                best_link_quality = np.max(link_qualities) if link_qualities else 0.0

            connectivity = max(0.0, min(1.0, valid_neighbors / max(self.simulator.n_drones * 0.3, 1)))
            self.neighbor_history.append(valid_neighbors)

            state = np.array([
                x_norm, y_norm, energy_ratio, distance_ratio,
                dx, dy, speed_norm, connectivity,
                avg_neighbor_energy, best_link_quality
            ], dtype=np.float32)

            return state

        except Exception as e:
            print(f"[ERROR] Agent {self.agent_id} get_local_state failed: {e}")
            return np.zeros(self.state_size, dtype=np.float32)

    def get_augmented_state(self):
        """
        State mở rộng với thông tin từ communication:
        [local_state(10), aggregated_neighbor_info(6)]
        Total: 16 dimensions
        """
        local_state = self.get_local_state()

        if not self.received_messages:
            # Không có message -> padding zeros
            padding = np.zeros(6, dtype=np.float32)
            return np.concatenate([local_state, padding])

        try:
            # Decode messages từ neighbors
            neighbor_encoded_states = []
            for msg_data in self.received_messages.values():
                msg_tensor = torch.tensor(msg_data['message'], dtype=torch.float32)
                decoded = self.trainer.message_decoder(msg_tensor)
                neighbor_encoded_states.append(decoded.detach().numpy())

            if neighbor_encoded_states:
                # Aggregate neighbor information
                aggregated = np.mean(neighbor_encoded_states, axis=0)
                # Take first 6 dimensions
                neighbor_info = aggregated[:6]
            else:
                neighbor_info = np.zeros(6, dtype=np.float32)

            augmented_state = np.concatenate([local_state, neighbor_info])
            return augmented_state

        except Exception as e:
            print(f"[WARN] Agent {self.agent_id} augmented state error: {e}")
            padding = np.zeros(6, dtype=np.float32)
            return np.concatenate([local_state, padding])

    # -------------------- COMMUNICATION --------------------

    def broadcast_intention(self):
        """
        Broadcast state và intention cho neighbors trong communication range
        """
        current_step = getattr(self.simulator, 'cur_step', 0)

        # Broadcast theo interval để giảm overhead
        if current_step - self.last_broadcast_time < self.broadcast_interval:
            return None

        try:
            state = self.get_local_state()
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Encode thành message vector
            with torch.no_grad():
                message = self.trainer.message_encoder(state_tensor)

            self.last_broadcast_time = current_step

            return {
                'sender_id': self.agent_id,
                'message': message.numpy(),
                'timestamp': current_step,
                'position': self.drone.coords,
                'energy': self.drone.residual_energy / self.drone.initial_energy
            }

        except Exception as e:
            print(f"[WARN] Agent {self.agent_id} broadcast failed: {e}")
            return None

    def receive_messages(self, messages):
        """
        Nhận messages từ neighbors
        Chỉ giữ messages mới nhất và trong comm range
        """
        current_step = getattr(self.simulator, 'cur_step', 0)
        message_timeout = 20  # steps

        # Clear old messages
        expired = [sid for sid, msg in self.received_messages.items()
                   if current_step - msg['timestamp'] > message_timeout]
        for sid in expired:
            del self.received_messages[sid]

        # Add new messages
        for msg in messages:
            if msg['sender_id'] != self.agent_id:
                self.received_messages[msg['sender_id']] = msg

    # -------------------- ACTION SELECTION --------------------

    def select_action(self, valid_neighbors_mask):
        """
        Chọn action (relay) với epsilon-greedy + augmented state
        valid_neighbors_mask: binary array chỉ neighbors hợp lệ
        """
        try:
            # Get augmented state
            augmented_state = self.get_augmented_state()
            self.last_augmented_state = augmented_state

            # Epsilon-greedy
            if random.random() < self.epsilon:
                # Explore: chọn random trong valid neighbors
                valid_indices = np.where(valid_neighbors_mask)[0]
                if len(valid_indices) == 0:
                    return None
                action = random.choice(valid_indices)
            else:
                # Exploit: chọn theo Q-values
                with torch.no_grad():
                    state_tensor = torch.tensor(augmented_state, dtype=torch.float32)
                    q_values = self.trainer.shared_model(state_tensor)

                    # Mask invalid neighbors
                    q_values_np = q_values.numpy()
                    q_values_np[~valid_neighbors_mask] = -float('inf')

                    action = int(np.argmax(q_values_np))

                    if not valid_neighbors_mask[action]:
                        # Fallback
                        valid_indices = np.where(valid_neighbors_mask)[0]
                        if len(valid_indices) == 0:
                            return None
                        action = random.choice(valid_indices)

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            return action

        except Exception as e:
            print(f"[ERROR] Agent {self.agent_id} select_action failed: {e}")
            # Fallback to random
            valid_indices = np.where(valid_neighbors_mask)[0]
            return random.choice(valid_indices) if len(valid_indices) > 0 else None

    # -------------------- REWARD & LEARNING --------------------

    def compute_reward(self, outcome):
        """
        Compute reward với ưu tiên delivery success
        outcome: 1 (success), 0 (forward), -1 (failure)
        """
        try:
            if outcome == 1:
                base = 5.0
                self.success_count += 1
                self.delivery_success_history.append(1.0)
            elif outcome == -1:
                base = -3.0
                self.delivery_success_history.append(0.0)
            else:
                base = 0.5

            self.total_attempts += 1

            # Adjust based on recent success rate
            if len(self.delivery_success_history) > 10:
                recent = np.mean(list(self.delivery_success_history)[-20:])
                if recent > 0.8:
                    base += 1.0
                elif recent < 0.6:
                    base -= 0.5

            # Energy conservation
            er = self.drone.residual_energy / max(self.drone.initial_energy, 1)
            if er > 0.7:
                base += 0.3
            elif er < 0.3:
                base -= 0.2

            # Connectivity bonus
            avg_nb = np.mean(self.neighbor_history) if self.neighbor_history else 1.0
            if avg_nb > 2.0:
                base += 0.2

            # Conservative mode bonus
            if self.conservative_mode and outcome == 1:
                base += 0.5

            return float(max(-5.0, min(7.0, base)))

        except Exception as e:
            print(f"[WARN] Agent {self.agent_id} reward computation error: {e}")
            return 0.0

    def update(self, action, reward, next_state, done):
        """
        Update sau khi nhận feedback
        Gửi experience lên centralized trainer
        """
        try:
            if self.last_augmented_state is None:
                return

            # Send to centralized trainer
            priority = (done and reward > 0) or abs(reward) > 2.5
            self.trainer.collect_experience(
                self.agent_id,
                self.last_augmented_state,
                action,
                reward,
                next_state,
                done,
                priority=priority
            )

            self.reward_history.append(reward)

            # Update conservative mode
            if len(self.delivery_success_history) > 50:
                success_rate = np.mean(list(self.delivery_success_history)[-50:])
                self.conservative_mode = success_rate < 0.7

        except Exception as e:
            print(f"[ERROR] Agent {self.agent_id} update failed: {e}")

    # -------------------- STATISTICS --------------------

    def get_stats(self):
        """Get agent statistics"""
        return {
            'agent_id': self.agent_id,
            'epsilon': float(self.epsilon),
            'success_count': self.success_count,
            'total_attempts': self.total_attempts,
            'success_rate': self.success_count / max(self.total_attempts, 1),
            'avg_reward': float(np.mean(self.reward_history)) if self.reward_history else 0.0,
            'avg_neighbors': float(np.mean(self.neighbor_history)) if self.neighbor_history else 0.0,
            'conservative_mode': self.conservative_mode,
            'messages_received': len(self.received_messages)
        }


# ====================== RELAY COORDINATOR ======================

class RelayCoordinator:
    """
    Giải quyết xung đột khi nhiều agents chọn cùng relay:
    - Track pending selections
    - Resolve conflicts theo priority (Q-value)
    - Đảm bảo load balancing
    """

    def __init__(self):
        self.pending_selections = defaultdict(list)  # {relay_id: [(agent_id, priority, timestamp)]}
        self.relay_load = defaultdict(int)  # {relay_id: current_load}
        self.max_relay_load = 3  # Max concurrent packets per relay

    def register_selection(self, agent_id, relay_id, priority, timestamp):
        """Đăng ký lựa chọn relay từ một agent"""
        self.pending_selections[relay_id].append((agent_id, priority, timestamp))

    def resolve_conflicts(self, current_timestamp):
        """
        Giải quyết xung đột và return mapping {agent_id: approved_relay_id}
        """
        approved = {}

        for relay_id, selections in self.pending_selections.items():
            if len(selections) == 0:
                continue

            # Sort by priority (descending) then timestamp (ascending)
            selections.sort(key=lambda x: (-x[1], x[2]))

            # Approve agents cho đến khi relay đầy
            current_load = self.relay_load[relay_id]
            for agent_id, priority, ts in selections:
                if current_load < self.max_relay_load:
                    approved[agent_id] = relay_id
                    current_load += 1
                else:
                    # Relay đầy -> agent này phải chọn relay khác
                    approved[agent_id] = None  # Signal to reselect

            self.relay_load[relay_id] = current_load

        # Clear pending
        self.pending_selections.clear()

        return approved

    def release_relay(self, relay_id):
        """Giải phóng relay sau khi gửi gói xong"""
        if relay_id in self.relay_load:
            self.relay_load[relay_id] = max(0, self.relay_load[relay_id] - 1)

    def reset(self):
        """Reset coordinator (mỗi step hoặc episode)"""
        self.pending_selections.clear()
        # Giữ relay_load để tracking qua time


# ====================== MULTI-AGENT SYSTEM ======================

class MultiAgentEDQN_Routing(BASE_routing):
    """
    Multi-Agent Enhanced DQN Routing với:
    - Centralized Training Decentralized Execution (CTDE)
    - Communication giữa agents
    - Relay coordination để tránh xung đột
    """

    def __init__(self, drone, simulator):
        super().__init__(drone, simulator)

        # Validate inputs
        if drone is None:
            raise ValueError("[ERROR] Drone cannot be None")
        if simulator is None:
            raise ValueError("[ERROR] Simulator cannot be None")
        if not hasattr(simulator, 'drones') or len(simulator.drones) == 0:
            raise ValueError("[ERROR] Simulator must have drones initialized")

        # Config
        self.config = self._load_config()

        # Initialize MARL system (singleton per simulator)
        if not hasattr(simulator, '_marl_system') or simulator._marl_system is None:
            print(f"[INFO] Initializing Multi-Agent EDQN System with {len(simulator.drones)} drones...")
            try:
                simulator._marl_system = self._initialize_marl_system(simulator)
                print(f"[INFO] MARL System initialized successfully")
            except Exception as e:
                print(f"[ERROR] Failed to initialize MARL system: {e}")
                raise

        self.marl_system = simulator._marl_system

        # Get agent cho drone này
        if drone.identifier not in self.marl_system['agents']:
            raise ValueError(f"[ERROR] Agent for drone {drone.identifier} not found in MARL system")
        
        self.agent = self.marl_system['agents'][drone.identifier]

        # For BASE_routing compatibility
        self.last_state = None
        self.last_action = None

    def _load_config(self):
        """Load configuration từ config file"""
        return {
            'state_size': getattr(cfg, 'MARL_STATE_SIZE', 10),
            'memory_size': getattr(cfg, 'MARL_MEMORY_SIZE', 10000),
            'priority_size': getattr(cfg, 'MARL_PRIORITY_SIZE', 3000),
            'gamma': getattr(cfg, 'MARL_GAMMA', 0.98),
            'epsilon_start': getattr(cfg, 'MARL_EPSILON_START', 0.95),
            'epsilon_min': getattr(cfg, 'MARL_EPSILON_MIN', 0.02),
            'epsilon_decay': getattr(cfg, 'MARL_EPSILON_DECAY', 0.9995),
            'learning_rate': getattr(cfg, 'MARL_LEARNING_RATE', 3e-4),
            'batch_size': getattr(cfg, 'MARL_BATCH_SIZE', 128),
            'target_update_freq': getattr(cfg, 'MARL_TARGET_UPDATE_FREQ', 100),
            'broadcast_interval': getattr(cfg, 'MARL_BROADCAST_INTERVAL', 5),
        }

    def _initialize_marl_system(self, simulator):
        """Initialize toàn bộ MARL system một lần duy nhất"""
        n_agents = len(simulator.drones)
        
        if n_agents == 0:
            raise ValueError("[ERROR] Cannot initialize MARL system with 0 drones")
        
        print(f"[INFO] Initializing MARL system for {n_agents} agents...")
        
        augmented_state_size = self.config['state_size'] + 6  # local + neighbor info

        # Centralized trainer
        try:
            trainer = CentralizedTrainer(
                n_agents=n_agents,
                state_size=augmented_state_size,
                action_size=n_agents,
                config=self.config
            )
            print(f"[INFO] Centralized trainer initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize trainer: {e}")
            raise

        # Create agents
        agents = {}
        for i, drone in enumerate(simulator.drones):
            try:
                if drone is None:
                    print(f"[WARN] Skipping None drone at index {i}")
                    continue
                    
                agent = CommunicatingAgent(drone, simulator, trainer, self.config)
                agents[drone.identifier] = agent
                print(f"[INFO] Agent {drone.identifier} initialized")
            except Exception as e:
                print(f"[ERROR] Failed to initialize agent for drone {drone.identifier}: {e}")
                raise

        if len(agents) == 0:
            raise ValueError("[ERROR] No agents were successfully initialized")

        # Relay coordinator
        coordinator = RelayCoordinator()
        print(f"[INFO] Relay coordinator initialized")

        return {
            'trainer': trainer,
            'agents': agents,
            'coordinator': coordinator,
            'step_count': 0
        }

    # -------------------- ROUTING INTERFACE --------------------

    def relay_selection(self, packet, optimize_residual_energy=False):
        """
        Chọn relay cho packet (override từ BASE_routing)
        Flow:
        1. Broadcast intention
        2. Receive messages từ neighbors
        3. Select action với augmented state
        4. Coordinate để tránh xung đột
        """
        try:
            # Validate system
            if self.agent is None:
                print(f"[ERROR] Agent is None for drone {self.drone.identifier}")
                return None
                
            if not hasattr(self, 'marl_system') or self.marl_system is None:
                print(f"[ERROR] MARL system not initialized")
                return None

            # Step 1: Broadcast intention
            try:
                intention = self.agent.broadcast_intention()
            except Exception as e:
                print(f"[WARN] Broadcast failed: {e}")
                intention = None

            # Step 2: Collect messages từ neighbors
            if intention is not None:
                try:
                    neighbor_messages = self._collect_neighbor_messages(intention)
                    self.agent.receive_messages(neighbor_messages)
                except Exception as e:
                    print(f"[WARN] Message collection failed: {e}")

            # Step 3: Get valid neighbors
            try:
                valid_mask = self._get_valid_neighbors_mask()
            except Exception as e:
                print(f"[ERROR] Failed to get valid neighbors: {e}")
                return None

            if not np.any(valid_mask):
                return None

            # Step 4: Select action
            try:
                action = self.agent.select_action(valid_mask)
            except Exception as e:
                print(f"[ERROR] Action selection failed: {e}")
                # Fallback to random
                valid_indices = np.where(valid_mask)[0]
                action = random.choice(valid_indices) if len(valid_indices) > 0 else None

            if action is None:
                return None

            # Step 5: Coordinate với other agents
            try:
                current_step = getattr(self.simulator, 'cur_step', 0)
                action = self._coordinate_relay_selection(action, current_step)
            except Exception as e:
                print(f"[WARN] Coordination failed, using uncoordinated action: {e}")

            # Save for feedback
            self.last_state = self.agent.last_augmented_state
            self.last_action = action

            return action

        except Exception as e:
            print(f"[ERROR] MultiAgent relay_selection critical failure: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency fallback to random selection
            try:
                valid_mask = self._get_valid_neighbors_mask()
                valid_indices = np.where(valid_mask)[0]
                return random.choice(valid_indices) if len(valid_indices) > 0 else None
            except:
                return None

    def _collect_neighbor_messages(self, own_intention):
        """Thu thập messages từ neighbors trong comm range"""
        messages = []
        
        try:
            comm_range = getattr(self.simulator, 'communication_range', 200)

            if not hasattr(self, 'marl_system') or 'agents' not in self.marl_system:
                return messages

            for other_id, other_agent in self.marl_system['agents'].items():
                try:
                    if other_id == self.agent.agent_id:
                        continue

                    if other_agent is None or other_agent.drone is None:
                        continue

                    # Check distance
                    distance = util.euclidean_distance(
                        self.drone.coords,
                        other_agent.drone.coords
                    )

                    if distance <= comm_range:
                        # Lấy intention của neighbor
                        neighbor_intention = other_agent.broadcast_intention()
                        if neighbor_intention is not None:
                            messages.append(neighbor_intention)
                            
                except Exception as e:
                    # Skip agent nếu có lỗi
                    continue

        except Exception as e:
            print(f"[WARN] Failed to collect neighbor messages: {e}")

        return messages

    def _get_valid_neighbors_mask(self):
        """
        Tạo binary mask cho valid neighbors
        """
        try:
            mask = np.zeros(self.simulator.n_drones, dtype=bool)

            if not hasattr(self.drone, 'neighbor_table') or self.drone.neighbor_table is None:
                return mask

            current_step = getattr(self.simulator, 'cur_step', 0)

            for i in range(min(len(self.drone.neighbor_table), self.simulator.n_drones)):
                try:
                    arrival_time = self.drone.neighbor_table[i, 6]

                    # Check if neighbor is valid
                    is_valid = (
                        arrival_time > 0 and
                        (current_step - arrival_time) < self.simulator.ExpireTime and
                        i < len(self.simulator.drones)
                    )

                    if is_valid:
                        neighbor = self.simulator.drones[i]
                        # Check energy
                        if neighbor is not None and hasattr(neighbor, 'residual_energy') and neighbor.residual_energy > 0:
                            mask[i] = True
                except Exception as e:
                    # Skip neighbor nếu có lỗi
                    continue

            return mask
            
        except Exception as e:
            print(f"[ERROR] Failed to create valid neighbors mask: {e}")
            # Return empty mask
            return np.zeros(self.simulator.n_drones, dtype=bool)

    def _coordinate_relay_selection(self, preferred_action, current_step):
        """
        Coordinate relay selection với other agents để tránh xung đột
        """
        try:
            coordinator = self.marl_system['coordinator']

            # Lấy priority (Q-value của action)
            state_tensor = torch.tensor(self.agent.last_augmented_state, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.marl_system['trainer'].shared_model(state_tensor)
                priority = float(q_values[preferred_action].item())

            # Register selection
            coordinator.register_selection(
                self.agent.agent_id,
                preferred_action,
                priority,
                current_step
            )

            # Resolve conflicts (tất cả agents register xong)
            # Note: Trong thực tế, cần sync point hoặc async resolution
            # Ở đây đơn giản hóa: resolve sau mỗi N registrations
            if len(coordinator.pending_selections) >= len(self.marl_system['agents']) * 0.5:
                approved = coordinator.resolve_conflicts(current_step)

                if self.agent.agent_id in approved:
                    approved_action = approved[self.agent.agent_id]
                    if approved_action is not None:
                        return approved_action

            # Nếu không có conflict hoặc chưa resolve, giữ nguyên
            return preferred_action

        except Exception as e:
            print(f"[WARN] Coordination failed: {e}")
            return preferred_action

    # -------------------- FEEDBACK & LEARNING --------------------

    def feedback(self, outcome, id_j, q_best):
        """
        Nhận feedback sau khi gửi packet (override từ BASE_routing)
        outcome: 1 (success), 0 (forward), -1 (failure)
        """
        try:
            if self.last_state is None or self.last_action is None:
                return

            # Compute reward
            reward = self.agent.compute_reward(outcome)

            # Get next state
            next_state = self.agent.get_augmented_state()
            done = (outcome == 1 or outcome == -1)

            # Update agent (send to trainer)
            self.agent.update(self.last_action, reward, next_state, done)

            # Centralized training
            trainer = self.marl_system['trainer']
            if (len(trainer.global_memory) + len(trainer.global_priority_memory)) >= trainer.batch_size:
                loss = trainer.train_step()
                if loss is not None and self.simulator.cur_step % 100 == 0:
                    print(f"[TRAIN] Step {self.simulator.cur_step}, Loss: {loss:.4f}")

            # Release relay
            if outcome != 0:  # Gói không còn forward nữa
                self.marl_system['coordinator'].release_relay(self.last_action)

            # Reset cho episode tiếp theo
            if done:
                self.last_state = None
                self.last_action = None

        except Exception as e:
            print(f"[ERROR] Feedback processing failed: {e}")

    # -------------------- STATISTICS & MONITORING --------------------

    def get_performance_metrics(self):
        """Lấy metrics của toàn bộ MARL system"""
        try:
            agent_stats = self.agent.get_stats()
            trainer_stats = self.marl_system['trainer'].get_training_stats()

            # Aggregate stats từ tất cả agents
            all_agents = self.marl_system['agents'].values()
            system_success_rate = np.mean([a.success_count / max(a.total_attempts, 1) for a in all_agents])
            system_avg_reward = np.mean([np.mean(a.reward_history) if a.reward_history else 0.0 for a in all_agents])

            return {
                'agent': agent_stats,
                'trainer': trainer_stats,
                'system': {
                    'total_agents': len(self.marl_system['agents']),
                    'system_success_rate': float(system_success_rate),
                    'system_avg_reward': float(system_avg_reward),
                }
            }

        except Exception as e:
            print(f"[ERROR] Get metrics failed: {e}")
            return {}

    def print_system_stats(self):
        """In thống kê của toàn hệ thống (debug)"""
        metrics = self.get_performance_metrics()

        print("\n" + "="*60)
        print("MULTI-AGENT EDQN SYSTEM STATISTICS")
        print("="*60)

        if 'system' in metrics:
            print(f"Total Agents: {metrics['system']['total_agents']}")
            print(f"System Success Rate: {metrics['system']['system_success_rate']:.2%}")
            print(f"System Avg Reward: {metrics['system']['system_avg_reward']:.3f}")

        if 'trainer' in metrics:
            print(f"\nTraining Steps: {metrics['trainer']['training_steps']}")
            print(f"Avg Loss: {metrics['trainer']['avg_loss']:.4f}")
            print(f"Memory: {metrics['trainer']['memory_size']}")
            print(f"Priority Memory: {metrics['trainer']['priority_memory_size']}")

        if 'agent' in metrics:
            print(f"\nCurrent Agent {metrics['agent']['agent_id']}:")
            print(f"  Success Rate: {metrics['agent']['success_rate']:.2%}")
            print(f"  Epsilon: {metrics['agent']['epsilon']:.3f}")
            print(f"  Avg Neighbors: {metrics['agent']['avg_neighbors']:.1f}")
            print(f"  Conservative Mode: {metrics['agent']['conservative_mode']}")

        print("="*60 + "\n")


# ====================== UTILITY FUNCTIONS ======================

def save_marl_model(marl_system, filepath):
    """Lưu model của MARL system"""
    try:
        torch.save({
            'shared_model': marl_system['trainer'].shared_model.state_dict(),
            'shared_target': marl_system['trainer'].shared_target.state_dict(),
            'message_encoder': marl_system['trainer'].message_encoder.state_dict(),
            'message_decoder': marl_system['trainer'].message_decoder.state_dict(),
            'attention_aggregator': marl_system['trainer'].attention_aggregator.state_dict(),
            'training_steps': marl_system['trainer'].training_steps,
        }, filepath)
        print(f"[INFO] MARL model saved to {filepath}")
    except Exception as e:
        print(f"[ERROR] Save model failed: {e}")


def load_marl_model(marl_system, filepath):
    """Load model cho MARL system"""
    try:
        checkpoint = torch.load(filepath)
        marl_system['trainer'].shared_model.load_state_dict(checkpoint['shared_model'])
        marl_system['trainer'].shared_target.load_state_dict(checkpoint['shared_target'])
        marl_system['trainer'].message_encoder.load_state_dict(checkpoint['message_encoder'])
        marl_system['trainer'].message_decoder.load_state_dict(checkpoint['message_decoder'])
        marl_system['trainer'].attention_aggregator.load_state_dict(checkpoint['attention_aggregator'])
        marl_system['trainer'].training_steps = checkpoint['training_steps']
        print(f"[INFO] MARL model loaded from {filepath}")
    except Exception as e:
        print(f"[ERROR] Load model failed: {e}")


# ====================== NOTES ======================

"""
USAGE EXAMPLE:

# In routing_registry.py, add:
from src.routing_algorithms.multi_agent_edqn_routing import MultiAgentEDQN_Routing

# In config.py, add parameters:
MARL_STATE_SIZE = 10
MARL_MEMORY_SIZE = 10000
MARL_PRIORITY_SIZE = 3000
MARL_GAMMA = 0.98
MARL_EPSILON_START = 0.95
MARL_EPSILON_MIN = 0.02
MARL_EPSILON_DECAY = 0.9995
MARL_LEARNING_RATE = 3e-4
MARL_BATCH_SIZE = 128
MARL_TARGET_UPDATE_FREQ = 100
MARL_BROADCAST_INTERVAL = 5

# Initialize:
drone_routing = MultiAgentEDQN_Routing(drone, simulator)

# During simulation:
relay_id = drone_routing.relay_selection(packet)
drone_routing.feedback(outcome, relay_id, q_best)

# Monitoring:
metrics = drone_routing.get_performance_metrics()
drone_routing.print_system_stats()

# Save/Load:
save_marl_model(simulator._marl_system, 'marl_model.pth')
load_marl_model(simulator._marl_system, 'marl_model.pth')
"""
