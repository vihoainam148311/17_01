import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
import math

# Experience tuple for cleaner code
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'valid_actions'])


class EnhancedDQN(nn.Module):
    """
    Enhanced DQN Network with improved architecture for UAV routing
    """

    def __init__(self, state_dim, hidden_dim=256, dropout=0.2):
        super(EnhancedDQN, self).__init__()

        # State processing layers
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Neighbor feature processing
        self.neighbor_processor = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),  # 6 neighbor features
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Combined processing and Q-value estimation
        combined_dim = hidden_dim + hidden_dim // 2
        self.q_network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single Q-value output
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)

    def forward(self, state, neighbor_features):
        """
        Forward pass for Q-value estimation

        Args:
            state: Tensor [batch_size, state_dim] - drone state
            neighbor_features: Tensor [batch_size, max_neighbors, 6] - neighbor features

        Returns:
            q_values: Tensor [batch_size, max_neighbors] - Q-values for each neighbor
        """
        batch_size, max_neighbors, _ = neighbor_features.shape

        # Process drone state
        state_features = self.state_processor(state)  # [batch_size, hidden_dim]

        # Expand state features to match neighbor dimension
        state_expanded = state_features.unsqueeze(1).expand(-1, max_neighbors,
                                                            -1)  # [batch_size, max_neighbors, hidden_dim]

        # Process neighbor features
        neighbor_features_flat = neighbor_features.view(-1, 6)  # [batch_size * max_neighbors, 6]
        neighbor_processed = self.neighbor_processor(
            neighbor_features_flat)  # [batch_size * max_neighbors, hidden_dim//2]
        neighbor_processed = neighbor_processed.view(batch_size, max_neighbors,
                                                     -1)  # [batch_size, max_neighbors, hidden_dim//2]

        # Combine state and neighbor features
        combined = torch.cat([state_expanded, neighbor_processed], dim=2)  # [batch_size, max_neighbors, combined_dim]

        # Calculate Q-values
        combined_flat = combined.view(-1, combined.shape[2])  # [batch_size * max_neighbors, combined_dim]
        q_values_flat = self.q_network(combined_flat)  # [batch_size * max_neighbors, 1]
        q_values = q_values_flat.view(batch_size, max_neighbors)  # [batch_size, max_neighbors]

        return q_values


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer for better learning
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, experience):
        """Add new experience with maximum priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch with prioritized sampling"""
        if len(self.buffer) < batch_size:
            return None, None, None

        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return experiences, indices, weights

    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-6

    def __len__(self):
        return len(self.buffer)


class EnhancedDQNRouting(BASE_routing):
    """
    Enhanced DQN Routing Algorithm for UAV Networks
    """

    def __init__(self, drone, simulator):
        super().__init__(drone, simulator)

        # Network parameters
        self.state_dim = 8  # Enhanced state representation
        self.hidden_dim = 256
        self.max_neighbors = 20  # Maximum neighbors to consider

        # DQN Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network = EnhancedDQN(self.state_dim, self.hidden_dim).to(self.device)
        self.target_network = EnhancedDQN(self.state_dim, self.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=0.0003, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

        # Experience replay
        self.memory = PrioritizedReplayBuffer(capacity=10000)

        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005  # Soft update parameter

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9995

        # Training control
        self.learning_starts = 1000
        self.train_frequency = 4
        self.target_update_frequency = 100
        self.step_count = 0

        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        self.q_value_history = deque(maxlen=1000)

        # Conservative mode
        self.conservative_threshold = 0.7
        self.conservative_mode = False
        self.performance_window = 100

        print(f"[EDQN] Initialized Enhanced DQN on {self.device}")

    def get_enhanced_state(self, packet=None):
        """
        Get enhanced state representation for the drone

        Returns:
            state: numpy array of normalized state features
        """
        # Basic drone state
        norm_x = self.drone.coords[0] / self.simulator.env_width
        norm_y = self.drone.coords[1] / self.simulator.env_height
        energy_ratio = self.drone.residual_energy / self.drone.initial_energy

        # Buffer state
        buffer_ratio = self.drone.buffer_length() / self.drone.max_buffer_size

        # Network connectivity
        n_neighbors = len(self.opt_neighbors)
        connectivity_ratio = n_neighbors / max(self.simulator.n_drones - 1, 1)

        # Destination information (if packet available)
        if packet and hasattr(packet, 'dst_depot'):
            depot_coords = self.simulator.depot.coords
            dst_distance = util.euclidean_distance(self.drone.coords, depot_coords)
            norm_dst_distance = dst_distance / self.simulator.max_distance
        else:
            norm_dst_distance = 0.5  # Default neutral value

        # Network quality metrics
        avg_link_quality = np.mean([self.drone.neighbor_table[neighbor.identifier, 12]
                                    for neighbor in self.opt_neighbors]) if self.opt_neighbors else 0.0

        # Movement state
        norm_speed = self.drone.speed / getattr(self.simulator, 'max_speed', 20.0)

        state = np.array([
            norm_x, norm_y,  # Position [0-1]
            energy_ratio,  # Energy [0-1]
            buffer_ratio,  # Buffer utilization [0-1]
            connectivity_ratio,  # Network connectivity [0-1]
            norm_dst_distance,  # Distance to destination [0-1]
            avg_link_quality,  # Average link quality [0-1]
            norm_speed  # Speed [0-1]
        ], dtype=np.float32)

        return state

    def get_neighbor_features(self, neighbors):
        """
        Extract features for each neighbor

        Args:
            neighbors: List of neighbor drones

        Returns:
            features: numpy array [n_neighbors, 6] of neighbor features
        """
        if not neighbors:
            return np.zeros((1, 6), dtype=np.float32)  # Dummy neighbor

        features = []
        depot_coords = self.simulator.depot.coords

        for neighbor in neighbors:
            neighbor_id = neighbor.identifier

            # Distance to neighbor
            distance = util.euclidean_distance(self.drone.coords, neighbor.coords)
            norm_distance = distance / self.drone.communication_range

            # Neighbor's distance to depot
            neighbor_depot_distance = util.euclidean_distance(neighbor.coords, depot_coords)
            norm_neighbor_depot_dist = neighbor_depot_distance / self.simulator.max_distance

            # Neighbor's energy
            neighbor_energy = neighbor.residual_energy / neighbor.initial_energy

            # Link quality
            link_quality = self.drone.neighbor_table[neighbor_id, 12]

            # Neighbor's buffer state
            neighbor_buffer = neighbor.buffer_length() / neighbor.max_buffer_size

            # Expected delay
            expected_delay = self.drone.neighbor_table[neighbor_id, 8]
            norm_delay = min(expected_delay / 100.0, 1.0)  # Normalize delay

            neighbor_feature = np.array([
                norm_distance,  # Distance to neighbor [0-1]
                norm_neighbor_depot_dist,  # Neighbor's distance to depot [0-1]
                neighbor_energy,  # Neighbor's energy [0-1]
                link_quality,  # Link quality [0-1]
                neighbor_buffer,  # Neighbor's buffer utilization [0-1]
                1.0 - norm_delay  # Inverse delay (higher is better) [0-1]
            ], dtype=np.float32)

            features.append(neighbor_feature)

        return np.array(features)

    def select_action(self, state, neighbor_features, valid_neighbors, training=True):
        """
        Select action using epsilon-greedy with enhanced exploration

        Args:
            state: Current state
            neighbor_features: Features of available neighbors
            valid_neighbors: List of valid neighbor drones
            training: Whether in training mode

        Returns:
            action_idx: Index of selected neighbor, or None if no valid neighbors
        """
        if not valid_neighbors:
            return None

        # Conservative mode: use traditional metrics when performance is low
        if self.conservative_mode and training:
            return self._conservative_action_selection(valid_neighbors)

        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(len(valid_neighbors))

        # Prepare inputs for network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, state_dim]

        # Pad neighbor features to max_neighbors
        padded_features = np.zeros((self.max_neighbors, 6), dtype=np.float32)
        n_neighbors = min(len(neighbor_features), self.max_neighbors)
        padded_features[:n_neighbors] = neighbor_features[:n_neighbors]

        neighbor_tensor = torch.FloatTensor(padded_features).unsqueeze(0).to(self.device)  # [1, max_neighbors, 6]

        # Get Q-values
        with torch.no_grad():
            q_values = self.main_network(state_tensor, neighbor_tensor)  # [1, max_neighbors]
            q_values = q_values.squeeze(0)[:len(valid_neighbors)]  # [n_valid_neighbors]

        # Select best action
        action_idx = torch.argmax(q_values).item()

        # Track Q-values for monitoring
        if training:
            avg_q_value = q_values.mean().item()
            self.q_value_history.append(avg_q_value)

        return action_idx

    def _conservative_action_selection(self, valid_neighbors):
        """
        Conservative action selection using traditional routing metrics
        """
        best_score = -float('inf')
        best_idx = 0

        depot_coords = self.simulator.depot.coords

        for idx, neighbor in enumerate(valid_neighbors):
            neighbor_id = neighbor.identifier

            # Distance to depot (progress towards destination)
            neighbor_depot_distance = util.euclidean_distance(neighbor.coords, depot_coords)
            my_depot_distance = util.euclidean_distance(self.drone.coords, depot_coords)
            progress_score = max(0, my_depot_distance - neighbor_depot_distance)

            # Energy consideration
            energy_score = neighbor.residual_energy / neighbor.initial_energy

            # Link quality
            link_quality = self.drone.neighbor_table[neighbor_id, 12]

            # Buffer state (prefer neighbors with more buffer space)
            buffer_score = 1.0 - (neighbor.buffer_length() / neighbor.max_buffer_size)

            # Combined score
            total_score = (0.4 * progress_score / self.simulator.max_distance +
                           0.2 * energy_score +
                           0.2 * link_quality +
                           0.2 * buffer_score)

            if total_score > best_score:
                best_score = total_score
                best_idx = idx

        return best_idx

    def store_experience(self, state, action, reward, next_state, done, valid_actions):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done, valid_actions)
        self.memory.add(experience)

    def compute_reward(self, outcome, neighbor, packet, delivery_time=None):
        """
        Enhanced reward computation for UAV routing

        Args:
            outcome: 0=forwarded, 1=delivered, -1=dropped/failed
            neighbor: Selected neighbor drone
            packet: The routed packet
            delivery_time: Time taken for delivery (if delivered)

        Returns:
            reward: Computed reward value
        """
        if outcome == 1:  # Successfully delivered
            base_reward = 10.0

            # Time bonus (faster delivery is better)
            if delivery_time is not None:
                time_bonus = max(0, 5.0 - delivery_time / 100.0)
                base_reward += time_bonus

            return base_reward

        elif outcome == 0:  # Forwarded to neighbor
            # Progress reward
            depot_coords = self.simulator.depot.coords
            my_distance = util.euclidean_distance(self.drone.coords, depot_coords)
            neighbor_distance = util.euclidean_distance(neighbor.coords, depot_coords)
            progress = max(0, my_distance - neighbor_distance) / self.simulator.max_distance

            # Energy efficiency
            energy_efficiency = neighbor.residual_energy / neighbor.initial_energy

            # Link quality
            link_quality = self.drone.neighbor_table[neighbor.identifier, 12]

            # Network load balancing
            neighbor_load = neighbor.buffer_length() / neighbor.max_buffer_size
            load_penalty = neighbor_load * 0.5

            reward = (2.0 * progress +  # Progress towards destination
                      1.0 * energy_efficiency +  # Energy consideration
                      1.0 * link_quality +  # Communication quality
                      0.5 - load_penalty)  # Load balancing

            return reward

        else:  # Failed (dropped, lost, etc.)
            return -5.0

    def train(self):
        """Train the DQN network"""
        if len(self.memory) < self.learning_starts:
            return

        if self.step_count % self.train_frequency != 0:
            return

        # Sample experiences
        experiences, indices, weights = self.memory.sample(self.batch_size)
        if experiences is None:
            return

        # Prepare batch data
        states = []
        neighbor_features_batch = []
        actions = []
        rewards = []
        next_states = []
        next_neighbor_features_batch = []
        dones = []

        for exp in experiences:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.done)

            # Handle neighbor features (simplified for training)
            if len(exp.valid_actions) > 0:
                # Use dummy neighbor features for training
                dummy_features = np.random.random((self.max_neighbors, 6)).astype(np.float32)
                neighbor_features_batch.append(dummy_features)

                if exp.next_state is not None:
                    next_states.append(exp.next_state)
                    next_neighbor_features_batch.append(dummy_features)
                else:
                    next_states.append(np.zeros_like(exp.state))
                    next_neighbor_features_batch.append(dummy_features)
            else:
                # No valid neighbors
                zero_features = np.zeros((self.max_neighbors, 6), dtype=np.float32)
                neighbor_features_batch.append(zero_features)
                next_states.append(np.zeros_like(exp.state))
                next_neighbor_features_batch.append(zero_features)

        # Convert to tensors
        state_batch = torch.FloatTensor(states).to(self.device)
        neighbor_batch = torch.FloatTensor(neighbor_features_batch).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        next_neighbor_batch = torch.FloatTensor(next_neighbor_features_batch).to(self.device)
        done_batch = torch.BoolTensor(dones).to(self.device)
        weight_batch = torch.FloatTensor(weights).to(self.device)

        # Current Q-values
        current_q_values = self.main_network(state_batch, neighbor_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Next Q-values (Double DQN)
        with torch.no_grad():
            next_q_values_main = self.main_network(next_state_batch, next_neighbor_batch)
            next_actions = next_q_values_main.argmax(1)
            next_q_values_target = self.target_network(next_state_batch, next_neighbor_batch)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)

        # Compute loss with importance sampling weights
        td_errors = target_q_values - current_q_values
        loss = (weight_batch * td_errors.pow(2)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Track loss
        self.loss_history.append(loss.item())

        # Soft update target network
        if self.step_count % self.target_update_frequency == 0:
            self._soft_update_target_network()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _soft_update_target_network(self):
        """Soft update of target network"""
        for target_param, main_param in zip(self.target_network.parameters(),
                                            self.main_network.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

    def update_performance_tracking(self, outcome):
        """Update performance tracking and conservative mode"""
        self.performance_history.append(1 if outcome == 1 else 0)

        if len(self.performance_history) >= self.performance_window:
            recent_performance = np.mean(list(self.performance_history)[-self.performance_window:])
            self.conservative_mode = recent_performance < self.conservative_threshold

    def relay_selection(self, opt_neighbors, packet):
        """
        Main relay selection method

        Args:
            opt_neighbors: List of available neighbor drones
            packet: Packet to be routed

        Returns:
            selected_neighbor: Chosen neighbor drone or None
        """
        if not opt_neighbors:
            return None

        # Get current state
        current_state = self.get_enhanced_state(packet[0] if isinstance(packet, tuple) else packet)

        # Get neighbor features
        neighbor_features = self.get_neighbor_features(opt_neighbors)

        # Select action
        action_idx = self.select_action(current_state, neighbor_features, opt_neighbors, training=True)

        if action_idx is None or action_idx >= len(opt_neighbors):
            return None

        selected_neighbor = opt_neighbors[action_idx]

        # Store current state for experience replay (will be completed in feedback)
        self.current_state = current_state
        self.current_action = action_idx
        self.current_neighbors = opt_neighbors.copy()
        self.current_packet = packet

        self.step_count += 1

        # Train the network
        self.train()

        return selected_neighbor

    def feedback(self, outcome, neighbor_id, best_q_value):
        """
        Process feedback and store experience

        Args:
            outcome: Routing outcome (0=forwarded, 1=delivered, -1=failed)
            neighbor_id: ID of neighbor that was selected
            best_q_value: Best Q-value (legacy parameter, not used)
        """
        if not hasattr(self, 'current_state'):
            return

        # Find the neighbor object
        selected_neighbor = None
        for neighbor in self.current_neighbors:
            if neighbor.identifier == neighbor_id:
                selected_neighbor = neighbor
                break

        if selected_neighbor is None:
            return

        # Compute reward
        reward = self.compute_reward(outcome, selected_neighbor, self.current_packet)

        # Get next state
        next_state = self.get_enhanced_state()

        # Store experience
        done = (outcome == 1 or outcome == -1)  # Terminal if delivered or failed
        valid_actions = list(range(len(self.current_neighbors)))

        self.store_experience(
            self.current_state,
            self.current_action,
            reward,
            next_state if not done else None,
            done,
            valid_actions
        )

        # Update performance tracking
        self.update_performance_tracking(outcome)

        # Clean up
        delattr(self, 'current_state')
        delattr(self, 'current_action')
        delattr(self, 'current_neighbors')
        delattr(self, 'current_packet')

    def get_performance_metrics(self):
        """Get performance metrics for monitoring"""
        recent_performance = (np.mean(list(self.performance_history)[-100:])
                              if len(self.performance_history) >= 100 else 0.0)

        return {
            'epsilon': self.epsilon,
            'conservative_mode': self.conservative_mode,
            'recent_performance': recent_performance,
            'avg_loss': np.mean(list(self.loss_history)[-100:]) if self.loss_history else 0.0,
            'avg_q_value': np.mean(list(self.q_value_history)[-100:]) if self.q_value_history else 0.0,
            'memory_size': len(self.memory),
            'total_steps': self.step_count
        }

    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'performance_history': list(self.performance_history),
        }, filepath)

        print(f"[EDQN] Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.performance_history = deque(checkpoint['performance_history'], maxlen=1000)

        print(f"[EDQN] Model loaded from {filepath}")