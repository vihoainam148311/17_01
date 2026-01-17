import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util

# Simple experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class SimpleDQN(nn.Module):
    """
    Simple DQN Network for UAV routing with improved architecture
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(SimpleDQN, self).__init__()

        # Use LayerNorm instead of BatchNorm to avoid batch size issues
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, state):
        return self.network(state)


class DQN_Routing(BASE_routing):
    """
    Fixed Simple Pure DQN Routing Algorithm for UAV Networks
    """

    def __init__(self, drone, simulator):
        super().__init__(drone, simulator)

        # Enhanced state: more features for better decision making
        self.state_dim = 9
        self.max_actions = 15  # Increased to handle more neighbors

        # DQN Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network = SimpleDQN(self.state_dim, self.max_actions).to(self.device)
        self.target_network = SimpleDQN(self.state_dim, self.max_actions).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())

        # Optimizer with better parameters
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=0.0005, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.95)

        # Enhanced experience replay
        self.memory = deque(maxlen=8000)  # Increased memory

        # Training parameters
        self.batch_size = 64  # Increased batch size
        self.gamma = 0.99  # Higher discount factor

        # Fixed exploration parameters
        self.epsilon = 0.9  # Lower starting epsilon
        self.epsilon_min = 0.05  # Higher minimum epsilon
        self.epsilon_decay = 0.9990  # Faster decay

        # Training control
        self.learning_starts = 1500  # More initial experiences
        self.target_update_frequency = 200  # More frequent updates
        self.step_count = 0

        # Performance tracking
        self.routing_hole_count = 0
        self.successful_forwards = 0
        self.recent_performance = deque(maxlen=500)

        # Current experience tracking
        self.current_state = None
        self.current_action = None
        self.current_neighbors = []
        self.current_packet = None

        # Adaptive exploration
        self.performance_window = 200
        self.poor_performance_threshold = 0.3

        print(f"[Fixed Simple DQN] Initialized on {self.device}")
        print(f"[Fixed Simple DQN] Enhanced features: {self.state_dim}, Max actions: {self.max_actions}")

    def get_enhanced_state(self, packet=None):
        """
        Enhanced state representation with more critical features
        """
        # Normalize coordinates
        norm_x = self.drone.coords[0] / self.simulator.env_width
        norm_y = self.drone.coords[1] / self.simulator.env_height

        # Energy ratio
        energy_ratio = self.drone.residual_energy / self.drone.initial_energy

        # Distance to depot
        depot_coords = self.simulator.depot.coords
        depot_distance = util.euclidean_distance(self.drone.coords, depot_coords)
        max_distance = np.sqrt(self.simulator.env_width ** 2 + self.simulator.env_height ** 2)
        norm_depot_distance = depot_distance / max_distance

        # Number of neighbors (connectivity)
        n_neighbors = len(self.opt_neighbors)
        norm_neighbors = min(n_neighbors / 15.0, 1.0)  # Normalize to max 15 neighbors

        # Buffer utilization
        max_buffer_size = getattr(self.drone, 'max_buffer_size',
                                  getattr(self.simulator, 'DRONE_MAX_BUFFER_SIZE', 100))
        buffer_ratio = self.drone.buffer_length() / max_buffer_size

        # NEW: Packet urgency (if packet available)
        packet_urgency = 0.5  # Default neutral value
        if packet is not None:
            # Extract packet from tuple if needed
            actual_packet = packet[0] if isinstance(packet, tuple) else packet
            if hasattr(actual_packet, 'time_step_creation') and hasattr(actual_packet, 'ttl'):
                packet_age = self.simulator.cur_step - actual_packet.time_step_creation
                packet_urgency = min(packet_age / actual_packet.ttl, 1.0)
            elif hasattr(actual_packet, 'creation_time'):
                packet_age = self.simulator.cur_step - actual_packet.creation_time
                max_age = getattr(self.simulator, 'PACKETS_MAX_TTL', 150)
                packet_urgency = min(packet_age / max_age, 1.0)

        # NEW: Average neighbor quality
        avg_neighbor_quality = 0.0
        if self.opt_neighbors:
            qualities = []
            for neighbor in self.opt_neighbors:
                # Calculate neighbor quality based on multiple factors
                neighbor_depot_dist = util.euclidean_distance(neighbor.coords, depot_coords)
                neighbor_energy = neighbor.residual_energy / neighbor.initial_energy

                # Distance progress score (closer to depot is better)
                progress_score = max(0, (depot_distance - neighbor_depot_dist) / depot_distance)

                # Combined quality score
                quality = 0.5 * progress_score + 0.3 * neighbor_energy + 0.2 * (1 - neighbor_depot_dist / max_distance)
                qualities.append(quality)

            avg_neighbor_quality = np.mean(qualities)

        # NEW: Network stability (based on neighbor changes)
        neighbor_stability = self.drone.discount_factor if hasattr(self.drone, 'discount_factor') else 0.5

        # NEW: Current mission progress (how many packets delivered recently)
        mission_progress = min(self.successful_forwards / 100.0, 1.0)

        state = np.array([
            norm_x,  # X position [0-1]
            norm_y,  # Y position [0-1]
            energy_ratio,  # Energy level [0-1]
            norm_depot_distance,  # Distance to depot [0-1]
            norm_neighbors,  # Number of neighbors [0-1]
            buffer_ratio,  # Buffer utilization [0-1]
            packet_urgency,  # Packet urgency [0-1] - NEW
            avg_neighbor_quality,  # Average neighbor quality [0-1] - NEW
            neighbor_stability  # Network stability [0-1] - NEW
        ], dtype=np.float32)

        return state

    def select_action(self, state, valid_neighbors):
        """
        Improved epsilon-greedy action selection with adaptive exploration
        """
        if not valid_neighbors:
            return None

        n_neighbors = len(valid_neighbors)

        # Adaptive epsilon based on recent performance
        current_epsilon = self.epsilon
        if len(self.recent_performance) >= self.performance_window:
            recent_success_rate = np.mean(self.recent_performance)
            if recent_success_rate < self.poor_performance_threshold:
                # Increase exploration when performance is poor
                current_epsilon = min(0.4, self.epsilon * 1.5)

        # Epsilon-greedy exploration
        if np.random.random() < current_epsilon:
            return np.random.randint(n_neighbors)

        # Get Q-values from network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.main_network(state_tensor)
            # Only consider valid actions (first n_neighbors outputs)
            valid_q_values = q_values[0][:n_neighbors]

            # Add small random noise to break ties
            noise = torch.randn_like(valid_q_values) * 0.01
            action = torch.argmax(valid_q_values + noise).item()

        return action

    def evaluate_neighbor_quality(self, neighbor):
        """
        Evaluate the quality of a neighbor for routing decisions
        """
        depot_coords = self.simulator.depot.coords
        my_distance = util.euclidean_distance(self.drone.coords, depot_coords)
        neighbor_distance = util.euclidean_distance(neighbor.coords, depot_coords)

        # Progress towards depot
        progress_score = max(0, (my_distance - neighbor_distance) / my_distance) if my_distance > 0 else 0

        # Neighbor's energy level
        energy_score = neighbor.residual_energy / neighbor.initial_energy

        # Link quality (if available)
        link_quality = 0.8  # Default good quality
        if hasattr(self.drone, 'neighbor_table') and neighbor.identifier < self.drone.neighbor_table.shape[0]:
            link_quality = self.drone.neighbor_table[neighbor.identifier, 12]

        # Neighbor's buffer availability
        neighbor_max_buffer = getattr(neighbor, 'max_buffer_size',
                                      getattr(self.simulator, 'DRONE_MAX_BUFFER_SIZE', 100))
        buffer_availability = 1.0 - (neighbor.buffer_length() / neighbor_max_buffer)

        # Combined quality score
        quality = (0.4 * progress_score +
                   0.2 * energy_score +
                   0.2 * link_quality +
                   0.2 * buffer_availability)

        return quality

    def compute_enhanced_reward(self, outcome, selected_neighbor, packet=None):
        """
        Enhanced reward function with stronger penalties for routing holes

        Args:
            outcome: 0=forwarded, 1=delivered, -1=failed/routing_hole
            selected_neighbor: The neighbor that was selected
            packet: The packet being routed

        Returns:
            reward: Enhanced reward value
        """
        if outcome == 1:  # Delivered successfully
            base_reward = 15.0

            # Time bonus for fast delivery
            if packet and hasattr(packet, 'creation_time'):
                delivery_time = self.simulator.cur_step - packet.creation_time
                time_bonus = max(0, 5.0 - delivery_time / 50.0)  # Bonus for delivery < 250 steps
                base_reward += time_bonus

            return base_reward

        elif outcome == 0:  # Forwarded to neighbor
            # Enhanced progress-based reward
            depot_coords = self.simulator.depot.coords
            my_distance = util.euclidean_distance(self.drone.coords, depot_coords)
            neighbor_distance = util.euclidean_distance(selected_neighbor.coords, depot_coords)

            # Base reward for forwarding
            base_reward = 2.0

            # Progress reward (making progress toward depot)
            if neighbor_distance < my_distance:
                progress = (my_distance - neighbor_distance) / my_distance
                progress_reward = progress * 3.0
            else:
                # Small penalty for not making progress
                progress_reward = -0.5

            # Neighbor quality reward
            neighbor_quality = self.evaluate_neighbor_quality(selected_neighbor)
            quality_reward = neighbor_quality * 1.5

            # Energy efficiency reward
            energy_reward = (selected_neighbor.residual_energy / selected_neighbor.initial_energy) * 0.5

            total_reward = base_reward + progress_reward + quality_reward + energy_reward
            return max(total_reward, 0.1)  # Minimum positive reward for forwarding

        else:  # Failed, routing hole, or dropped
            # Strong penalty for routing holes to discourage them
            penalty = -8.0

            # Extra penalty if this drone has high energy (should be able to route)
            if self.drone.residual_energy / self.drone.initial_energy > 0.7:
                penalty -= 2.0

            # Extra penalty if there were available neighbors (routing hole is really bad)
            if len(self.opt_neighbors) > 0:
                penalty -= 3.0

            return penalty

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def train(self):
        """Enhanced training with better stability"""
        if len(self.memory) < self.learning_starts:
            return

        if len(self.memory) < self.batch_size:
            return

        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)

        # Prepare batch data
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state if e.next_state is not None
                                         else np.zeros(self.state_dim) for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)

        # Current Q-values
        current_q_values = self.main_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q-values from target network (Double DQN)
        with torch.no_grad():
            # Use main network to select actions
            next_actions = self.main_network(next_states).argmax(1)
            # Use target network to evaluate actions
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss with Huber loss for stability
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Update target network
        if self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def relay_selection(self, opt_neighbors, packet):
        """
        Enhanced relay selection with routing hole detection and fallback

        Args:
            opt_neighbors: List of available neighbor drones
            packet: Packet to be routed (tuple format from base class)

        Returns:
            selected_neighbor: Chosen neighbor drone or "ROUTING_HOLE"
        """
        if not opt_neighbors:
            self.routing_hole_count += 1
            return "ROUTING_HOLE"

        # Limit number of neighbors to consider (for fixed action space)
        neighbors_to_consider = opt_neighbors[:self.max_actions]

        # Get current enhanced state
        current_state = self.get_enhanced_state(packet)

        # Select action using DQN
        action_idx = self.select_action(current_state, neighbors_to_consider)

        if action_idx is None or action_idx >= len(neighbors_to_consider):
            # Fallback: use geographic routing (select neighbor closest to depot)
            depot_coords = self.simulator.depot.coords
            best_neighbor = min(neighbors_to_consider,
                                key=lambda n: util.euclidean_distance(n.coords, depot_coords))
            action_idx = neighbors_to_consider.index(best_neighbor)

        selected_neighbor = neighbors_to_consider[action_idx]

        # Store current experience data for later completion in feedback
        self.current_state = current_state
        self.current_action = action_idx
        self.current_neighbors = neighbors_to_consider.copy()
        self.current_packet = packet

        self.step_count += 1

        # Train the network
        self.train()

        return selected_neighbor

    def feedback(self, outcome, neighbor_id, best_q_value):
        """
        Enhanced feedback processing with better performance tracking

        Args:
            outcome: Routing outcome (0=forwarded, 1=delivered, -1=failed)
            neighbor_id: ID of neighbor that was selected
            best_q_value: Not used in pure DQN
        """
        # Check if we have current experience to complete
        if self.current_state is None:
            return

        # Find the selected neighbor
        selected_neighbor = None
        for neighbor in self.current_neighbors:
            if neighbor.identifier == neighbor_id:
                selected_neighbor = neighbor
                break

        if selected_neighbor is None:
            # Handle routing hole case
            if isinstance(neighbor_id, str) or neighbor_id == -1:
                self.routing_hole_count += 1
                reward = self.compute_enhanced_reward(-1, None, self.current_packet)
                self.recent_performance.append(0)  # Failed
            else:
                # Clean up and return
                self._cleanup_current_experience()
                return
        else:
            # Compute reward for successful neighbor selection
            reward = self.compute_enhanced_reward(outcome, selected_neighbor, self.current_packet)

            # Track performance
            if outcome == 1:  # Delivered
                self.successful_forwards += 1
                self.recent_performance.append(1)
            elif outcome == 0:  # Forwarded
                self.recent_performance.append(0.5)  # Partial success
            else:  # Failed
                self.recent_performance.append(0)

        # Get next state
        next_state = self.get_enhanced_state()

        # Determine if episode is done
        done = (outcome == 1 or outcome == -1)  # Terminal if delivered or failed

        # Store complete experience
        self.store_experience(
            self.current_state,
            self.current_action,
            reward,
            next_state if not done else None,
            done
        )

        # Clean up
        self._cleanup_current_experience()

    def _cleanup_current_experience(self):
        """Clean up current experience data"""
        self.current_state = None
        self.current_action = None
        self.current_neighbors = []
        self.current_packet = None

    def get_performance_metrics(self):
        """Get enhanced performance metrics"""
        recent_success_rate = 0.0
        if len(self.recent_performance) > 0:
            recent_success_rate = np.mean(self.recent_performance)

        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'total_steps': self.step_count,
            'learning_started': len(self.memory) >= self.learning_starts,
            'routing_hole_count': self.routing_hole_count,
            'successful_forwards': self.successful_forwards,
            'recent_success_rate': recent_success_rate,
            'total_attempts': self.routing_hole_count + self.successful_forwards,
            'success_rate': self.successful_forwards / max(self.routing_hole_count + self.successful_forwards, 1)
        }

    def save_model(self, filepath):
        """Save the enhanced model"""
        torch.save({
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'routing_hole_count': self.routing_hole_count,
            'successful_forwards': self.successful_forwards,
            'recent_performance': list(self.recent_performance)
        }, filepath)
        print(f"[Fixed Simple DQN] Model saved to {filepath}")

    def load_model(self, filepath):
        """Load the enhanced model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.routing_hole_count = checkpoint.get('routing_hole_count', 0)
        self.successful_forwards = checkpoint.get('successful_forwards', 0)
        if 'recent_performance' in checkpoint:
            self.recent_performance = deque(checkpoint['recent_performance'], maxlen=500)
        print(f"[Fixed Simple DQN] Model loaded from {filepath}")