"""
VDN-QMAR: Value Decomposition Networks for Multi-Agent Q-learning Routing

Key Features:
- Centralized training, decentralized execution (CTDE)
- Value decomposition: Q_total = sum(Q_i)
- Coordination through joint value function
- No communication overhead during execution
"""

import numpy as np
import random
import math
from collections import deque
from typing import List, Dict, Tuple

try:
    from src.routing_algorithms.BASE_routing import BASE_routing
    from src.utilities import utilities as util
except:
    from BASE_routing import BASE_routing
    import utilities as util


# ========================================
# GLOBAL COORDINATOR (Centralized Training)
# ========================================

class VDN_Coordinator:
    """
    Centralized coordinator for training
    Manages joint Q-value and coordinates updates
    """
    
    def __init__(self, n_drones, simulator):
        self.n_drones = n_drones
        self.simulator = simulator
        
        # Experience replay buffer (shared across all agents)
        self.replay_buffer = deque(maxlen=10000)
        
        # Training stats
        self.training_steps = 0
        self.joint_rewards = []
        
        # Coordination parameters
        self.batch_size = 64
        self.target_update_freq = 500
        
        print(f"[VDN-Coordinator] Initialized for {n_drones} drones")
    
    def store_transition(self, joint_state, actions, rewards, next_joint_state, dones):
        """
        Store joint transition
        
        Args:
            joint_state: Dict of {drone_id: state}
            actions: Dict of {drone_id: action}
            rewards: Dict of {drone_id: reward}
            next_joint_state: Dict of {drone_id: next_state}
            dones: Dict of {drone_id: done}
        """
        self.replay_buffer.append({
            'joint_state': joint_state,
            'actions': actions,
            'rewards': rewards,
            'next_joint_state': next_joint_state,
            'dones': dones
        })
    
    def compute_joint_q_value(self, drone_q_values):
        """
        VDN: Q_total = sum of individual Q-values
        
        Args:
            drone_q_values: Dict of {drone_id: Q_value}
        
        Returns:
            joint_q: Scalar joint Q-value
        """
        return sum(drone_q_values.values())
    
    def update_all_agents(self, agents):
        """
        Centralized training update for all agents
        
        Args:
            agents: Dict of {drone_id: VDN_QMAR_Agent}
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        for transition in batch:
            joint_state = transition['joint_state']
            actions = transition['actions']
            rewards = transition['rewards']
            next_joint_state = transition['next_joint_state']
            dones = transition['dones']
            
            # Compute current joint Q-value
            current_q_values = {}
            for drone_id, agent in agents.items():
                if drone_id in joint_state and drone_id in actions:
                    state = joint_state[drone_id]
                    action = actions[drone_id]
                    q_val = agent.get_q_value(state, action)
                    current_q_values[drone_id] = q_val
            
            current_joint_q = self.compute_joint_q_value(current_q_values)
            
            # Compute target joint Q-value
            next_q_values = {}
            for drone_id, agent in agents.items():
                if drone_id in next_joint_state:
                    next_state = next_joint_state[drone_id]
                    max_next_q = agent.get_max_q_value(next_state)
                    next_q_values[drone_id] = max_next_q
            
            next_joint_q = self.compute_joint_q_value(next_q_values)
            
            # Joint reward (sum of individual rewards)
            joint_reward = sum(rewards.values())
            
            # TD target
            gamma = 0.99
            done = any(dones.values())
            target_joint_q = joint_reward + (0 if done else gamma * next_joint_q)
            
            # TD error
            td_error = target_joint_q - current_joint_q
            
            # Update each agent's Q-table proportionally
            for drone_id, agent in agents.items():
                if drone_id in joint_state and drone_id in actions:
                    state = joint_state[drone_id]
                    action = actions[drone_id]
                    reward = rewards.get(drone_id, 0)
                    
                    # Distribute TD error based on agent's contribution
                    if len(current_q_values) > 0:
                        contribution = current_q_values.get(drone_id, 0) / max(abs(current_joint_q), 1e-6)
                        agent_td_error = td_error * contribution
                    else:
                        agent_td_error = td_error / len(agents)
                    
                    agent.update_q_value(state, action, agent_td_error)
        
        self.training_steps += 1
        
        if self.training_steps % 100 == 0:
            print(f"[VDN-Coordinator] Training step {self.training_steps}, "
                  f"Buffer size: {len(self.replay_buffer)}")


# ========================================
# MULTI-AGENT QMAR
# ========================================

class VDN_QMAR(BASE_routing):
    """
    Multi-Agent QMAR with Value Decomposition Networks
    
    Improvements over single-agent QMAR:
    - Coordinated learning through VDN
    - Shared experience replay
    - Stable training with centralized critic
    - Better load balancing
    """
    
    def __init__(self, drone, simulator):
        super().__init__(drone, simulator)
        
        # Q-table (same as original QMAR)
        n = self.simulator.n_drones
        self.q_table = np.zeros((n, n))
        
        # Initialize Q-table with distance heuristic
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        if j < len(simulator.drones):
                            drone_j_coords = simulator.drones[j].coords
                        else:
                            drone_j_coords = simulator.depot_coordinates
                        
                        dist = util.euclidean_distance(drone_j_coords, simulator.depot_coordinates)
                        max_dist = self.drone.communication_range * 2
                        q_init = 1 - (dist / max_dist) if max_dist > 0 else 0
                        self.q_table[i, j] = np.clip(q_init, -3, 3)
                    except:
                        self.q_table[i, j] = 0.0
        
        # Multi-agent parameters
        self.epsilon = 0.9
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        
        # Coordination
        self.neighbor_actions = {}  # Track neighbors' actions for coordination
        self.last_state = None
        self.last_action = None
        
        # Stats
        self.routing_hole_count = 0
        self.successful_forwards = 0
        self.coordination_score = 0.0
        
        # Get or create global coordinator
        if not hasattr(simulator, 'vdn_coordinator'):
            simulator.vdn_coordinator = VDN_Coordinator(n, simulator)
        self.coordinator = simulator.vdn_coordinator
        
        print(f"[VDN-QMAR] Drone {drone.identifier} initialized")
    
    # ========================================
    # STATE REPRESENTATION
    # ========================================
    
    def get_state_key(self, neighbors):
        """
        Create state representation
        
        State: (my_position, my_energy, neighbors_info)
        """
        # Discretize position
        grid_x = int(self.drone.coords[0] / 100)
        grid_y = int(self.drone.coords[1] / 100)
        
        # Discretize energy
        energy_level = int(self.drone.residual_energy / self.drone.initial_energy * 10)
        
        # Neighbor count
        n_neighbors = len(neighbors)
        
        state = (grid_x, grid_y, energy_level, n_neighbors)
        return state
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        try:
            i = self.drone.identifier
            j = action  # action is drone_id
            return self.q_table[i, j]
        except:
            return 0.0
    
    def get_max_q_value(self, state):
        """Get maximum Q-value for state"""
        try:
            i = self.drone.identifier
            return np.max(self.q_table[i, :])
        except:
            return 0.0
    
    def update_q_value(self, state, action, td_error):
        """Update Q-value using TD error from coordinator"""
        try:
            i = self.drone.identifier
            j = action
            self.q_table[i, j] += self.alpha * td_error
            self.q_table[i, j] = np.clip(self.q_table[i, j], -10, 10)
        except:
            pass
    
    # ========================================
    # RELAY SELECTION (Multi-Agent)
    # ========================================
    
    def relay_selection(self, opt_neighbors, packet):
        """
        Multi-agent relay selection with coordination
        """
        if not opt_neighbors:
            self.routing_hole_count += 1
            return "ROUTING_HOLE"
        
        # Get current state
        state = self.get_state_key(opt_neighbors)
        
        # Get Q-values for all neighbors
        q_values = []
        neighbor_ids = []
        
        for neighbor in opt_neighbors:
            j = neighbor.identifier
            neighbor_ids.append(j)
            
            # Base Q-value
            q_val = self.q_table[self.drone.identifier, j]
            
            # Coordination bonus: avoid congested neighbors
            congestion_penalty = 0
            if hasattr(neighbor, 'buffer_length'):
                buffer_ratio = neighbor.buffer_length() / max(neighbor.buffer_max_size, 1)
                congestion_penalty = buffer_ratio * 0.3
            
            # Hotspot avoidance: penalize neighbors that have been selected frequently in recent transitions.
            # NOTE: replay_buffer stores transition dicts, so we must inspect the 'actions' field of each transition.
            recent_window = 50
            recent = list(self.coordinator.replay_buffer)[-recent_window:]
            recent_selections = sum(
                1 for tr in recent
                if isinstance(tr, dict) and j in tr.get('actions', {}).values()
            )
            # Scale to a bounded penalty in [0, 0.5]
            hotspot_penalty = min(recent_selections / 10.0, 0.5)
            
            # Adjusted Q-value with coordination
            adjusted_q = q_val - congestion_penalty - hotspot_penalty
            q_values.append(adjusted_q)
        
        # Epsilon-greedy with coordination awareness
        if np.random.rand() < self.epsilon:
            # Exploration: random but avoid very congested nodes
            # Softmax exploration biased toward better relays (numerically stable)
            weights = self.softmax(q_values)
            action_idx = np.random.choice(len(q_values), p=weights)
        else:
            # Exploitation: best Q-value
            action_idx = np.argmax(q_values)
        
        chosen_neighbor = opt_neighbors[action_idx]
        chosen_id = neighbor_ids[action_idx]
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store for coordination
        self.last_state = state
        self.last_action = chosen_id
        
        return chosen_neighbor
    
    # ========================================
    # FEEDBACK (Multi-Agent Learning)
    # ========================================
    
    def feedback(self, outcome, id_j, best_action, link_quality=0.0):
        """
        Multi-agent feedback with VDN coordination
        """
        # Individual reward
        if outcome == 1:  # Delivered
            reward = 15.0
            done = True
            self.successful_forwards += 1
        elif outcome == 0:  # Forwarded
            reward = 2.0 + 0.5 * link_quality
            done = False
        else:  # Failed
            reward = -8.0
            done = True
            self.routing_hole_count += 1
        
        # If we have previous state-action, store transition
        if self.last_state is not None and self.last_action is not None:
            # Get current state
            next_state = self.get_state_key(self.opt_neighbors)
            
            # Collect joint information from all drones
            joint_state = {}
            actions = {}
            rewards_dict = {}
            next_joint_state = {}
            dones_dict = {}
            
            # Add this drone's transition
            joint_state[self.drone.identifier] = self.last_state
            actions[self.drone.identifier] = self.last_action
            rewards_dict[self.drone.identifier] = reward
            next_joint_state[self.drone.identifier] = next_state
            dones_dict[self.drone.identifier] = done
            
            # Try to get neighbors' information for better coordination
            for neighbor in self.opt_neighbors:
                if hasattr(neighbor, 'routing_algorithm'):
                    neighbor_algo = neighbor.routing_algorithm
                    if isinstance(neighbor_algo, VDN_QMAR):
                        if neighbor_algo.last_state is not None:
                            joint_state[neighbor.identifier] = neighbor_algo.last_state
                            if neighbor_algo.last_action is not None:
                                actions[neighbor.identifier] = neighbor_algo.last_action
            
            # Store in coordinator's buffer
            self.coordinator.store_transition(
                joint_state, actions, rewards_dict, next_joint_state, dones_dict
            )
            
            # Centralized training update
            if len(self.coordinator.replay_buffer) >= self.coordinator.batch_size:
                # Get all VDN-QMAR agents
                agents = {}
                for drone in self.simulator.drones:
                    if isinstance(drone.routing_algorithm, VDN_QMAR):
                        agents[drone.identifier] = drone.routing_algorithm
                
                # Update all agents jointly
                if self.drone.identifier == 0:  # Only update once per step
                    self.coordinator.update_all_agents(agents)
        
        # Reset state-action for next decision
        if done:
            self.last_state = None
            self.last_action = None
    
    # ========================================
    # COORDINATION UTILITIES
    # ========================================
    
    def get_coordination_info(self):
        """
        Get information about coordination with other agents
        """
        recent_transitions = list(self.coordinator.replay_buffer)[-100:]
        
        # Count coordinated actions (different drones chose different relays)
        coordinated = 0
        for t in recent_transitions:
            actions = t.get('actions', {})
            if len(set(actions.values())) == len(actions):  # All different
                coordinated += 1
        
        self.coordination_score = coordinated / max(len(recent_transitions), 1)
        return self.coordination_score
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        coordination = self.get_coordination_info()
        
        return {
            'epsilon': self.epsilon,
            'routing_holes': self.routing_hole_count,
            'successful_forwards': self.successful_forwards,
            'coordination_score': coordination,
            'q_table_stats': {
                'mean': np.mean(self.q_table),
                'std': np.std(self.q_table),
                'min': np.min(self.q_table),
                'max': np.max(self.q_table)
            }
        }
    
    # ========================================
    # AUXILIARY FUNCTIONS (from original QMAR)
    # ========================================
    
    def calculate_link_quality(self, drone, neighbor, max_communication_range):
        """Calculate link quality (simplified)"""
        try:
            distance = util.euclidean_distance(drone.coords, neighbor.coords)
            distance_factor = max(0, 1 - (distance / max_communication_range))
            
            # Simple quality estimate
            link_quality = distance_factor * 0.8 + 0.2
            return max(0, min(1, link_quality))
        except:
            return 0.5
    
    def softmax(self, q_values):
        """Softmax for action selection"""
        try:
            q_values = np.array(q_values)
            q_values = np.nan_to_num(q_values, nan=0.0)
            q_values -= np.max(q_values)
            exp_q = np.exp(q_values / 1.0)
            sum_exp_q = np.sum(exp_q)
            if sum_exp_q == 0 or np.isnan(sum_exp_q):
                return np.ones_like(q_values) / len(q_values)
            return exp_q / sum_exp_q
        except:
            return np.ones_like(q_values) / len(q_values)
