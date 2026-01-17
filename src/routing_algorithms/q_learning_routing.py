import random
import numpy as np
import math

from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util

INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.995

#turn of debug

class QMAR(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)

        # Reward parameters
        self.maxReward = 5
        self.minReward = -5
        self.w = 0.7
        self.tau = 1.0

        # Exploration parameters - tương tự DQN
        self.epsilon = 0.9  # Giảm từ 1.0 xuống 0.9
        self.epsilon_decay = 0.9990  # Tăng tốc độ decay như DQN
        self.epsilon_min = 0.05  # Tăng min epsilon như DQN
        self.ucb_c = 2.0

        # Q-table initialization
        n = self.simulator.n_drones
        self.q_table = np.zeros((n, n))

        # Initialize Q-table với depot coordinates thay vì có thể thiếu
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        # Lấy coordinates của drone j, fallback về depot nếu không có
                        if j < len(simulator.drones):
                            drone_j_coords = simulator.drones[j].coords
                        else:
                            drone_j_coords = simulator.depot_coordinates

                        dist = util.euclidean_distance(drone_j_coords, simulator.depot_coordinates)
                        max_dist = self.drone.communication_range * 2
                        q_init = 1 - (dist / max_dist) if max_dist > 0 else 0
                        self.q_table[i, j] = np.clip(q_init, -3, 3)
                    except Exception as e:
                        # print(f"[DEBUG] Error initializing Q-table [{i},{j}]: {e}")  # Disabled for performance
                        self.q_table[i, j] = 0.0

        # Performance tracking như DQN
        self.routing_hole_count = 0
        self.successful_forwards = 0
        self.recent_performance = []

        # print(f"[DEBUG] QMAR Q-table size: {self.q_table.shape}")  # Disabled for performance

    def feedback(self, outcome, id_j, best_action, link_quality=0):
        """
        Enhanced feedback processing tương tự DQN
        """
        # Calculate error penalty
        error_penalty = 0
        if hasattr(best_action, 'error_code') and best_action.error_code:
            if best_action.error_code == 2:
                error_penalty = -2.0
            elif best_action.error_code == 1:
                error_penalty = -1.0

        if hasattr(best_action, 'failure_reason'):
            if best_action.failure_reason == "no_neighbors":
                error_penalty = -1.5
            elif best_action.failure_reason == "poor_link":
                error_penalty = -1.0

        # Validate id_j - xử lý routing hole như DQN
        if isinstance(id_j, str) or id_j == -1:
            # This is a routing hole
            self.routing_hole_count += 1
            reward = -8.0 + error_penalty  # Strong penalty like DQN
            self.recent_performance.append(0)
            # print(f"[DEBUG] QMAR: Routing hole detected, penalty: {reward}")  # Disabled for performance
            return

        if not isinstance(id_j, int) or id_j < 0 or id_j >= self.simulator.n_drones:
            # print(f"[DEBUG] QMAR: Invalid id_j {id_j} in feedback, skipping Q update")  # Disabled for performance
            return

        # Get learning parameters safely
        try:
            alpha = self.drone.neighbor_table[id_j, 10] if hasattr(self.drone, 'neighbor_table') else 0.1
            gamma = self.drone.neighbor_table[id_j, 7] if hasattr(self.drone, 'neighbor_table') else 0.99
            delay = (self.drone.neighbor_table[id_j, 8] + self.drone.neighbor_table[id_j, 11]) if hasattr(self.drone,
                                                                                                          'neighbor_table') else 0.1
        except (IndexError, AttributeError):
            alpha = 0.1
            gamma = 0.99
            delay = 0.1

        # Get current Q-value
        Q_value_i_j = self.q_table[self.drone.identifier, id_j]

        # Get neighbor safely
        neighbor = self.simulator.get_drone_by_id(id_j)
        if neighbor is None:
            # print(f"[DEBUG] QMAR: Cannot find neighbor with id {id_j}")  # Disabled for performance
            return

        # Calculate reward using enhanced method like DQN
        reward = self.compute_enhanced_reward(outcome, neighbor, delay, link_quality)
        reward += error_penalty

        # Update Q-value
        max_q_next = np.max(self.q_table[id_j, :]) if np.any(self.q_table[id_j, :]) else 0
        self.q_table[self.drone.identifier, id_j] = Q_value_i_j + alpha * (reward + gamma * max_q_next - Q_value_i_j)

        # Track performance like DQN
        if outcome == 1:  # Delivered
            self.successful_forwards += 1
            self.recent_performance.append(1)
        elif outcome == 0:  # Forwarded
            self.recent_performance.append(0.5)
        else:  # Failed
            self.recent_performance.append(0)

        # Keep recent performance within limits
        if len(self.recent_performance) > 500:
            self.recent_performance.pop(0)

    def compute_enhanced_reward(self, outcome, neighbor, delay, link_quality):
        """
        Enhanced reward computation inspired by DQN
        """
        depot_coords = self.simulator.depot_coordinates
        my_distance = util.euclidean_distance(self.drone.coords, depot_coords)

        if outcome == 1:  # Delivered successfully
            base_reward = 15.0
            # Time bonus for fast delivery
            if delay < 50:
                time_bonus = 5.0 - (delay / 10.0)
                base_reward += time_bonus
            return base_reward

        elif outcome == 0:  # Forwarded to neighbor
            base_reward = 2.0

            # Progress reward
            neighbor_distance = util.euclidean_distance(neighbor.coords, depot_coords)
            if neighbor_distance < my_distance and my_distance > 0:
                progress = (my_distance - neighbor_distance) / my_distance
                progress_reward = progress * 3.0
            else:
                progress_reward = -0.5

            # Link quality reward
            quality_reward = link_quality * 1.5

            # Energy efficiency reward
            try:
                energy_reward = (neighbor.residual_energy / neighbor.initial_energy) * 0.5
            except (AttributeError, ZeroDivisionError):
                energy_reward = 0.0

            total_reward = base_reward + progress_reward + quality_reward + energy_reward
            return max(total_reward, 0.1)  # Minimum positive reward

        else:  # Failed, routing hole, or dropped
            penalty = -8.0

            # Extra penalty if this drone has high energy
            try:
                if self.drone.residual_energy / self.drone.initial_energy > 0.7:
                    penalty -= 2.0
            except (AttributeError, ZeroDivisionError):
                pass

            return penalty

    def calculate_link_quality(self, drone, neighbor, max_communication_range):
        """
        Improved link quality calculation with error handling
        """
        try:
            alpha = 0.3
            beta = 0.2
            gamma = 0.2
            delta = 0.2

            distance = util.euclidean_distance(drone.coords, neighbor.coords)
            distance_factor = max(0, 1 - (distance / max_communication_range)) if max_communication_range > 0 else 0

            # Safe speed calculation
            drone_speed = getattr(drone, 'speed', 0)
            neighbor_speed = getattr(neighbor, 'speed', 0)
            relative_speed = abs(drone_speed - neighbor_speed)
            speed_factor = math.exp(-relative_speed / 10) if relative_speed < 100 else 0

            packet_loss_rate = self.calculate_packet_loss_rate()
            packet_loss_factor = max(0, min(1, 1 - packet_loss_rate))

            rssi = -20 * math.log10(distance + 1e-5)
            rssi_normalized = max(0, min(1, (rssi + 100) / 100))

            bitrate = self.get_link_bitrate(drone, neighbor)
            bitrate_factor = min(1, bitrate / 54000000)

            link_quality = (
                    alpha * distance_factor +
                    beta * speed_factor +
                    gamma * packet_loss_factor +
                    delta * rssi_normalized +
                    0.1 * bitrate_factor
            )
            return max(0, min(1, link_quality))
        except Exception as e:
            # print(f"[DEBUG] QMAR: Error calculating link quality: {e}")  # Disabled for performance
            return 0.5  # Default neutral quality

    def calculate_packet_loss_rate(self):
        """Safe packet loss rate calculation"""
        try:
            total_packets = getattr(self.simulator, 'all_data_packets_in_simulation', 1)
            delivered_packets = len(getattr(self.simulator, 'drones_packets_to_depot', []))
            if total_packets == 0:
                return 0.0
            return 1 - (delivered_packets / total_packets)
        except Exception:
            return 0.0  # Default to no loss

    def get_link_bitrate(self, drone, neighbor):
        """Safe bitrate calculation"""
        try:
            # Simple distance-based bitrate model
            distance = util.euclidean_distance(drone.coords, neighbor.coords)
            max_range = drone.communication_range
            if distance >= max_range:
                return 1000000  # 1 Mbps minimum

            # Linearly decrease from 54 Mbps to 1 Mbps based on distance
            bitrate = 54000000 - (distance / max_range) * 53000000
            return max(bitrate, 1000000)
        except Exception:
            return 11000000  # Default 11 Mbps

    def softmax(self, q_values):
        """Safe softmax calculation"""
        try:
            q_values = np.array(q_values)
            q_values = np.nan_to_num(q_values, nan=0.0)
            q_values -= np.max(q_values)
            exp_q = np.exp(q_values / self.tau)
            sum_exp_q = np.sum(exp_q)
            if sum_exp_q == 0 or np.isnan(sum_exp_q):
                return np.ones_like(q_values) / len(q_values)
            return exp_q / sum_exp_q
        except Exception:
            # Fallback to uniform distribution
            return np.ones_like(q_values) / len(q_values)

    def epsilon_greedy(self, q_values):
        """Safe epsilon-greedy selection"""
        try:
            if np.random.rand() < self.epsilon:
                return random.randint(0, len(q_values) - 1)
            return np.argmax(q_values)
        except Exception:
            return random.randint(0, len(q_values) - 1)

    def predict_future_position(self, drone, steps_ahead=3):
        """Safe future position prediction"""
        try:
            if not hasattr(drone, 'past_positions') or len(drone.past_positions) < 2:
                return drone.coords

            deltas = []
            for i in range(len(drone.past_positions) - 1):
                delta = np.subtract(drone.past_positions[i + 1], drone.past_positions[i])
                deltas.append(delta)

            avg_delta = np.mean(deltas, axis=0)
            future_pos = np.add(drone.coords, avg_delta * steps_ahead)
            return tuple(future_pos)
        except Exception:
            return drone.coords

    def computeActualVel(self, j, node_j, distance_i):
        """Safe velocity computation"""
        try:
            future_pos = self.predict_future_position(node_j, steps_ahead=3)
            distance_j = util.euclidean_distance(future_pos, self.simulator.depot_coordinates)
            distance_i_j = util.euclidean_distance(self.drone.coords, future_pos)

            # Safe delay calculation
            try:
                delay = self.drone.neighbor_table[j, 8] + self.drone.neighbor_table[j, 11]
            except (AttributeError, IndexError):
                delay = 0.01

            delay = delay if delay > 0 else 0.01
            return (distance_i - distance_j) / delay, distance_i_j
        except Exception as e:
            # print(f"[DEBUG] QMAR: Error in computeActualVel: {e}")  # Disabled for performance
            return 0.0, distance_i

    def relay_selection(self, opt_neighbors, data):
        """
        Enhanced relay selection với error handling như DQN
        """
        if not opt_neighbors:
            self.routing_hole_count += 1
            return "ROUTING_HOLE"  # Thay đổi từ "RHP" thành "ROUTING_HOLE" như DQN

        packet = data[0] if isinstance(data, (list, tuple)) else data
        candidates = []
        candidates2 = []
        q_values = []

        priority_factor = getattr(packet, 'priority', 0) * 0.2

        # Safe IP comparison - loại bỏ util.ip_to_int
        try:
            depot_ip = self.simulator.depot.ip_address if hasattr(self.simulator.depot, 'ip_address') else None
            packet_dst_ip = getattr(packet, 'dst_ip', None)

            # Direct string comparison instead of ip_to_int
            if depot_ip and packet_dst_ip and packet_dst_ip != depot_ip:
                # print(f"[DEBUG] QMAR: Packet {getattr(packet, 'identifier', 'unknown')} not destined for depot")  # Disabled
                return "ROUTING_HOLE"
        except Exception as e:
            # print(f"[DEBUG] QMAR: Error in IP comparison: {e}")  # Disabled for performance
            # Continue with routing if IP comparison fails
            pass

        # Process neighbors
        for node_j in opt_neighbors:
            try:
                j = node_j.identifier

                # Safe deadline calculation
                deadline_seconds = 1.0  # Default deadline
                try:
                    if hasattr(packet, 'event_ref') and hasattr(packet.event_ref, 'deadline'):
                        deadline = packet.event_ref.deadline - self.simulator.cur_step
                        deadline_seconds = deadline * self.simulator.time_step_duration
                    elif hasattr(packet, 'ttl'):
                        deadline_seconds = packet.ttl * self.simulator.time_step_duration
                except Exception:
                    deadline_seconds = 1.0

                if deadline_seconds < 0.2:
                    # print(f"[DEBUG] QMAR: Drone {j} skipped: deadline {deadline_seconds} < 0.2")  # Disabled
                    continue

                distance_i = util.euclidean_distance(self.drone.coords, self.simulator.depot_coordinates)
                actual_v, distance_i_j = self.computeActualVel(j, node_j, distance_i)
                req_v = distance_i / max(deadline_seconds, 0.02)

                if actual_v >= req_v * 0.5:
                    # Safe neighbor table access
                    try:
                        LQ = self.drone.neighbor_table[j, 12] if hasattr(self.drone, 'neighbor_table') else 0.8
                    except (AttributeError, IndexError):
                        LQ = 0.8  # Default good link quality

                    R = self.drone.communication_range
                    M = 1 - (distance_i_j / R) if distance_i_j <= R and R > 0 else 0
                    k = M * LQ
                    candidates.append((node_j, k))

                    # Safe Q-table access
                    try:
                        q_val = self.q_table[self.drone.identifier, j] * k + priority_factor
                    except (IndexError, AttributeError):
                        q_val = k + priority_factor

                    q_values.append(q_val)
                    # print(f"[DEBUG] QMAR: Drone {j} added to candidates: actual_v={actual_v}, req_v={req_v}, k={k}")  # Disabled

                elif actual_v > 0:
                    candidates2.append((node_j, actual_v))
                    # print(f"[DEBUG] QMAR: Drone {j} added to candidates2: actual_v={actual_v}")  # Disabled

            except Exception as e:
                # print(f"[DEBUG] QMAR: Error processing neighbor {getattr(node_j, 'identifier', 'unknown')}: {e}")  # Disabled
                continue

        # Selection logic
        chosen = None
        try:
            if candidates:
                # Adaptive epsilon based on performance like DQN
                current_epsilon = self.epsilon
                if len(self.recent_performance) >= 200:
                    recent_success_rate = np.mean(self.recent_performance[-200:])
                    if recent_success_rate < 0.3:
                        current_epsilon = min(0.4, self.epsilon * 1.5)

                if current_epsilon > self.epsilon_min:
                    selected_idx = self.epsilon_greedy(q_values)
                    # Decay epsilon like DQN
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                else:
                    probabilities = self.softmax(np.array(q_values))
                    probabilities = np.nan_to_num(probabilities, nan=1.0 / len(probabilities))
                    probabilities /= np.sum(probabilities)
                    selected_idx = np.random.choice(len(candidates), p=probabilities)

                chosen = candidates[selected_idx][0]
                # print(f"[DEBUG] QMAR: Selected drone {chosen.identifier} from candidates")  # Disabled

            elif candidates2:
                chosen = max(candidates2, key=lambda x: x[1])[0]
                # print(f"[DEBUG] QMAR: Selected drone {chosen.identifier} from candidates2")  # Disabled

            else:
                # Fallback to closest to depot
                chosen = min(opt_neighbors,
                             key=lambda x: util.euclidean_distance(x.coords, self.simulator.depot_coordinates),
                             default=None)
                if chosen:
                    pass  # print(f"[DEBUG] QMAR: Selected drone {chosen.identifier} as closest to depot")  # Disabled

        except Exception as e:
            # print(f"[DEBUG] QMAR: Error in selection logic: {e}")  # Disabled for performance
            chosen = None

        if chosen is None:
            # print(f"[DEBUG] QMAR: No suitable neighbor found")  # Disabled for performance
            self.routing_hole_count += 1
            return "ROUTING_HOLE"

        return chosen

    def get_performance_metrics(self):
        """Get performance metrics like DQN"""
        recent_success_rate = 0.0
        if len(self.recent_performance) > 0:
            recent_success_rate = np.mean(self.recent_performance[-200:])

        return {
            'epsilon': self.epsilon,
            'routing_hole_count': self.routing_hole_count,
            'successful_forwards': self.successful_forwards,
            'recent_success_rate': recent_success_rate,
            'total_attempts': self.routing_hole_count + self.successful_forwards,
            'success_rate': self.successful_forwards / max(self.routing_hole_count + self.successful_forwards, 1),
            'q_table_stats': {
                'mean': np.mean(self.q_table),
                'std': np.std(self.q_table),
                'min': np.min(self.q_table),
                'max': np.max(self.q_table)
            }
        }