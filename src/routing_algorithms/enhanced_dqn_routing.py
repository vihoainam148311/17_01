# enhanced_dqn_routing.py
# EDQN cho FANET UAV dân dụng – tối ưu tốc độ mô phỏng, giữ DDQN và 2 bộ nhớ ưu tiên
# PEP8, Python 3.7 compatible

import random
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util
from src.utilities import config as cfg


class BalancedDQN(nn.Module):
    """
    Mạng DQN cân bằng: đủ sâu để học tốt nhưng vẫn nhẹ để chạy nhanh.

    Kiến trúc:
      - 3 tầng ẩn (256 → 128 → 64)
      - BatchNorm dùng khi batch_size > 1 (tránh lỗi khi batch=1)
      - Dropout nhỏ để ổn định (không dùng trong đánh giá)
    """

    def __init__(self, input_dim, output_dim):
        super(BalancedDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        if x.numel() == 0:
            # Phòng thủ: không có state hợp lệ
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

        # Trả về (A,) khi batch=1; (B, A) khi batch>1
        return x.squeeze(0) if x.shape[0] == 1 else x


class EnhancedDQN_Routing(BASE_routing):
    """
    EDQN được tinh chỉnh cho UAV dân dụng:
      - Giữ cơ chế Double DQN (DDQN): online chọn a*, target ước lượng Q.
      - Hai bộ nhớ replay: thường + ưu tiên (ưu tiên chứa lần giao gói thành công
        và phần thưởng lớn) -> học nhanh nhưng ổn định.
      - Trạng thái mở rộng nhưng gọn (mặc định 10 phần tử) để mô phỏng chạy nhanh.
      - Chế độ 'thận trọng' khi tỉ lệ giao thành công giảm.
    """

    def __init__(self, drone, simulator):
        super().__init__(drone, simulator)

        # ----- Kích thước trạng thái/hành động (giữ gọn để nhanh) -----
        # Gợi ý: có thể tăng lên 12 nếu muốn thêm buffer_ratio & success_rate
        self.state_size = 10
        self.action_size = simulator.n_drones

        # ----- Replay buffers -----
        mem_size = getattr(cfg, "EDQN_MEMORY_SIZE", 3000)
        self.memory = deque(maxlen=mem_size)
        # Bộ nhớ ưu tiên (ưu tiên các mẫu reward lớn & thành công)
        self.priority_memory = deque(maxlen=max(1000, mem_size // 3))

        # ----- Siêu tham số (đọc từ config nếu có) -----
        self.gamma = getattr(cfg, "EDQN_GAMMA", 0.98)
        self.epsilon = getattr(cfg, "EDQN_EPSILON_START", 0.95)
        self.epsilon_min = getattr(cfg, "EDQN_EPSILON_MIN", 0.02)
        self.epsilon_decay = getattr(cfg, "EDQN_EPSILON_DECAY", 0.9995)
        self.learning_rate = getattr(cfg, "EDQN_LEARNING_RATE", 3e-4)
        self.batch_size = getattr(cfg, "EDQN_BATCH_SIZE", 64)
        self.target_update_freq = getattr(cfg, "EDQN_TARGET_UPDATE_FREQ", 100)
        self.cons_threshold = getattr(cfg, "EDQN_CONSERVATIVE_THRESHOLD", 0.7)

        # ----- Mạng Q -----
        self.model = BalancedDQN(self.state_size, self.action_size)
        self.target_model = BalancedDQN(self.state_size, self.action_size)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        self.loss_function = torch.nn.MSELoss()
        self.update_target_network()

        # ----- Theo dõi hiệu năng -----
        self.neighbor_history = deque(maxlen=200)
        self.delivery_success_history = deque(maxlen=500)
        self.reward_history = deque(maxlen=2000)
        self.step_count = 0
        self.success_count = 0
        self.total_attempts = 0
        self.conservative_mode = False

        # Biến tạm cho học tăng cường mỗi lần chọn hành động
        self.last_state = None
        self.last_action = None
        self.last_smart_neighbors = None
        self.last_chosen_neighbor_id = None

    # ----------------- Tiện ích mạng DDQN -----------------

    def update_target_network(self):
        """Cập nhật tham số mạng target = mạng online (soft copy đơn giản)."""
        self.target_model.load_state_dict(self.model.state_dict())

    # ----------------- Trạng thái gọn nhẹ -----------------

    def get_comprehensive_state(self):
        """
        Trạng thái gọn (<= self.state_size phần tử) – chuẩn hoá để mô phỏng nhanh:
          [x_norm, y_norm, energy, dist2depot, dx_norm, dy_norm,
           speed_norm, connectivity, avg_nb_energy, best_link_q, ...]
        """
        try:
            drone = self.drone
            if drone is None:
                return np.zeros(self.state_size, dtype=np.float32)

            # Chuẩn hoá vị trí
            x_norm = max(0.0, min(1.0, drone.coords[0] /
                                  max(self.simulator.env_width, 1)))
            y_norm = max(0.0, min(1.0, drone.coords[1] /
                                  max(self.simulator.env_height, 1)))

            # Năng lượng & tốc độ
            energy_ratio = max(
                0.0, min(1.0, drone.residual_energy / max(drone.initial_energy, 1))
            )
            speed_norm = max(0.0, min(1.0, drone.speed / 20.0))

            # Khoảng cách & hướng đến depot
            depot_distance = util.euclidean_distance(
                drone.coords, self.simulator.depot_coordinates
            )
            max_distance = math.sqrt(
                self.simulator.env_width ** 2 + self.simulator.env_height ** 2
            )
            distance_ratio = max(0.0, min(1.0, depot_distance / max(max_distance, 1)))
            dx = (self.simulator.depot_coordinates[0] - drone.coords[0]) / max(
                self.simulator.env_width, 1
            )
            dy = (self.simulator.depot_coordinates[1] - drone.coords[1]) / max(
                self.simulator.env_height, 1
            )

            # Phân tích láng giềng (nhanh)
            current_step = getattr(self.simulator, "cur_step", 0)
            valid_neighbors = 0
            avg_neighbor_energy = 0.0
            best_link_quality = 0.0

            if hasattr(drone, "neighbor_table") and drone.neighbor_table is not None:
                neighbor_energies = []
                link_qualities = []

                n_rows = min(len(drone.neighbor_table), self.simulator.n_drones)
                for i in range(n_rows):
                    arrival_time = drone.neighbor_table[i, 6]
                    if arrival_time > 0 and (current_step - arrival_time) < \
                            self.simulator.ExpireTime:
                        valid_neighbors += 1

                        # Năng lượng láng giềng
                        if i < len(self.simulator.drones):
                            nb = self.simulator.drones[i]
                            if nb and hasattr(nb, "residual_energy"):
                                er = nb.residual_energy / max(nb.initial_energy, 1)
                                neighbor_energies.append(er)

                        # Chất lượng liên kết
                        lq = drone.neighbor_table[i, 12]
                        if not np.isnan(lq) and lq >= 0:
                            link_qualities.append(lq)

                avg_neighbor_energy = np.mean(neighbor_energies) \
                    if neighbor_energies else 0.0
                best_link_quality = np.max(link_qualities) \
                    if link_qualities else 0.0

            connectivity = valid_neighbors / max(self.simulator.n_drones, 1)

            state = np.array(
                [
                    x_norm,
                    y_norm,
                    energy_ratio,
                    distance_ratio,
                    dx,
                    dy,
                    speed_norm,
                    connectivity,
                    avg_neighbor_energy,
                    best_link_quality,
                ][: self.state_size],
                dtype=np.float32,
            )

            # log số láng giềng
            self.neighbor_history.append(valid_neighbors)

            return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)

        except Exception as exc:  # phòng thủ
            print("[ERROR] State computation failed:", exc)
            return np.zeros(self.state_size, dtype=np.float32)

    # ----------------- Lọc láng giềng thông minh -----------------

    def get_smart_neighbors(self, opt_neighbors):
        """Chấm điểm láng giềng: gần hơn, năng lượng cao hơn, link tốt, gần depot hơn."""
        if not opt_neighbors:
            return []

        valid_neighbors = []
        neighbor_scores = []

        try:
            current_step = getattr(self.simulator, "cur_step", 0)
            my_pos = self.drone.coords
            rng = max(self.drone.communication_range, 1)
            depot = self.simulator.depot_coordinates

            my_to_depot = math.hypot(depot[0] - my_pos[0], depot[1] - my_pos[1])

            for nb in opt_neighbors:
                if not nb or not hasattr(nb, "identifier"):
                    continue

                # Khoảng cách
                d = util.euclidean_distance(my_pos, nb.coords)
                if d > rng:
                    continue

                score = 0.0
                # Càng gần càng tốt
                score += 0.3 * max(0.0, 1.0 - (d / rng))

                # Năng lượng
                if hasattr(nb, "residual_energy") and hasattr(nb, "initial_energy"):
                    er = nb.residual_energy / max(nb.initial_energy, 1)
                    if er < 0.05:  # quá thấp -> bỏ qua
                        continue
                    score += 0.25 * er
                else:
                    score += 0.125

                # Tươi mới & chất lượng liên kết
                if (hasattr(self.drone, "neighbor_table") and
                        self.drone.neighbor_table is not None and
                        nb.identifier < len(self.drone.neighbor_table)):
                    at = self.drone.neighbor_table[nb.identifier, 6]
                    if at > 0:
                        td = current_step - at
                        if td < self.simulator.ExpireTime:
                            fresh = max(0.0, 1.0 - (td / self.simulator.ExpireTime))
                            score += 0.2 * fresh

                        lq = self.drone.neighbor_table[nb.identifier, 12]
                        if not np.isnan(lq) and lq >= 0:
                            score += 0.25 * min(1.0, lq)

                # Tiến gần depot
                nb_to_depot = math.hypot(depot[0] - nb.coords[0], depot[1] - nb.coords[1])
                if nb_to_depot < my_to_depot:
                    progress = (my_to_depot - nb_to_depot) / max(my_to_depot, 1.0)
                    score += 0.3 * progress

                valid_neighbors.append(nb)
                neighbor_scores.append(score)

        except Exception as exc:
            print("[ERROR] Smart neighbor selection failed:", exc)
            return opt_neighbors  # fallback

        if valid_neighbors and neighbor_scores:
            scored = list(zip(valid_neighbors, neighbor_scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [nb for nb, _ in scored]

        return valid_neighbors

    # ----------------- Chính sách chọn hành động -----------------

    def select_balanced_action(self, state, smart_neighbors):
        """Epsilon-greedy thích nghi; khi kém sẽ chuyển chế độ thận trọng."""
        if not smart_neighbors:
            return None

        try:
            recent_success = np.mean(self.delivery_success_history) \
                if self.delivery_success_history else 0.5

            if recent_success < self.cons_threshold:
                self.conservative_mode = True
                adaptive_eps = max(0.1, self.epsilon * 0.5)
            else:
                self.conservative_mode = False
                adaptive_eps = self.epsilon

            # Khám phá
            if np.random.rand() <= adaptive_eps:
                if self.conservative_mode and len(smart_neighbors) > 1:
                    top_k = min(3, len(smart_neighbors))
                    return random.choice(range(top_k))
                return random.choice(range(len(smart_neighbors)))

            # Khai thác – dùng Q của mạng
            st = torch.as_tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.model(st)

            if q_values.numel() == 0:
                return 0

            # Trộn Q-value với vị trí trong danh sách đã sắp xếp
            action_vals = []
            n_nb = len(smart_neighbors)
            q_flat = q_values
            for i, nb in enumerate(smart_neighbors):
                if q_flat.ndim == 0:
                    base_q = float(q_flat.item())
                elif nb.identifier < q_flat.numel():
                    base_q = float(q_flat[nb.identifier].item())
                else:
                    base_q = float(q_flat.mean().item())

                pos_bonus = (n_nb - i) / max(n_nb, 1)
                action_vals.append((i, 0.8 * base_q + 0.2 * pos_bonus))

            if action_vals:
                return max(action_vals, key=lambda x: x[1])[0]

        except Exception as exc:
            print("[ERROR] Balanced action selection failed:", exc)

        return 0

    # ----------------- Hàm chính chọn relay -----------------

    def relay_selection(self, opt_neighbors, data):
        """
        Chọn relay tối ưu.
        Ghi nhớ state/act để học qua feedback (ACK/timeout) từ BASE_routing.
        """
        self.total_attempts += 1

        try:
            smart_neighbors = self.get_smart_neighbors(opt_neighbors)
            if not smart_neighbors:
                # Fallback an toàn
                return opt_neighbors[0] if opt_neighbors else None

            # Log nhanh mỗi 100 lần
            if self.total_attempts % 100 == 0 and self.neighbor_history:
                avg_nb = float(np.mean(self.neighbor_history))
                succ = float(np.mean(self.delivery_success_history)) \
                    if self.delivery_success_history else 0.0
                if getattr(cfg, "DEBUG", False):
                    print(
                        "[DEBUG] Drone {}: avg_neighbors={:.1f}, "
                        "success_rate={:.3f}, conservative={}"
                        .format(self.drone.identifier, avg_nb, succ,
                                self.conservative_mode)
                    )

            # Tạo state & chọn hành động
            state = self.get_comprehensive_state()
            act = self.select_balanced_action(state, smart_neighbors)
            if act is None or act >= len(smart_neighbors):
                chosen = smart_neighbors[0]
            else:
                chosen = smart_neighbors[act]

            # Lưu để huấn luyện khi có feedback
            self.last_state = state.copy()
            self.last_action = act if act is not None else 0
            self.last_smart_neighbors = smart_neighbors
            self.last_chosen_neighbor_id = chosen.identifier

            return chosen

        except Exception as exc:
            print("[ERROR] Relay selection failed:", exc)
            return opt_neighbors[0] if opt_neighbors else None

    # ----------------- Phần thưởng ưu tiên tỉ lệ giao gói -----------------

    def compute_delivery_focused_reward(self, outcome, neighbor_id):
        """
        outcome:  1 = thành công (đến depot), 0 = trung gian (forward),
                 -1 = thất bại (mất gói/hết TTL)
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
                base = 0.5  # forward nhỏ

            # Điều chỉnh theo tỉ lệ thành công gần đây
            if len(self.delivery_success_history) > 10:
                recent = np.mean(list(self.delivery_success_history)[-20:])
                if recent > 0.8:
                    base += 1.0
                elif recent < 0.6:
                    base -= 0.5

            # Bảo toàn năng lượng
            er = self.drone.residual_energy / max(self.drone.initial_energy, 1)
            if er > 0.7:
                base += 0.3
            elif er < 0.3:
                base -= 0.2

            # Kết nối trung bình
            avg_nb = np.mean(self.neighbor_history) if self.neighbor_history else 1.0
            if avg_nb > 2.0:
                base += 0.2

            # Thành công trong chế độ thận trọng -> thưởng thêm
            if self.conservative_mode and outcome == 1:
                base += 0.5

            return float(max(-5.0, min(7.0, base)))

        except Exception as exc:
            print("[ERROR] Reward computation failed:", exc)
            return 0.0

    # ----------------- Replay học tăng cường (DDQN + ưu tiên) -----------------

    def replay_enhanced(self):
        """Lấy mẫu minibatch: 40% ưu tiên + 60% thường, DDQN target, clip grad."""
        total = len(self.memory) + len(self.priority_memory)
        if total < self.batch_size:
            return

        try:
            minibatch = []

            # 40% từ priority
            if self.priority_memory:
                k = min(int(self.batch_size * 0.4), len(self.priority_memory))
                minibatch.extend(random.sample(list(self.priority_memory), k))

            # Phần còn lại từ memory thường
            remain = self.batch_size - len(minibatch)
            if remain > 0 and self.memory:
                minibatch.extend(
                    random.sample(list(self.memory), min(remain, len(self.memory)))
                )

            for state, action, reward, next_state, done in minibatch:
                if next_state is None or len(next_state) == 0:
                    continue

                # ---- DDQN target ----
                target_val = reward
                if not done:
                    with torch.no_grad():
                        next_state_t = torch.as_tensor(
                            next_state, dtype=torch.float32
                        )
                        # online chọn a*
                        next_q_online = self.model(next_state_t)
                        if next_q_online.ndim == 0:
                            next_a = 0
                        else:
                            next_a = int(torch.argmax(next_q_online).item())

                        # target ước lượng Q(s', a*)
                        next_q_target = self.target_model(next_state_t)
                        if next_q_target.numel() > next_a:
                            target_val = reward + self.gamma * float(
                                next_q_target[next_a].item()
                            )

                # ---- Cập nhật Q(s, a) ----
                state_t = torch.as_tensor(state, dtype=torch.float32)
                current_q = self.model(state_t)

                if current_q.numel() > action:
                    target_q = current_q.clone().detach()
                    target_q = target_q.to(dtype=torch.float32)
                    current_q = current_q.to(dtype=torch.float32)

                    target_q[action] = target_val

                    loss = self.loss_function(current_q, target_q)
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

            # Giảm epsilon (không quá thấp)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        except Exception as exc:
            print("[ERROR] Enhanced replay failed:", exc)

    # ----------------- Học online qua phản hồi từ BASE_routing -----------------

    def feedback(self, outcome, id_j, q_best):
        """
        Được gọi khi nhận ACK/timeout:
          - outcome: 1/0/-1
          - id_j: id hàng xóm được chọn
          - q_best: Q tốt nhất (từ bảng láng giềng) để cập nhật tương thích legacy
        """
        try:
            if self.last_state is None:
                return

            reward = self.compute_delivery_focused_reward(outcome, id_j)
            self.reward_history.append(reward)

            next_state = self.get_comprehensive_state()
            done = (outcome == 1 or outcome == -1)

            exp = (self.last_state, self.last_action, reward, next_state, done)
            priority = (outcome == 1) or (abs(reward) > 2.0)

            if priority:
                self.priority_memory.append(exp)
            else:
                self.memory.append(exp)

            # Học ngay khi đủ mẫu
            if len(self.memory) + len(self.priority_memory) >= self.batch_size:
                self.replay_enhanced()

            # Cập nhật target định kỳ
            self.step_count += 1
            if self.step_count % self.target_update_freq == 0:
                self.update_target_network()

            # Cập nhật tương thích bảng láng giềng (nếu dùng ở nơi khác)
            if id_j < len(self.drone.neighbor_table):
                try:
                    alpha = max(0.1, float(self.drone.neighbor_table[id_j, 10]))
                    gamma = max(0.1, float(self.drone.neighbor_table[id_j, 7]))
                    old_q = float(self.drone.neighbor_table[id_j, 9])

                    if q_best is not None and not np.isnan(q_best):
                        new_q = old_q + alpha * (reward + gamma * q_best - old_q)
                        self.drone.neighbor_table[id_j, 9] = new_q
                except Exception:
                    # Không chặn toàn bộ pipeline nếu bảng thiếu cột
                    pass

        except Exception as exc:
            print("[ERROR] Enhanced feedback failed:", exc)

    # ----------------- Thông số theo dõi -----------------

    def get_performance_metrics(self):
        avg_neighbors = float(np.mean(self.neighbor_history)) \
            if self.neighbor_history else 0.0
        avg_reward = float(np.mean(self.reward_history)) \
            if self.reward_history else 0.0
        delivery_success_rate = float(np.mean(self.delivery_success_history)) \
            if self.delivery_success_history else 0.0

        return {
            "epsilon": float(self.epsilon),
            "avg_neighbors": avg_neighbors,
            "avg_reward": avg_reward,
            "delivery_success_rate": delivery_success_rate,
            "conservative_mode": self.conservative_mode,
            "memory_size": len(self.memory),
            "priority_memory_size": len(self.priority_memory),
            "step_count": self.step_count,
            "success_count": self.success_count,
            "total_attempts": self.total_attempts,
        }


# Tham chiếu nhanh tới các phần liên quan trong dự án:
#   # routing_registry.py: khai báo EnhancedDQN_Routing trong enum
#   # config.py: các tham số EDQN_* và môi trường UAV
#   # BASE_routing.py: callback feedback() và cơ chế ACK/hello
