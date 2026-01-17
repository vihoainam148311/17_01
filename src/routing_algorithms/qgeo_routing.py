import numpy as np
import random
from src.routing_algorithms.georouting import GeoRouting
from src.utilities import utilities as util


class QGeoRouting(GeoRouting):
    def __init__(self, drone, simulator):
        super().__init__(drone, simulator)
        self.alpha = 0.2
        self.gamma = 0.4
        self.epsilon = 0.1
        self.pos_precision = 20  # phân đoạn trạng thái

        # Q-table: [pos_state][neighbor_id] → Q-value
        self.q_table = np.random.uniform(low=-1.0, high=0.0, size=(self.pos_precision, simulator.n_drones))
        self.prev_state_action = {}  # Lưu trạng thái và hành động trước để feedback

    def get_state(self):
        dist = util.euclidean_distance(self.drone.coords, self.drone.depot.coords)
        max_dist = self.simulator.env_width + self.simulator.env_height
        pos_idx = min(int((dist / max_dist) * self.pos_precision), self.pos_precision - 1)
        return pos_idx

    def relay_selection(self, opt_neighbors, packet_tuple):
        packet = packet_tuple[0]
        state = self.get_state()
        neighbors = self._normalise_neighbors(opt_neighbors)
        candidates = [nb for _, nb in neighbors]

        if not candidates:
            return None

        if np.random.rand() < self.epsilon:
            chosen = random.choice(candidates)
        else:
            q_vals = [self.q_table[state][nb.identifier] for nb in candidates]
            chosen = candidates[np.argmax(q_vals)]

        # Lưu để feedback sau
        self.prev_state_action[packet.event_ref.identifier] = (state, chosen.identifier)
        return chosen

    def feedback(self, packet_id, outcome, delay):
        if packet_id not in self.prev_state_action:
            return

        state, action_id = self.prev_state_action.pop(packet_id)
        reward = self.compute_reward(outcome, delay)

        current_q = self.q_table[state][action_id]
        next_max = np.max(self.q_table[state])
        self.q_table[state][action_id] += self.alpha * (reward + self.gamma * next_max - current_q)

    def compute_reward(self, outcome, delay):
        if outcome == -1:
            return -2.0
        delay_factor = 1 - delay / (self.simulator.event_duration + 1e-5)
        return 1.0 * delay_factor

    def _greedy(self, neighbors):
        # Ghi đè greedy trong GeoRouting bằng Q-learning
        packet_stub = type('DummyPacket', (), {'event_ref': type('Ev', (), {'identifier': -999})()})()
        return self.relay_selection(neighbors, (packet_stub,))

    def _attach_state(self, packet):
        if not hasattr(packet, "gpsr_state"):
            packet.gpsr_state = type('State', (), {})()
            packet.gpsr_state.mode = "GREEDY"
            packet.gpsr_state.entry_coord = None
            packet.gpsr_state.last_hop = None
            packet.gpsr_state.last_edge_origin = None
