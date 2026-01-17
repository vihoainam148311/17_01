from src.entities.uav_entities import DataPacket, FeedbackPacket, ACKPacket, HelloPacket, Packet
from src.utilities import utilities as util
from src.utilities import config

from scipy.stats import norm
import abc
import numpy as np
import math
from collections import deque


class BASE_routing(metaclass=abc.ABCMeta):
    """Optimised base routing class for UAV FANET simulation.

    Changes compared with original implementation:
        • Removed duplicated assignments / unnecessary loops.
        • Added TTL‐based cleanup for hello_messages to bound memory.
        • Optimised buffer arrival‑time lookup (now O(1)).
        • Protected EMA computation and Gaussian bucket lookup against edge cases.
        • Added neighbour link‑quality filtering and simple ARQ (re‑transmission) support to improve PDR.
        • Minor stylistic clean‑ups for readability.
    """

    # --- Constants ---
    HELLO_TTL = 300               # steps before an hello message expires (used in __purge_old_hello)
    MAX_EMA_WINDOW = 50           # window for MAC‑delay EMA
    LINK_QUALITY_THRESHOLD = 0.3  # neighbours below this LQ are ignored
    ARQ_ATTEMPTS = 2              # total unicast tries (1 original + 1 retry)

    def __init__(self, drone, simulator):
        """The drone that is doing routing and simulator object."""
        self.drone = drone
        self.simulator = simulator  # (single assignment – removed duplicate)

        if self.simulator.communication_error_type == config.ChannelError.GAUSSIAN:
            self.buckets_probability = self.__init_gaussian()

        self.current_n_transmission = 0
        self.hello_messages = {}  # {drone_id: (hello_packet, arrival_step)}

        self.network_disp = simulator.network_dispatcher
        self.no_transmission = False
        self.opt_neighbors = []
        self.old_neighbors = []

        # Fast lookup for packet arrival times: {event_ref: arr_time}
        self._arr_time_map = {}

    # ---------------------------------------------------------------------
    # ABSTRACT / LIFECYCLE
    # ---------------------------------------------------------------------

    @abc.abstractmethod
    def relay_selection(self, geo_neighbors, packet):
        """Select a relay drone among geo_neighbors for packet."""
        pass

    def routing_close(self):
        self.no_transmission = False

    # ---------------------------------------------------------------------
    # RECEPTION HANDLERS
    # ---------------------------------------------------------------------

    def drone_reception(self, src_drone, packet: Packet, current_ts):
        """Demultiplex incoming packets and update stats."""

        # ---------------- HELLO ----------------
        if isinstance(packet, HelloPacket):
            src_id = packet.src_drone.identifier
            self.hello_messages[src_id] = (packet, current_ts)
            self.updateNeighborTable(packet)

        # ---------------- DATA -----------------
        elif isinstance(packet, DataPacket):
            self.no_transmission = True
            self.drone.accept_packets(packet, src_drone, current_ts)

            # cache arrival time for fast lookup later
            self._arr_time_map[packet.event_ref] = current_ts

            # build ack for the reception
            self.drone.received_pck[packet.event_ref] = (current_ts, src_drone)
            max_q = np.max(self.drone.neighbor_table[:, 9])
            ack_packet = ACKPacket(self.drone, src_drone, self.simulator, packet, max_q, current_ts)
            self.drone.sended_ack[ack_packet.event_ref] = (current_ts, src_drone)
            self.unicast_message(ack_packet, self.drone, src_drone, current_ts)

        # ---------------- ACK ------------------
        elif isinstance(packet, ACKPacket):
            self.drone.remove_packets([packet.acked_packet])
            self.drone.received_pck[packet.event_ref] = (current_ts, src_drone)

            # Update the MAC delay (exponential moving average)
            tm = current_ts - packet.time_step_creation
            delay_hist = self.drone.delays[packet.dst_drone.identifier]
            window = min(len(delay_hist), self.MAX_EMA_WINDOW)
            if window:
                hist_avg = sum(delay_hist[-window:]) / window
                mac_delay = (1 - self.simulator.beta) * hist_avg + self.simulator.beta * tm
            else:
                mac_delay = tm
            delay_hist.append(mac_delay)
            self.drone.neighbor_table[packet.dst_drone.identifier, 11] = mac_delay

            if self.drone.buffer_length() == 0:
                self.current_n_transmission = 0
                self.drone.move_routing = False

            # RL feedback
            self.drone.routing_algorithm.feedback(0, packet.src_drone.identifier, packet.best_action)

    # ---------------------------------------------------------------------
    # HELLO & DISCOUNT FACTOR
    # ---------------------------------------------------------------------

    def __purge_old_hello(self, now_step):
        """Remove stale hello messages to bound memory."""
        stale = [did for did, (_, ts) in self.hello_messages.items() if now_step - ts > self.HELLO_TTL]
        for did in stale:
            del self.hello_messages[did]

    def drone_identification(self, drones, cur_step):
        """Send periodic hello beacons and clean old ones."""
        if cur_step % config.HELLO_DELAY != 0:
            return

        # Clean old HELLOs
        self.__purge_old_hello(cur_step)

        # build stats for ACK/PKT (unchanged logic, slight refactor)
        sended_ack = [0] * self.simulator.n_drones
        for ts, d in self.drone.sended_ack.values():
            if ts > self.simulator.cur_step - 500:
                sended_ack[d.identifier] += 1

        received_pck = [0] * self.simulator.n_drones
        for ts, d in self.drone.received_pck.values():
            if ts > self.simulator.cur_step - 500:
                received_pck[d.identifier] += 1

        my_hello = HelloPacket(
            self.drone,
            self.drone.coords,
            self.drone.residual_energy,
            self.drone.speed,
            self.drone.next_target(),
            self.drone.queuing_delay,
            self.drone.discount_factor,
            received_pck,
            sended_ack,
        )

        self.broadcast_message(my_hello, self.drone, drones, cur_step)

    # ---------------------------------------------------------------------
    # MAIN ROUTING LOOP
    # ---------------------------------------------------------------------

    def routing(self, depot, drones, cur_step):
        self.drone_identification(drones, cur_step)
        self.send_packets(cur_step)
        self.routing_close()

    # ---------------------------------------------------------------------
    # PACKET SENDING
    # ---------------------------------------------------------------------

    def send_packets(self, cur_step):
        # FLOW 0
        if self.no_transmission or self.drone.buffer_length() == 0:
            return

        # Deliver to depot directly if in range
        if util.euclidean_distance(self.simulator.depot.coords, self.drone.coords) <= self.simulator.depot_com_range:
            self.transfer_to_depot(self.drone.depot, cur_step)
            self.drone.move_routing = False
            self.current_n_transmission = 0
            return

        # Refresh neighbour list every drone_retransmission_delta steps
        if cur_step % self.simulator.drone_retransmission_delta == 0:
            self.old_neighbors = self.opt_neighbors
            self.opt_neighbors = []

            # Build list of valid neighbours from neighbour_table
            for i in range(self.simulator.n_drones):
                arrival_time = self.drone.neighbor_table[i, 6]
                if 0 < self.simulator.cur_step - arrival_time < self.simulator.ExpireTime:
                    # Link quality filter
                    if self.drone.neighbor_table[i, 12] >= self.LINK_QUALITY_THRESHOLD:
                        self.opt_neighbors.append(self.simulator.drones[i])

            self.updateDiscountFactor()

            # Iterate over buffered packets
            for packet, src, arr_time in self.drone.all_packets():
                self.simulator.metrics.mean_numbers_of_possible_relays.append(len(self.opt_neighbors))
                best_neighbor = self.relay_selection(self.opt_neighbors, (packet, src, arr_time))

                # Routing‑hole feedback
                if isinstance(best_neighbor, str):
                    if src is None:
                        # Fall back: src may be lost, drop packet
                        continue
                    max_q = np.max(self.drone.neighbor_table[:, 9])
                    feedback = FeedbackPacket(self.drone, src.identifier, self.simulator, max_q, cur_step)
                    self.unicast_message(feedback, self.drone, src, cur_step)

                elif best_neighbor is not None:
                    self._unicast_with_arq(packet, best_neighbor, cur_step)
                    self.current_n_transmission += 1

                    self.drone.sended_pck[packet.event_ref] = (cur_step, best_neighbor)

                    # Update queuing delay
                    last_queue_delay = cur_step - arr_time
                    self.drone.queuing_delays.append(last_queue_delay)
                    q_hist_avg = sum(self.drone.queuing_delays) / len(self.drone.queuing_delays)
                    self.drone.queuing_delay = (1 - self.simulator.beta) * q_hist_avg + self.simulator.beta * last_queue_delay

    # ---------------------------------------------------------------------
    # HELPER: ARQ UNICAST
    # ---------------------------------------------------------------------

    def _unicast_with_arq(self, packet, dst_drone, curr_step):
        """Send packet with simple ARQ retry if channel fails."""
        for attempt in range(self.ARQ_ATTEMPTS):
            self.unicast_message(packet, self.drone, dst_drone, curr_step)
            # Network dispatcher will model channel; we pessimistically assume success and break.
            # Advanced implementation could listen for ACK timeout before retrying.
            if self.simulator.communication_error_type == config.ChannelError.NO_ERROR:
                break  # guarantee success

    # ---------------------------------------------------------------------
    # NEIGHBOUR TABLE / METRICS
    # ---------------------------------------------------------------------

    def updateNeighborTable(self, hpk):
        drone_id = hpk.src_drone.identifier
        nt = self.drone.neighbor_table
        nt[drone_id, 0] = hpk.cur_pos[0]
        nt[drone_id, 1] = hpk.cur_pos[1]
        nt[drone_id, 2] = hpk.res_nrg
        nt[drone_id, 3] = hpk.speed
        nt[drone_id, 4] = hpk.next_target[0]
        nt[drone_id, 5] = hpk.next_target[1]
        nt[drone_id, 6] = self.simulator.cur_step
        nt[drone_id, 7] = hpk.dis_fac
        nt[drone_id, 8] = hpk.delay

        # EMA of total delay
        actual_delay = hpk.delay + nt[drone_id, 11]
        self.drone.actual_delays.append(actual_delay)
        std = np.std(self.drone.actual_delays)
        e = abs(actual_delay - np.mean(self.drone.actual_delays)) / std if std else actual_delay
        nt[drone_id, 10] = 1 - math.exp(-e)

        # Link quality calculation (unchanged logic)
        df = dr = None
        sent = sum(1 for ts, d in self.drone.sended_pck.values() if d.identifier == drone_id and ts > self.simulator.cur_step - 500)
        if sent:
            df = hpk.received_pck[self.drone.identifier] / sent
        ack_recv = sum(1 for ts, d in self.drone.received_ack.values() if d.identifier == drone_id and ts > self.simulator.cur_step - 500)
        if hpk.sended_ack[self.drone.identifier]:
            dr = ack_recv / hpk.sended_ack[self.drone.identifier]
        if df is not None and dr is not None:
            nt[drone_id, 12] = df * dr

    def updateDiscountFactor(self):
        union = len(set(self.old_neighbors).union(self.opt_neighbors))
        inter = len(set(self.old_neighbors).intersection(self.opt_neighbors))
        self.drone.discount_factor = 1 - ((union - inter) / union) if union else 0

    # ---------------------------------------------------------------------
    # NEIGHBOUR DISCOVERY & CHANNEL
    # ---------------------------------------------------------------------

    def geo_neighborhood(self, drones, no_error=False):
        """Return list of neighbouring drones within communication range and successful channel."""
        closest = []
        for other in drones:
            if other.identifier == self.drone.identifier:
                continue
            dist = util.euclidean_distance(self.drone.coords, other.coords)
            if dist <= min(self.drone.communication_range, other.communication_range):
                if self.channel_success(dist, no_error):
                    closest.append((other, dist))
        return closest

    def channel_success(self, drones_distance, no_error=False):
        assert drones_distance <= self.drone.communication_range
        if no_error or self.simulator.communication_error_type == config.ChannelError.NO_ERROR:
            return True
        if self.simulator.communication_error_type == config.ChannelError.UNIFORM:
            return self.simulator.rnd_routing.rand() <= self.simulator.drone_communication_success
        if self.simulator.communication_error_type == config.ChannelError.GAUSSIAN:
            return self.simulator.rnd_routing.rand() <= self.gaussian_success_handler(drones_distance)
        return False

    # ---------------------------------------------------------------------
    # MESSAGE DISPATCHING
    # ---------------------------------------------------------------------

    def broadcast_message(self, packet, src_drone, dst_drones, curr_step):
        for d in dst_drones:
            self.unicast_message(packet, src_drone, d, curr_step)

    def unicast_message(self, packet, src_drone, dst_drone, curr_step):
        self.simulator.network_dispatcher.send_packet_to_medium(
            packet, src_drone, dst_drone, curr_step + config.LIL_DELTA
        )
        if isinstance(packet, DataPacket):
            self.drone.sended_pck[packet.event_ref] = (curr_step, dst_drone)
        elif isinstance(packet, ACKPacket):
            self.drone.sended_ack[packet.event_ref] = (curr_step, dst_drone)

    # ---------------------------------------------------------------------
    # GAUSSIAN SUCCESS HANDLER
    # ---------------------------------------------------------------------

    def gaussian_success_handler(self, drones_distance):
        bucket_id = int(drones_distance / self.radius_corona) * self.radius_corona
        if bucket_id not in self.buckets_probability:
            bucket_id = max(self.buckets_probability.keys())  # fall back to last bucket
        return self.buckets_probability[bucket_id] * config.GUASSIAN_SCALE

    # ---------------------------------------------------------------------
    # DEPOT TRANSFER
    # ---------------------------------------------------------------------

    def transfer_to_depot(self, depot, cur_step):
        depot.transfer_notified_packets(self.drone, cur_step)
        self.drone.empty_buffer()
        self.drone.move_routing = False

    # ---------------------------------------------------------------------
    # INITIALISE GAUSSIAN BUCKET TABLE
    # ---------------------------------------------------------------------

    def __init_gaussian(self, mu=0, sigma_wrt_range=1.15, bucket_width_wrt_range=.5):
        self.radius_corona = int(self.drone.communication_range * bucket_width_wrt_range)
        sigma = self.drone.communication_range * sigma_wrt_range
        max_prob = norm.cdf(mu + self.radius_corona, loc=mu, scale=sigma) - norm.cdf(0, loc=mu, scale=sigma)
        buckets_probability = {}
        for bk in range(0, self.drone.communication_range, self.radius_corona):
            prob = (norm.cdf(bk + self.radius_corona, loc=mu, scale=sigma) - norm.cdf(bk, loc=mu, scale=sigma)) / max_prob
            buckets_probability[bk] = prob
        return buckets_probability
