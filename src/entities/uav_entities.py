import numpy as np
from src.utilities import config, utilities


class SimulatedEntity:
    def __init__(self, simulator):
        self.simulator = simulator


class Entity(SimulatedEntity):
    def __init__(self, identifier: int, coords: tuple, simulator):
        super().__init__(simulator)
        self.identifier = identifier
        self.coords = coords

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return other.identifier == self.identifier

    def __hash__(self):
        return hash(self.identifier)


class Event(Entity):
    def __init__(self, coords: tuple, current_time: int, simulator, deadline=None):
        super().__init__(id(self), coords, simulator)
        self.current_time = current_time
        self.deadline = current_time + self.simulator.event_duration if deadline is None else deadline
        if not coords == (-1, -1) and not current_time == -1:
            self.simulator.metrics.events.add(self)

    def to_json(self):
        return {
            "coord": self.coords,
            "i_gen": self.current_time,
            "i_dead": self.deadline,
            "id": self.identifier
        }

    def is_expired(self, cur_step):
        return cur_step > self.deadline

    def as_packet(self, time_step_creation, drone):
        pck = DataPacket(time_step_creation, self.simulator, event_ref=self)
        pck.add_hop(drone)
        return pck

    def __repr__(self):
        return f"Ev id:{self.identifier} c:{self.coords}"


class Packet(Entity):
    def __init__(self, time_step_creation, simulator, event_ref: Event = None):
        event_ref_crafted = event_ref if event_ref is not None else Event((-1, -1), -1, simulator)
        super().__init__(id(self), event_ref_crafted.coords, simulator)
        self.time_step_creation = time_step_creation
        self.event_ref = event_ref_crafted
        self.__TTL = -1
        self.__max_TTL = self.simulator.packets_max_ttl
        self.number_retransmission_attempt = 0
        self.last_2_hops = []
        self.hop_count = 0  # Khởi tạo biến đếm số hop

        if event_ref is not None:
            self.simulator.metrics.drones_packets.add(self)
        self.optional_data = None
        self.time_delivery = None
        self.is_move_packet = None

    def __eq__(self, other):
        if not isinstance(other, Packet):
            return False
        return self.event_ref.identifier == other.event_ref.identifier

    def __hash__(self):
        return hash(self.event_ref.identifier)

    def distance_from_depot(self):
        return utilities.euclidean_distance(self.simulator.depot_coordinates, self.coords)

    def age_of_packet(self, cur_step):
        return cur_step - self.time_step_creation

    def to_json(self):
        return {
            "coord": self.coords,
            "i_gen": self.time_step_creation,
            "i_dead": self.event_ref.deadline,
            "id": self.identifier,
            "TTL": self.__TTL,
            "id_event": self.event_ref.identifier
        }

    def add_hop(self, drone):
        if len(self.last_2_hops) == 2:
            self.last_2_hops = self.last_2_hops[1:]
        self.last_2_hops.append(drone)
        self.increase_TTL_hops()
        self.hop_count += 1 #add 08_12

    def increase_TTL_hops(self):
        self.__TTL += 1

    def increase_transmission_attempt(self):
        self.number_retransmission_attempt += 1

    def is_expired(self, cur_step):
        return cur_step > self.event_ref.deadline or self.__TTL >= self.__max_TTL

    def __repr__(self):
        packet_type = str(self.__class__).split(".")[-1].split("'")[0]
        return f"{packet_type} id:{self.identifier} event id:{self.event_ref.identifier} c:{self.coords}"

    def append_optional_data(self, data):
        self.optional_data = data


class DataPacket(Packet):
    def __init__(self, time_step_creation, simulator, event_ref: Event = None):
        super().__init__(time_step_creation, simulator, event_ref)
        self.creation_time = time_step_creation


class ACKPacket(Packet):
    def __init__(self, src_drone, dst_drone, simulator, acked_packet, best_action, time_step_creation=None):
        super().__init__(time_step_creation, simulator, None)
        self.acked_packet = acked_packet
        self.src_drone = src_drone
        self.dst_drone = dst_drone
        self.best_action = best_action


class FeedbackPacket(Packet):
    def __init__(self, src_drone, dst_drone, simulator, best_action, time_step_creation=None):
        super().__init__(time_step_creation, simulator, None)
        self.src_drone = src_drone
        self.dst_drone = dst_drone
        self.best_action = best_action


class HelloPacket:
    def __init__(self, src_drone, cur_pos, energy, speed, next_target, delay, dis_fac, received_pck, sended_ack):
        self.src_drone = src_drone
        self.cur_pos = cur_pos
        self.res_nrg = energy
        self.speed = speed
        self.next_target = next_target
        self.delay = delay
        self.dis_fac = dis_fac
        self.received_pck = received_pck
        self.sended_ack = sended_ack


class Depot(Entity):
    def __init__(self, coords, communication_range, simulator):
        super().__init__(id(self), coords, simulator)
        self.communication_range = communication_range
        self.buffer = list()

    def all_packets(self):
        return self.buffer

    def transfer_notified_packets(self, drone, cur_step):
        packets_to_offload = drone.all_packets()
        self.buffer += packets_to_offload
        for packet_data in packets_to_offload:
            if isinstance(packet_data, tuple):
                if len(packet_data) == 2:
                    pck, _ = packet_data
                else:
                    pck = packet_data[0]
            else:
                pck = packet_data
            if self.simulator.routing_algorithm.name not in ["GEO", "RND"]:
                feedback = 1
                delivery_delay = cur_step - pck.event_ref.current_time
                drone.routing_algorithm.feedback(pck.event_ref.identifier, feedback, 0)
            if pck not in self.simulator.metrics.drones_packets_to_depot:
                self.simulator.metrics.drones_packets_to_depot.add(pck)
                self.simulator.metrics.drones_packets_to_depot_list.append((pck, cur_step))
                pck.time_delivery = cur_step
            else:
                if config.DEBUG:
                    print(f"[DEBUG] Duplicate packet {pck.identifier} detected at depot!")
        drone.empty_buffer()


class Drone(Entity):
    def __init__(self, identifier: int, path: list, depot: Depot, simulator):
        super().__init__(identifier, path[0], simulator)
        self.depot = depot
        self.path = path
        self.speed = self.simulator.drone_speed
        self.sensing_range = self.simulator.drone_sen_range
        self.communication_range = self.simulator.drone_com_range
        self.buffer_max_size = self.simulator.drone_max_buffer_size
        self.residual_energy = self.simulator.drone_max_energy
        self.initial_energy = self.simulator.drone_max_energy
        self.come_back_to_mission = False
        self.last_move_routing = False
        self.neighbor_table = np.zeros((int(self.simulator.n_drones) + 1, 13))
        self.neighbor_table[:, 9] = 0.5
        self.neighbor_table[:, 10] = 0.3
        self.neighbor_table[:, 12] = 0.1
        self.queuing_delay = 1
        self.queuing_delays = []
        self.actual_delays = []
        self.sended_ack = {}
        self.delays = [[] for i in range(self.simulator.n_drones)]
        self.discount_factor = 0
        self.sended_pck = {}
        self.received_pck = {}
        self.received_ack = {}
        self.tightest_event_deadline = None
        self.current_waypoint = 0
        self.buffer = []
        self.distance_from_depot = 0
        self.move_routing = False
        self.routing_algorithm = self.simulator.routing_algorithm.value(self, self.simulator)
        self.last_mission_coords = None
        self.routing_hole_count = 0
        self.wait_steps = 0
        self.feedback_cooldown = 0

    def get_neighbors(self, drones):
        neighbors = []
        for drone in drones:
            if drone.identifier != self.identifier:
                distance = utilities.euclidean_distance(self.coords, drone.coords)
                if distance <= self.communication_range:
                    neighbors.append(drone)
        return neighbors

    def accept_packets(self, packet, src_drone, current_ts):
        if not self.is_known_packet(packet):
            if isinstance(packet, DataPacket):
                packet.add_hop(self) #add 12_08
            self.buffer.append((packet, src_drone, current_ts))
            if config.DEBUG:
                print(
                    f"[DEBUG] Drone {self.identifier} added packet {packet.identifier} to buffer at step {current_ts}")
            neighbors = self.get_neighbors(self.simulator.drones)
            if len(neighbors) < 2:
                if config.DEBUG:
                    print(
                        f"[DEBUG] Drone {self.identifier} received packet {packet.identifier} at step {current_ts} with only {len(neighbors)} neighbors. Potential Routing Hole.")
                self.routing_hole_count += 1

    def routing(self, drones, depot, cur_step):
        self.distance_from_depot = utilities.euclidean_distance(self.depot.coords, self.coords)
        neighbors = self.get_neighbors(drones)

        def get_event_ref(item):
            if isinstance(item, tuple):
                return item[0].event_ref
            elif isinstance(item, DataPacket):
                return item.event_ref
            else:
                if config.DEBUG:
                    print(f"[DEBUG] Invalid buffer item type: {type(item)} in Drone {self.identifier}")
                return None

        valid_neighbors = [
            n for n in neighbors
            if not any(
                get_event_ref(p) == get_event_ref(packet)
                for packet in n.buffer for p in self.buffer
                if get_event_ref(p) is not None and get_event_ref(packet) is not None
            )
        ]

        if len(valid_neighbors) == 0 and self.buffer and self.feedback_cooldown == 0:
            self.routing_hole_count += 1
            if self.wait_steps < 10:
                self.wait_steps += 1
                if config.DEBUG:
                    print(
                        f"[DEBUG] Drone {self.identifier} at step {cur_step} has no valid neighbors. Waiting {self.wait_steps}/10 steps.")
            else:
                self.move_routing = True
                self.wait_steps = 0
                if config.DEBUG:
                    print(f"[DEBUG] Drone {self.identifier} at step {cur_step} activating move_routing after waiting.")
            feedback = FeedbackPacket(self, None, self.simulator, best_action=None, time_step_creation=cur_step)
            for neighbor in neighbors:
                neighbor.accept_packets(feedback, self, cur_step)
            self.feedback_cooldown = 50
        else:
            self.routing_algorithm.routing(depot, drones, cur_step)
            self.wait_steps = 0
            self.feedback_cooldown = max(0, self.feedback_cooldown - 1)

    def update_packets(self, cur_step):
        to_remove_packets = 0
        tmp_buffer = []
        self.tightest_event_deadline = np.nan

        for item in self.buffer:
            if isinstance(item, tuple):
                pck, src, t = item
            elif isinstance(item, DataPacket):
                pck, src, t = item, None, cur_step
                if config.DEBUG:
                    print(
                        f"[DEBUG] Drone {self.identifier} found direct DataPacket in buffer at step {cur_step}. Converting to tuple.")
            else:
                if config.DEBUG:
                    print(f"[DEBUG] Invalid buffer item type: {type(item)} in Drone {self.identifier}")
                continue

            if not pck.is_expired(cur_step):
                tmp_buffer.append((pck, src, t))
                self.tightest_event_deadline = np.nanmin([self.tightest_event_deadline, pck.event_ref.deadline])
            else:
                to_remove_packets += 1
                if config.DEBUG:
                    print(f"[DEBUG] Drone {self.identifier} removed expired packet {pck.identifier} at step {cur_step}")

            if cur_step - t > self.simulator.event_duration / 4:
                self.routing_hole_count += 1
                if config.DEBUG:
                    print(
                        f"[DEBUG] Drone {self.identifier} detected stuck packet {pck.identifier} at step {cur_step}. Possible Routing Hole.")
                self.wait_steps += 1
                if self.wait_steps >= 10:
                    self.move_routing = True
                    self.wait_steps = 0

        self.buffer = tmp_buffer
        if self.buffer_length() == 0:
            self.move_routing = False
            self.wait_steps = 0

    def packet_is_expiring(self, cur_step):
        time_to_depot = self.distance_from_depot / self.speed
        event_time_to_deadline = (self.tightest_event_deadline - cur_step) * self.simulator.time_step_duration
        return event_time_to_deadline - 20 < time_to_depot <= event_time_to_deadline

    def next_move_to_mission_point(self):
        current_waypoint = self.current_waypoint
        if current_waypoint >= len(self.path) - 1:
            current_waypoint = -1
        p0 = self.coords
        p1 = self.path[current_waypoint + 1]
        all_distance = utilities.euclidean_distance(p0, p1)
        distance = self.simulator.time_step_duration * self.speed
        if all_distance == 0 or distance == 0:
            return self.path[current_waypoint]
        t = distance / all_distance
        if t >= 1:
            return self.path[current_waypoint]
        elif t <= 0:
            print("Error move drone, ratio < 0")
            exit(1)
        else:
            return ((1 - t) * p0[0] + t * p1[0]), ((1 - t) * p0[1] + t * p1[1])

    def feel_event(self, cur_step):
        ev = Event(self.coords, cur_step, self.simulator)
        pk = ev.as_packet(cur_step, self)
        if not self.move_routing and not self.come_back_to_mission:
            self.buffer.append((pk, self, cur_step))
            self.simulator.metrics.all_data_packets_in_simulation += 1
            if config.DEBUG:
                print(
                    f"[DEBUG] Drone {self.identifier} added event packet {pk.identifier} to buffer at step {cur_step}")
        else:
            self.simulator.metrics.events_not_listened.add(ev)

    def move(self, time):
        if self.move_routing or self.come_back_to_mission:
            self.simulator.metrics.time_on_active_routing += 1
        if self.move_routing:
            if not self.last_move_routing:
                self.last_mission_coords = self.coords
            self.__move_to_depot(time)
        else:
            if self.last_move_routing:
                self.come_back_to_mission = True
            self.__move_to_mission(time)
            self.simulator.metrics.time_on_mission += 1
        self.last_move_routing = self.move_routing
        self.updateEnergy()

    def updateEnergy(self):
        move_factor = 0.01
        tx_factor = 0.005
        rx_factor = 0.002
        proc_factor = 0.001
        energy_move = move_factor * (self.speed ** 2)
        num_packets_tx = len(self.sended_pck)
        energy_tx = tx_factor * num_packets_tx
        num_packets_rx = len(self.received_pck)
        energy_rx = rx_factor * num_packets_rx
        energy_proc = proc_factor * (num_packets_tx + num_packets_rx)
        total_energy_consumed = energy_move + energy_tx + energy_rx + energy_proc
        self.residual_energy -= total_energy_consumed
        if self.residual_energy < 0:
            self.residual_energy = 0

    def is_full(self):
        return self.buffer_length() == self.buffer_max_size

    def is_known_packet(self, packet: DataPacket):
        for item in self.buffer:
            if isinstance(item, tuple):
                pk = item[0]
            elif isinstance(item, DataPacket):
                pk = item
            else:
                continue
            if pk.event_ref == packet.event_ref:
                return True
        return False

    def empty_buffer(self):
        self.buffer = []

    def all_packets(self):
        return self.buffer

    def buffer_length(self):
        return len(self.buffer)

    def updateACK(self):
        for key in list(self.sended_pck.keys()):
            if self.simulator.cur_step > self.sended_pck[key][0] + 200:
                self.routing_algorithm.feedback(key, -1, self.sended_pck[key][1],
                                                self.neighbor_table[self.sended_pck[key][12]])
                del self.sended_pck[key]

    def remove_packets(self, packets):
        copy = self.buffer.copy()
        for packet in packets:
            for i in range(len(copy)):
                if isinstance(copy[i], tuple):
                    pck, d, f = copy[i]
                elif isinstance(copy[i], DataPacket):
                    pck, d, f = copy[i], None, None
                else:
                    continue
                if packet == pck:
                    j = self.buffer.index(copy[i])
                    del self.buffer[j]
                    self.sended_pck.pop(packet.event_ref.identifier, None)
                    if config.DEBUG:
                        print(f"[DEBUG] Removed packet id: {packet.identifier} from Drone {self.identifier}")

    def next_target(self):
        if self.move_routing:
            return self.depot.coords
        elif self.come_back_to_mission:
            return self.last_mission_coords
        else:
            if self.current_waypoint >= len(self.path) - 1:
                return self.path[0]
            else:
                return self.path[self.current_waypoint + 1]

    def __move_to_mission(self, time):
        if self.current_waypoint >= len(self.path) - 1:
            self.current_waypoint = -1
        p0 = self.coords
        if self.come_back_to_mission:
            p1 = self.last_mission_coords
        else:
            p1 = self.path[self.current_waypoint + 1]
        all_distance = utilities.euclidean_distance(p0, p1)
        distance = time * self.speed
        if all_distance == 0 or distance == 0:
            self.__update_position(p1)
            return
        t = distance / all_distance
        if t >= 1:
            self.__update_position(p1)
        elif t <= 0:
            print("Error move drone, ratio < 0")
            exit(1)
        else:
            self.coords = (((1 - t) * p0[0] + t * p1[0]), ((1 - t) * p0[1] + t * p1[1]))

    def __update_position(self, p1):
        if self.come_back_to_mission:
            self.come_back_to_mission = False
            self.coords = p1
        else:
            self.current_waypoint += 1
            self.coords = self.path[self.current_waypoint]

    def __move_to_depot(self, time):
        p0 = self.coords
        p1 = self.depot.coords
        all_distance = utilities.euclidean_distance(p0, p1)
        distance = time * self.speed
        if all_distance == 0:
            self.move_routing = False
            return
        t = distance / all_distance
        if t >= 1:
            self.coords = p1
        elif t <= 0:
            print("Error routing move drone, ratio < 0")
            exit(1)
        else:
            self.coords = (((1 - t) * p0[0] + t * p1[0]), ((1 - t) * p0[1] + t * p1[1]))

    def __repr__(self):
        return f"Drone {self.identifier}"

    def __hash__(self):
        return hash(self.identifier)


class Environment(SimulatedEntity):
    def __init__(self, width, height, simulator):
        super().__init__(simulator)
        self.depot = None
        self.drones = None
        self.width = width
        self.height = height
        self.event_generator = EventGenerator(height, width, simulator)
        self.active_events = []

    def add_drones(self, drones: list):
        self.drones = drones

    def add_depot(self, depot: Depot):
        self.depot = depot


class EventGenerator(SimulatedEntity):
    def __init__(self, height, width, simulator):
        super().__init__(simulator)
        self.height = height
        self.width = width

    def uniform_event_generator(self):
        x = self.simulator.rnd_env.randint(0, self.height)
        y = self.simulator.rnd_env.randint(0, self.width)
        return x, y

    def poisson_event_generator(self):
        pass

