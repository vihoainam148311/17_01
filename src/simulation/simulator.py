from enum import Enum

from src.drawing import pp_draw
from src.entities.uav_entities import *
from src.simulation.metrics import Metrics
from src.utilities import config, utilities
from src.routing_algorithms.net_routing import MediumDispatcher
from src.routing_algorithms.routing_registry import RoutingAlgorithm  # ✅ Thêm dòng này

from collections import defaultdict
from tqdm import tqdm
import numpy as np
import math
import time


class Simulator:

    def __init__(self,
                 n_drones=10,
                 seed=3,
                 len_simulation=2500,
                 time_step_duration=config.TS_DURATION,
                 env_width=config.ENV_WIDTH,
                 env_height=config.ENV_HEIGHT,
                 drone_com_range=config.COMMUNICATION_RANGE_DRONE,
                 drone_sen_range=config.SENSING_RANGE_DRONE,
                 drone_speed=config.DRONE_SPEED,
                 drone_max_buffer_size=config.DRONE_MAX_BUFFER_SIZE,
                 drone_max_energy=config.DRONE_MAX_ENERGY,
                 drone_retransmission_delta=config.RETRANSMISSION_DELAY,
                 drone_communication_success=config.COMMUNICATION_P_SUCCESS,
                 depot_com_range=config.DEPOT_COMMUNICATION_RANGE,
                 depot_coordinates=config.DEPOT_COO,
                 event_duration=config.EVENTS_DURATION,
                 event_generation_prob=config.P_FEEL_EVENT,
                 event_generation_delay=config.D_FEEL_EVENT,
                 packets_max_ttl=config.PACKETS_MAX_TTL,
                 show_plot=config.PLOT_SIM,
                 routing_algorithm=config.ROUTING_ALGORITHM,
                 communication_error_type=config.CHANNEL_ERROR_TYPE,
                 prob_size_cell_r=config.CELL_PROB_SIZE_R,
                 simulation_name=""):

        self.cur_step = None
        self.drone_com_range = drone_com_range
        self.drone_sen_range = drone_sen_range
        self.drone_speed = drone_speed
        self.drone_max_buffer_size = drone_max_buffer_size
        self.drone_max_energy = drone_max_energy
        self.drone_retransmission_delta = drone_retransmission_delta
        self.drone_communication_success = drone_communication_success
        self.n_drones = n_drones
        self.env_width = env_width
        self.env_height = env_height
        self.depot_com_range = depot_com_range
        self.depot_coordinates = depot_coordinates
        self.len_simulation = len_simulation
        self.time_step_duration = time_step_duration
        self.seed = seed
        self.event_duration = event_duration
        self.event_max_retrasmission = math.ceil(event_duration / drone_retransmission_delta)
        self.event_generation_prob = event_generation_prob
        self.event_generation_delay = event_generation_delay
        self.packets_max_ttl = packets_max_ttl
        self.show_plot = show_plot
        self.routing_algorithm_name = routing_algorithm  # ✅ Lưu tên thuật toán dạng chuỗi
        self.routing_algorithm = RoutingAlgorithm[routing_algorithm]  # ✅ ánh xạ sang lớp tương ứng
        self.communication_error_type = communication_error_type

        self.beta = 0.5
        self.omega = 0.6
        self.ExpireTime = 300

        self.prob_size_cell_r = prob_size_cell_r
        self.prob_size_cell = int(self.drone_com_range * self.prob_size_cell_r)
        self.cell_prob_map = defaultdict(lambda: [0, 0, 0])

        self.sim_save_file = config.SAVE_PLOT_DIR + self.__sim_name()
        self.path_to_depot = None

        self.__set_stepwise_discovery_mode()
        self.metrics = Metrics(self)
        self.__setup_net_dispatcher()
        self.__set_simulation()
        self.__set_metrics()

        self.simulation_name = "simulation-" + utilities.date() + "_" + str(simulation_name) + "_" + str(self.seed) + "_" + str(self.n_drones) + "_" + str(self.routing_algorithm_name)
        self.simulation_test_dir = self.simulation_name + "/"

        self.start = time.time()
        self.event_generator = utilities.EventGenerator(self)
        # Sau dòng 92 trong simulator.py 05_12_2025
        print(f"\n[SIM] Checking routing initialization...")
        for i, drone in enumerate(self.drones):
            routing_type = type(drone.routing).__name__
            print(f"  Drone {i}: {routing_type}")

            if hasattr(drone.routing, 'marl_system'):
                has_marl = drone.routing.marl_system is not None
                print(f"    - MARL system: {has_marl}")

    def test_communication(self):
        """Test communication setup 05_12_2025"""
        print("\n[TEST] Testing communication setup...")
        for i, drone in enumerate(self.drones[:3]):  # Test 3 drones đầu
            neighbors = self.environment.get_uavs_in_range(
                drone,
                max_range=drone.cur_max_comm_distance
            )
            print(f"  Drone {i}:")
            print(f"    Position: {drone.coords}")
            print(f"    Comm range: {drone.cur_max_comm_distance}")
            print(f"    Neighbors found: {len(neighbors)}")
            if neighbors:
                for n in neighbors[:3]:
                    dist = utilities.euclidean_distance(drone.coords, n.coords)
                    print(f"      - Neighbor {n.identifier}: dist={dist:.1f}")
    def __set_stepwise_discovery_mode(self):
        self.stepwise_discovery_mode = False
        if self.routing_algorithm_name == "QL":  # ✅ So sánh theo tên chuỗi
            if config.STEPWISE_NODE_DISCOVERY:
                self.stepwise_discovery_mode = True

    def __setup_net_dispatcher(self):
        self.network_dispatcher = MediumDispatcher(self.metrics)

    def __set_metrics(self):
        self.metrics.info_mission()

    def __set_random_generators(self):
        if self.seed is not None:
            self.rnd_network = np.random.RandomState(self.seed)
            self.rnd_routing = np.random.RandomState(self.seed)
            self.rnd_env = np.random.RandomState(self.seed)
            self.rnd_event = np.random.RandomState(self.seed)

    def __set_simulation(self):
        self.__set_random_generators()
        self.path_manager = utilities.PathManager(config.PATH_FROM_JSON, config.JSONS_PATH_PREFIX, self.seed)
        self.environment = Environment(self.env_width, self.env_height, self)
        self.depot = Depot(self.depot_coordinates, self.depot_com_range, self)
        self.drones = []
        for i in range(self.n_drones):
            self.drones.append(Drone(i, self.path_manager.path(i, self), self.depot, self))

        self.environment.add_drones(self.drones)
        self.environment.add_depot(self.depot)
        self.max_dist_drone_depot = utilities.euclidean_distance(self.depot.coords, (self.env_width, self.env_height))

        if self.show_plot or config.SAVE_PLOT:
            self.draw_manager = pp_draw.PathPlanningDrawer(self.environment, self, borders=True)

    def __sim_name(self):
        return "sim_seed" + str(self.seed) + "drones" + str(self.n_drones) + "_step"

    def __plot(self, cur_step):
        if cur_step % config.SKIP_SIM_STEP != 0:
            return
        if config.WAIT_SIM_STEP > 0:
            time.sleep(config.WAIT_SIM_STEP)
        for drone in self.drones:
            self.draw_manager.draw_drone(drone, cur_step)
        self.draw_manager.draw_depot(self.depot)
        for event in self.environment.active_events:
            self.draw_manager.draw_event(event)
        self.draw_manager.draw_simulation_info(cur_step=cur_step, max_steps=self.len_simulation)
        self.draw_manager.update(show=self.show_plot, save=config.SAVE_PLOT,
                                 filename=self.sim_save_file + str(cur_step) + ".png")

    def run(self):
        for cur_step in range(self.len_simulation):
            self.cur_step = cur_step
            self.network_dispatcher.run_medium(cur_step)
            self.event_generator.handle_events_generation(cur_step, self.drones)

            for drone in self.drones:
                drone.update_packets(cur_step)
                drone.routing(self.drones, self.depot, cur_step)
                drone.move(self.time_step_duration)

            if self.stepwise_discovery_mode:
                self.depot.update_packets(cur_step)
                self.depot.routing(self.drones, self.depot, cur_step)

            if config.ENABLE_PROBABILITIES:
                self.increase_meetings_probs(self.drones, cur_step)

            if self.show_plot or config.SAVE_PLOT:
                self.__plot(cur_step)

        if config.DEBUG:
            print("End of simulation, sim time: " + str((cur_step + 1) * self.time_step_duration) + " sec, #iteration: " + str(cur_step + 1))

    def increase_meetings_probs(self, drones, cur_step):
        cells = set()
        for drone in drones:
            coords = drone.coords
            cell_index = utilities.TraversedCells.coord_to_cell(size_cell=self.prob_size_cell,
                                                                width_area=self.env_width,
                                                                x_pos=coords[0],
                                                                y_pos=coords[1])
            cells.add(int(cell_index[0]))

        for cell, cell_center in utilities.TraversedCells.all_centers(self.env_width, self.env_height, self.prob_size_cell):
            index_cell = int(cell[0])
            old_vals = self.cell_prob_map[index_cell]
            if index_cell in cells:
                old_vals[0] += 1
            old_vals[1] = cur_step + 1
            old_vals[2] = old_vals[0] / max(1, old_vals[1])
            self.cell_prob_map[index_cell] = old_vals

    def close(self):
        self.metrics.compute_energy_metrics(self.drones)
        print("Closing simulation")
        self.print_metrics(plot_id="final")

    def print_metrics(self, plot_id="final"):
        self.metrics.compute_energy_metrics(self.drones)
        self.metrics.other_metrics()
        self.metrics.print_overall_stats()
        self.metrics.routing_hole_count = sum(drone.routing_hole_count for drone in self.drones)
        print(f"Total Routing Hole occurrences: {self.metrics.routing_hole_count}")
        print(f"[DEBUG] Unique packets delivered: {len(self.metrics.drones_packets_to_depot)}")
        print(f"[DEBUG] Total data packets: {self.metrics.all_data_packets_in_simulation}")

    def save_metrics(self, filename_path, save_pickle=False):
        self.metrics.save_as_json(filename_path + ".json")
        if save_pickle:
            self.metrics.save(filename_path + ".pickle")

    def score(self):
        score = round(self.metrics.score(), 2)
        return score

    def get_drone_by_id(self, drone_id):
        for drone in self.drones:
            if drone.identifier == drone_id:
                return drone
        return None  # nếu không tìm thấy drone
