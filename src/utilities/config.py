# config.py

# ----------------------- PATH DRONES -----------------------------------------#
CIRCLE_PATH = False
DEMO_PATH = False
PATH_FROM_JSON = False
JSONS_PATH_PREFIX = "data/tours/RANDOM_missions{}.json"
RANDOM_STEPS = [250, 500, 700, 960, 1100, 1400, 2000]
RANDOM_START_POINT = True

# ------------------------------- CONSTANTS ------------------------------- #
DEBUG = False
EXPERIMENTS_DIR = "data/evaluation_tests/"
PLOT_SIM = False
WAIT_SIM_STEP = 0
SKIP_SIM_STEP = 5
DRAW_SIZE = 700
IS_SHOW_NEXT_TARGET_VEC = True
SAVE_PLOT = False
SAVE_PLOT_DIR = "data/plots/"

# ----------------------------- SIMULATION PARAMS. ---------------------------- #
SIM_DURATION = 10000
TS_DURATION = 0.2
SEED = 10
N_DRONES = 50
ENV_WIDTH = 1800
ENV_HEIGHT = 1800

EVENTS_DURATION = 20000
D_FEEL_EVENT = 8
P_FEEL_EVENT = 1

COMMUNICATION_RANGE_DRONE = 300
SENSING_RANGE_DRONE = 0
DRONE_SPEED = 10
DRONE_MAX_BUFFER_SIZE = 1000
DRONE_MAX_ENERGY = 100000

DEPOT_COMMUNICATION_RANGE = 350
DEPOT_COO = (950, 950)

# ------------------------------- ROUTING PARAMS. ------------------------------- #

# ❗❗ Chỉ định tên giao thức định tuyến dưới dạng chuỗi
# Các giá trị có thể: "GEO", "RND", "QL", "AI", "DQN", "QGEO", "EDQN", "PPO" "MADQN"  "VDN_QMAR" QMIX_QMAR GAT_QMIX" MultiAgentEDQN "MAPPO"
ROUTING_ALGORITHM = "GEO"  #

class ChannelError:
    UNIFORM = 1
    GAUSSIAN = 2
    NO_ERROR = 3

    @staticmethod
    def keylist():
        return ["UNIFORM", "GAUSSIAN", "NO_ERROR"]

CHANNEL_ERROR_TYPE = ChannelError.GAUSSIAN

STEPWISE_NODE_DISCOVERY = False
COMMUNICATION_P_SUCCESS = 0.98
GUASSIAN_SCALE = .9
PACKETS_MAX_TTL = 200
RETRANSMISSION_DELAY = 10

# ------------------------------------------- ROUTING MISC --------------------------------- #
HELLO_DELAY = 20
RECEPTION_GRANTED = 0.95
LIL_DELTA = 1
OLD_HELLO_PACKET = 50

ROOT_EVALUATION_DATA = "data/evaluation_tests/"
NN_MODEL_PATH = "data/nnmodels/"

# --------------- new cell probabilities -------------- #
CELL_PROB_SIZE_R = 1.875
ENABLE_PROBABILITIES = False

# Multi-Agent EDQN Parameters
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
MARL_BROADCAST_INTERVAL = 5  # steps giữa các lần broadcast

