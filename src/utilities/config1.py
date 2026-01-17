# config.py - Realistic UAV Network Simulation _EDQN

# ----------------------- PATH DRONES -----------------------------------------#
CIRCLE_PATH = False
DEMO_PATH = False
PATH_FROM_JSON = False
JSONS_PATH_PREFIX = "data/tours/RANDOM_missions{}.json"
RANDOM_STEPS = [250, 500, 700, 960, 1100, 1400, 2000]
RANDOM_START_POINT = True

# ------------------------------- CONSTANTS ------------------------------- #
DEBUG = False  # Enable for better debugging
EXPERIMENTS_DIR = "data/evaluation_tests/"
PLOT_SIM = False  # Keep false for faster simulation
WAIT_SIM_STEP = 0
SKIP_SIM_STEP = 5
DRAW_SIZE = 700
IS_SHOW_NEXT_TARGET_VEC = True
SAVE_PLOT = False
SAVE_PLOT_DIR = "data/plots/"

# ================= REALISTIC UAV SIMULATION PARAMETERS =================

# Environment: 1.8km x 1.8km (realistic urban UAV operation area)
# Scale: 1 pixel = 1 meter
SIM_DURATION = 10000  # Reduced for faster testing
TS_DURATION = 0.2  # 50ms time steps (more realistic for UAV control)
SEED = 10
N_DRONES = 25  # Increased to compensate for realistic comm range
ENV_WIDTH = 1800  # 1.8km x 1.8km
ENV_HEIGHT = 1800

# Event generation parameters
EVENTS_DURATION = 25000  # Proportional to sim duration
D_FEEL_EVENT = 20  # Event every 1 second (20 * 0.05 = 1s)
P_FEEL_EVENT = 0.05  # More realistic event probability

# ================= REALISTIC COMMUNICATION PARAMETERS =================

# Commercial UAV scenario (DJI Phantom class)
COMMUNICATION_RANGE_DRONE = 300  # 350m - realistic for commercial UAVs
DEPOT_COMMUNICATION_RANGE = 400  # Depot has better antenna/power

# Alternative scenarios (comment/uncomment as needed):

# Consumer UAV scenario (DJI Mini class)
# COMMUNICATION_RANGE_DRONE = 200   # 200m
# DEPOT_COMMUNICATION_RANGE = 300

# Professional UAV scenario
# COMMUNICATION_RANGE_DRONE = 600   # 600m
# DEPOT_COMMUNICATION_RANGE = 800

print(f"[CONFIG] UAV Communication Range: {COMMUNICATION_RANGE_DRONE}m")
print(f"[CONFIG] Coverage: {(COMMUNICATION_RANGE_DRONE * 2 / ENV_WIDTH) * 100:.1f}% of environment width")

# ================= UAV PHYSICAL PARAMETERS =================

SENSING_RANGE_DRONE = 0
DRONE_SPEED = 8  # 8 m/s = 28.8 km/h (realistic cruise speed)
DRONE_MAX_BUFFER_SIZE = 100  # Smaller, more realistic buffer
DRONE_MAX_ENERGY = 100000  # Battery capacity

# Depot location - centered for optimal coverage
DEPOT_COO = (ENV_WIDTH // 2, ENV_HEIGHT // 2)  # (900, 900)

# ================= ROUTING ALGORITHM PARAMETERS =================

# Enhanced DQN as primary algorithm
ROUTING_ALGORITHM = "EDQN"  # Enhanced DQN


class ChannelError:
    UNIFORM = 1
    GAUSSIAN = 2
    NO_ERROR = 3

    @staticmethod
    def keylist():
        return ["UNIFORM", "GAUSSIAN", "NO_ERROR"]


# Realistic channel conditions
CHANNEL_ERROR_TYPE = ChannelError.GAUSSIAN  # More realistic than NO_ERROR
STEPWISE_NODE_DISCOVERY = False
COMMUNICATION_P_SUCCESS = 0.98  # 90% success rate (realistic outdoor)
GUASSIAN_SCALE = 0.85  # Moderate signal variation

# ================= NETWORK PROTOCOL PARAMETERS =================

PACKETS_MAX_TTL = 150  # Shorter TTL for faster decisions
RETRANSMISSION_DELAY = 20  # 1 second retransmission (20 * 0.05)

# Hello protocol - optimized for UAV mobility
HELLO_DELAY = 10  # Hello every 0.5 seconds (10 * 0.05)
RECEPTION_GRANTED = 0.95
LIL_DELTA = 1
OLD_HELLO_PACKET = 50

# Neighbor management
ExpireTime = 100  # 5 seconds neighbor expiry (100 * 0.05)

# ================= ENHANCED DQN SPECIFIC PARAMETERS =================

# Enhanced DQN tuning parameters
EDQN_LEARNING_RATE = 0.0003
EDQN_BATCH_SIZE = 64
EDQN_MEMORY_SIZE = 3000
EDQN_EPSILON_START = 0.95
EDQN_EPSILON_MIN = 0.02
EDQN_EPSILON_DECAY = 0.9995
EDQN_GAMMA = 0.98
EDQN_TARGET_UPDATE_FREQ = 100

# Conservative mode threshold
EDQN_CONSERVATIVE_THRESHOLD = 0.7  # Switch to conservative when delivery rate < 70%

# ================= DIRECTORY PATHS =================

ROOT_EVALUATION_DATA = "data/evaluation_tests/"
NN_MODEL_PATH = "data/nnmodels/"

# --------------- CELL PROBABILITIES -------------- #
CELL_PROB_SIZE_R = 1.875
ENABLE_PROBABILITIES = False


# ================= NETWORK ANALYSIS =================

def analyze_network_parameters():
    """Analyze network connectivity with current parameters"""
    import math

    # Calculate expected connectivity
    area = ENV_WIDTH * ENV_HEIGHT
    comm_area = math.pi * (COMMUNICATION_RANGE_DRONE ** 2)

    # Probability any two drones can communicate
    comm_prob = comm_area / area

    # Expected neighbors per drone
    expected_neighbors = (N_DRONES - 1) * comm_prob

    print(f"\n[NETWORK ANALYSIS]")
    print(f"Environment: {ENV_WIDTH}m x {ENV_HEIGHT}m ({area:,} m²)")
    print(f"Communication coverage per UAV: {comm_area:,.0f} m²")
    print(f"Expected neighbors per UAV: {expected_neighbors:.2f}")

    # Connectivity assessment
    if expected_neighbors < 1.5:
        print(f"⚠️  Low connectivity - challenging for routing")
        print(f"   Consider: increasing comm range or adding more UAVs")
    elif expected_neighbors > 4.0:
        print(f"✅ High connectivity - good for routing algorithms")
    else:
        print(f"✅ Moderate connectivity - realistic UAV network")

    # Time analysis
    sim_time_real = SIM_DURATION * TS_DURATION
    print(f"\nSimulation time: {sim_time_real:.1f} seconds ({sim_time_real / 60:.1f} minutes)")

    return expected_neighbors


def get_scenario_configs():
    """Get different realistic UAV scenario configurations"""
    scenarios = {
        'consumer': {
            'comm_range': 200,
            'depot_range': 300,
            'n_drones': 30,
            'speed': 5,
            'comm_success': 0.9,
            'description': 'Consumer UAVs (DJI Mini, etc.) - Dense deployment'
        },
        'commercial': {
            'comm_range': 350,
            'depot_range': 500,
            'n_drones': 25,
            'speed': 8,
            'comm_success': 0.9,
            'description': 'Commercial UAVs (DJI Phantom, etc.) - Current config'
        },
        'professional': {
            'comm_range': 500,
            'depot_range': 700,
            'n_drones': 20,
            'speed': 12,
            'comm_success': 0.95,
            'description': 'Professional UAVs - Extended range'
        },
        'research': {
            'comm_range': 800,
            'depot_range': 1000,
            'n_drones': 15,
            'speed': 15,
            'comm_success': 0.98,
            'description': 'Research UAVs - Advanced networking'
        }
    }
    return scenarios


# Run analysis on import
if __name__ == "__main__":
    print("=== REALISTIC UAV NETWORK SIMULATION CONFIG ===")
    analyze_network_parameters()

    print(f"\n=== AVAILABLE SCENARIOS ===")
    scenarios = get_scenario_configs()
    for name, config in scenarios.items():
        print(f"{name.upper()}: {config['description']}")
        print(f"  Range: {config['comm_range']}m, Drones: {config['n_drones']}, Speed: {config['speed']}m/s")
else:
    # Quick analysis when imported
    expected_neighbors = analyze_network_parameters()
    if expected_neighbors < 1.0:
        print(f"⚠️  WARNING: Very low expected connectivity ({expected_neighbors:.2f} neighbors/UAV)")
        print(f"   Consider adjusting parameters for better network performance")