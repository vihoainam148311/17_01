from enum import Enum
from src.routing_algorithms.georouting import GeoRouting
from src.routing_algorithms.q_learning_routing import QMAR
from src.routing_algorithms.random_routing import RandomRouting
from src.routing_algorithms.ai_routing import AIRouting
from src.routing_algorithms.dqn_routing import DQN_Routing
from src.routing_algorithms.qgeo_routing import QGeoRouting
from src.routing_algorithms.enhanced_dqn_routing import EnhancedDQN_Routing
from src.routing_algorithms.ppo_routing import PPO_Routing
from src.routing_algorithms.multi_agent_edqn_routing import MultiAgentEDQN_Routing
from src.routing_algorithms.vdn_qmar_routing import VDN_QMAR
from src.routing_algorithms.qmix_qmar_routing import QMIX_Routing
from src.routing_algorithms.gat_qmix_routing import GAT_QMIX
from src.routing_algorithms.mappo_routing import MAPPO_Routing

# Add to routing algorithms enum
class RoutingAlgorithm(Enum):
    GEO = GeoRouting
    RND = RandomRouting
    QL = QMAR
    AI = AIRouting
    DQN = DQN_Routing
    QGEO = QGeoRouting
    EDQN = EnhancedDQN_Routing
    PPO = PPO_Routing
    MultiAgentEDQN = MultiAgentEDQN_Routing
    VDN_QMAR = VDN_QMAR
    QMIX_QMAR = QMIX_Routing
    GAT_QMIX = GAT_QMIX
    MAPPO = MAPPO_Routing
    @staticmethod
    def keylist():
        return list(map(lambda c: c.name, RoutingAlgorithm))
