import math
from typing import List, Any, Optional, Tuple

import numpy as np

from src.routing_algorithms.BASE_routing import BASE_routing
from src.entities.uav_entities import DataPacket
import src.utilities.utilities as util


class GPSRPacketState:
    """Light‑weight helper kept inside each DataPacket for GPSR mode switching"""

    __slots__ = ("mode", "entry_coord", "last_hop", "last_edge_origin")

    def __init__(self):
        self.mode: str = "GREEDY"  # or "PERIMETER"
        self.entry_coord: Optional[Tuple[float, float]] = None
        self.last_hop: Optional[int] = None
        self.last_edge_origin: Optional[Tuple[float, float]] = None


class GeoRouting(BASE_routing):
    """GPSR routing (Greedy + Perimeter with RNG/GG planarisation).

    This class is drop‑in replacement for the original GeoRouting, keeping the same
    constructor signature.  It also exposes a no‑op `feedback` method so that
    BASE_routing can call it uniformly across different algorithms.
    """

    # ------------------------ constructor -----------------------------

    def __init__(self, drone, simulator, planar_method: str = "RNG"):
        super().__init__(drone, simulator)
        assert planar_method in {"RNG", "GG"}, "Unsupported planarisation method"
        self.planar_method = planar_method

    # ----------------- API required by BASE_routing ------------------

    def relay_selection(self, opt_neighbors: List[Any], packet_tuple):
        """Return next relay drone according to GPSR rules.

        `opt_neighbors` can be either a list of Drone objects (legacy BASE_routing)
        or a list of tuples `(hello_packet, drone)` – we normalise internally.
        """
        packet: DataPacket = packet_tuple[0]
        self._attach_state(packet)
        st: GPSRPacketState = packet.gpsr_state  # type: ignore

        neighbors = self._normalise_neighbors(opt_neighbors)
        if not neighbors:
            return None

        # ---------- GREEDY ----------
        if st.mode == "GREEDY":
            relay = self._greedy(neighbors)
            if relay:
                return relay
            # Greedy failed – enter perimeter mode
            st.mode = "PERIMETER"
            st.entry_coord = self.drone.coords
            st.last_hop = self.drone.identifier
            st.last_edge_origin = self.drone.coords

        # ---------- PERIMETER ----------
        planar = self._planarise(neighbors)
        return self._perimeter(planar, st)

    def feedback(self, *args, **kwargs):  # called by BASE_routing for RL algs
        """GPSR is pure heuristic – feedback is a no‑op."""
        return None

    # ----------------- GREEDY MODE -----------------------------------

    def _greedy(self, neighbors):
        depot = self.drone.depot.coords
        best_dist = util.euclidean_distance(self.drone.coords, depot)
        chosen = None
        for coord, nb in neighbors:
            d = util.euclidean_distance(coord, depot)
            if d < best_dist - 1e-6:
                best_dist = d
                chosen = nb
        return chosen

    # ----------------- PERIMETER MODE --------------------------------

    def _perimeter(self, neighbors, st: GPSRPacketState):
        if not neighbors:
            return None

        # If some neighbour is closer to sink than entry point → switch back to Greedy
        entry_dist = util.euclidean_distance(st.entry_coord, self.drone.depot.coords) if st.entry_coord else math.inf
        for coord, nb in neighbors:
            if util.euclidean_distance(coord, self.drone.depot.coords) < entry_dist - 1e-6:
                st.mode = "GREEDY"
                return nb

        origin = st.last_edge_origin or self.drone.coords
        prev_vec = None
        if st.last_hop is not None and st.last_hop != self.drone.identifier:
            prev_drone = self.simulator.drones[st.last_hop]
            prev_vec = np.array(self.drone.coords) - np.array(prev_drone.coords)

        min_angle = math.inf
        chosen = None
        for coord, nb in neighbors:
            edge_vec = np.array(coord) - np.array(origin)
            if np.linalg.norm(edge_vec) == 0:
                continue
            angle = self._positive_angle(prev_vec, edge_vec) if prev_vec is not None else self._bearing(edge_vec)
            if angle < min_angle:
                min_angle = angle
                chosen = nb
        if chosen:
            st.last_hop = self.drone.identifier
            st.last_edge_origin = self.drone.coords
        return chosen

    # --------------- neighbour helpers -------------------------------

    @staticmethod
    def _normalise_neighbors(opt_neighbors):
        norm = []
        for item in opt_neighbors:
            if isinstance(item, tuple):  # (hello_packet, drone)
                hello_pkt, nb = item
                coord = getattr(hello_pkt, "cur_pos", nb.coords)
            else:  # just Drone
                nb = item
                coord = nb.coords
            norm.append((coord, nb))
        return norm

    # ---------------- PLANARISATION ----------------------------------

    def _planarise(self, neighbors):
        return self._rng(neighbors) if self.planar_method == "RNG" else self._gg(neighbors)

    def _rng(self, neighbors):
        """Relative Neighbour Graph w.r.t current drone."""
        edges = []
        for i, (pos_i, nb_i) in enumerate(neighbors):
            keep = True
            for j, (pos_j, _) in enumerate(neighbors):
                if i == j:
                    continue
                dij = util.euclidean_distance(pos_i, pos_j)
                di0 = util.euclidean_distance(pos_i, self.drone.coords)
                d0j = util.euclidean_distance(self.drone.coords, pos_j)
                if dij < max(di0, d0j) - 1e-6:
                    keep = False
                    break
            if keep:
                edges.append((pos_i, nb_i))
        return edges

    def _gg(self, neighbors):
        """Gabriel Graph w.r.t current drone."""
        edges = []
        for i, (pos_i, nb_i) in enumerate(neighbors):
            keep = True
            for j, (pos_j, _) in enumerate(neighbors):
                if i == j:
                    continue
                mid = ((pos_i[0] + self.drone.coords[0]) / 2, (pos_i[1] + self.drone.coords[1]) / 2)
                radius = util.euclidean_distance(pos_i, self.drone.coords) / 2
                if util.euclidean_distance(pos_j, mid) < radius - 1e-6:
                    keep = False
                    break
            if keep:
                edges.append((pos_i, nb_i))
        return edges

    # ---------------- ANGLE UTILITIES --------------------------------

    @staticmethod
    def _positive_angle(vec_a: Optional[np.ndarray], vec_b: np.ndarray):
        if vec_a is None:
            return 0.0
        angle = math.atan2(np.cross(vec_a, vec_b), np.dot(vec_a, vec_b))
        return angle % (2 * math.pi)

    @staticmethod
    def _bearing(vec):
        return (math.atan2(vec[1], vec[0]) + 2 * math.pi) % (2 * math.pi)

    # ---------------- STATE ------------------------------------------

    @staticmethod
    def _attach_state(packet: DataPacket):
        if not hasattr(packet, "gpsr_state"):
            packet.gpsr_state = GPSRPacketState()  # type: ignore
