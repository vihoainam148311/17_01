# -*- coding: utf-8 -*-
"""
Speed-Optimized PPO Routing for FANET

Key Optimizations:
1. Lazy numpy/torch conversion - only when needed
2. Cached computations - reuse expensive calculations
3. Simplified networks - fewer layers
4. Reduced update frequency - less overhead
5. No-copy operations where possible
6. Vectorized operations
7. Minimal memory allocations
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    from src.routing_algorithms.BASE_routing import BASE_routing
    from src.utilities import utilities as util
except Exception:
    from BASE_routing import BASE_routing
    import utilities as util

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Lightweight networks
class FastActor(nn.Module):
    def __init__(self, state_dim: int, cand_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + cand_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor, cand_feats: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            S = state.unsqueeze(0).expand(cand_feats.size(0), -1)
            x = torch.cat([S, cand_feats], dim=-1)
            return self.net(x).squeeze(-1)
        else:
            B, N, F = cand_feats.shape
            S = state.unsqueeze(1).expand(-1, N, -1)
            x = torch.cat([S, cand_feats], dim=-1)
            return self.net(x).squeeze(-1)


class FastCritic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class PPO_Routing(BASE_routing):
    # Aggressive hyperparams for speed
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 5e-4
    batch_size: int = 128
    update_every: int = 2048  # Less frequent updates
    epochs: int = 4  # Fewer epochs
    device: str = "cpu"

    def __init__(self, drone, simulator):
        super().__init__(drone, simulator)

        # Minimal features
        self.state_dim = 6
        self.cand_dim = 5

        # Lightweight networks
        self.actor = FastActor(self.state_dim, self.cand_dim).to(self.device)
        self.critic = FastCritic(self.state_dim).to(self.device)
        self.optim = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr, eps=1e-5
        )

        # Minimal buffers - preallocate
        self._buffer_size = self.update_every
        self._states = np.zeros((self._buffer_size, self.state_dim), dtype=np.float32)
        self._actions = np.zeros(self._buffer_size, dtype=np.int32)
        self._logprobs = np.zeros(self._buffer_size, dtype=np.float32)
        self._values = np.zeros(self._buffer_size, dtype=np.float32)
        self._rewards = np.zeros(self._buffer_size, dtype=np.float32)
        self._dones = np.zeros(self._buffer_size, dtype=np.bool_)
        self._buffer_idx = 0

        # Cache for frequent computations
        self._depot_coords_cache = None
        self._grid_size_cache = None
        self._comm_range_cache = None

        # Pending decisions - minimal storage
        self._pending = {}

        # Stats
        self._updates = 0
        self.routing_hole_count = 0
        self.successful_forwards = 0

    def relay_selection(self, geo_neighbors, packet):
        candidates = [n for n in geo_neighbors
                      if getattr(n, "identifier", None) != self.drone.identifier]

        if not candidates:
            self.routing_hole_count += 1
            return "ROUTING_HOLE"

        # Fast state extraction - no intermediate allocations
        state = self._fast_state(self.drone)

        # Fast candidate features - vectorized
        cand_feats, cand_ids = self._fast_cand_features(candidates)

        # Quick torch conversion - view when possible
        s_t = torch.from_numpy(state).to(self.device)
        cf_t = torch.from_numpy(cand_feats).to(self.device)

        with torch.no_grad():
            logits = self.actor(s_t, cf_t)
            probs = F.softmax(logits, dim=-1)

            # Fast sampling
            a_idx = torch.multinomial(probs, 1).item()
            logprob = torch.log(probs[a_idx] + 1e-8).item()
            value = self.critic(s_t.unsqueeze(0)).item()

        chosen = candidates[a_idx]
        chosen_id = chosen.identifier

        # Minimal storage
        self._pending[chosen_id] = (self._buffer_idx, state, logprob, value)

        return chosen

    def feedback(self, outcome: int, id_j: int, best_action, link_quality: float = 0.0):
        if id_j not in self._pending:
            return

        idx, state, logprob, value = self._pending.pop(id_j)

        # Fast reward
        if outcome == 1:
            reward = 2.0
            done = True
        elif outcome == 0:
            reward = 0.1 - 0.01
            done = False
        else:
            reward = -2.0
            done = True

        # Direct buffer write - no append
        if idx < self._buffer_size:
            self._states[idx] = state
            self._logprobs[idx] = logprob
            self._values[idx] = value
            self._rewards[idx] = reward
            self._dones[idx] = done
            self._buffer_idx += 1

        # Update check
        if self._buffer_idx >= self._buffer_size:
            self._fast_update()
            self._buffer_idx = 0
            self._updates += 1

    def _fast_update(self):
        """Minimal PPO update"""
        if self._buffer_idx == 0:
            return

        # Slice active buffer
        n = self._buffer_idx
        states = self._states[:n]
        logprobs = self._logprobs[:n]
        values = self._values[:n]
        rewards = self._rewards[:n]
        dones = self._dones[:n]

        # Fast GAE - vectorized
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in range(n - 1, -1, -1):
            next_value = 0.0 if t == n - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        # Convert to torch once
        states_t = torch.from_numpy(states).to(self.device)
        logprobs_t = torch.from_numpy(logprobs).to(self.device)
        advantages_t = torch.from_numpy(advantages).to(self.device)
        returns_t = torch.from_numpy(returns).to(self.device)

        # Fast training - fewer epochs, larger batches
        for _ in range(self.epochs):
            # Full batch - no shuffling for speed
            values_pred = self.critic(states_t)

            # Value loss
            value_loss = F.mse_loss(values_pred, returns_t)

            # Simple policy loss (no ratio clipping for speed)
            policy_loss = -(logprobs_t * advantages_t).mean()

            # Combined loss
            loss = policy_loss + self.value_coef * value_loss

            # Quick update
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()

    def _fast_state(self, drone) -> np.ndarray:
        """Minimal state extraction"""
        # Cache grid size
        if self._grid_size_cache is None:
            self._grid_size_cache = float(getattr(self.simulator, "grid_size", 1000.0) or 1000.0)

        x, y = drone.coords if hasattr(drone, "coords") else (0.0, 0.0)
        x = x / self._grid_size_cache
        y = y / self._grid_size_cache

        energy = 0.5
        if hasattr(drone, "residual_energy") and hasattr(drone, "initial_energy"):
            energy = min(1.0, max(0.0, drone.residual_energy / max(drone.initial_energy, 1)))

        # Simplified metrics
        lq = 0.5
        delay = 0.5
        speed = 0.0

        if hasattr(drone, "speed"):
            speed = min(1.0, drone.speed / 30.0)

        return np.array([x, y, energy, lq, delay, speed], dtype=np.float32)

    def _fast_cand_features(self, candidates) -> Tuple[np.ndarray, List[int]]:
        """Vectorized candidate features"""
        n = len(candidates)
        feats = np.zeros((n, self.cand_dim), dtype=np.float32)
        ids = []

        # Cache depot coords
        if self._depot_coords_cache is None:
            try:
                self._depot_coords_cache = self.simulator.depot_coordinates
            except:
                self._depot_coords_cache = (0, 0)

        # Cache comm range
        if self._comm_range_cache is None:
            self._comm_range_cache = max(float(getattr(self.drone, "communication_range", 100) or 100), 1.0)

        try:
            my_coords = self.drone.coords
            dist_self_depot = util.euclidean_distance(my_coords, self._depot_coords_cache)
        except:
            my_coords = (0, 0)
            dist_self_depot = 0.0

        for i, nb in enumerate(candidates):
            ids.append(nb.identifier)

            # Fast distance
            try:
                d = util.euclidean_distance(my_coords, nb.coords)
                dist_norm = max(0.0, min(1.0, 1.0 - d / self._comm_range_cache))
            except:
                dist_norm = 0.5

            # Simple link quality estimate
            lq = dist_norm * 0.8 + 0.2

            # Bitrate proxy
            bitrate_norm = lq

            # Depot progress
            try:
                dist_nb_depot = util.euclidean_distance(nb.coords, self._depot_coords_cache)
                progress = (dist_self_depot - dist_nb_depot) / max(dist_self_depot, 1.0)
                depot_delta = max(0.0, min(1.0, 0.5 + 0.5 * progress))
            except:
                depot_delta = 0.5

            # Minimal relative speed
            rel_speed = 0.0

            feats[i] = [dist_norm, lq, bitrate_norm, depot_delta, rel_speed]

        return feats, ids

    def _estimate_link_quality(self, src, nb) -> float:
        """Fast link quality estimate"""
        try:
            dist = util.euclidean_distance(src.coords, nb.coords)
            comm_range = self._comm_range_cache or 100.0
            return max(0.0, min(1.0, 1.0 - dist / comm_range))
        except:
            return 0.5

    def get_performance_metrics(self):
        return {
            'updates': self._updates,
            'routing_holes': self.routing_hole_count,
            'successful_forwards': self.successful_forwards,
        }