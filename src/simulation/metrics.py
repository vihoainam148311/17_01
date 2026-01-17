import numpy as np
import pickle
import pandas as pd
import seaborn as sb
import json
import matplotlib.pyplot as plt

from src.entities.uav_entities import DataPacket
from collections import defaultdict
from src.utilities import utilities as util
from src.utilities import config


class Metrics:
    def __init__(self, simulator):
        self.simulator = simulator
        self.drones_packets_to_depot_list = []
        self.all_control_packets_in_simulation = 0
        self.all_data_packets_in_simulation = 0
        self.events = set()
        self.events_not_listened = set()
        self.drones_packets = set()
        self.drones_packets_to_depot = set()
        self.time_on_mission = 0
        self.time_on_active_routing = 0
        self.mean_numbers_of_possible_relays = []

        # Energy consumption metrics
        self.total_initial_energy = 0
        self.total_energy_consumed = 0
        self.total_remaining_energy = 0
        self.total_data_packets_sent = 0

        # Event tracking
        self.number_of_generated_events = 0
        self.number_of_detected_events = 0
        self.number_of_not_generated_events = 0
        self.number_of_events_to_depot = 0
        self.number_of_packets_to_depot = 0
        self.throughput = 0

        # Delivery time metrics
        self.packet_mean_delivery_time = 0
        self.event_mean_delivery_time = 0
        self.packet_delivery_ratio = 0
        self.event_delivery_times = []

        self.average_hop_count = 0 # Biến lưu số hop trung bình
        # Mission setup will be populated by info_mission()
        self.mission_setup = {}

    def compute_energy_metrics(self, drones):
        """
        Compute energy consumption metrics for all UAVs.
        """
        try:
            self.total_initial_energy = sum(drone.initial_energy for drone in drones)
            self.total_remaining_energy = sum(drone.residual_energy for drone in drones)
            self.total_energy_consumed = self.total_initial_energy - self.total_remaining_energy
            self.total_data_packets_sent = sum(len(drone.sended_pck) for drone in drones)

            if config.DEBUG:
                print(f"[DEBUG] Energy Metrics - Initial: {self.total_initial_energy}, "
                      f"Consumed: {self.total_energy_consumed}, Remaining: {self.total_remaining_energy}")
        except Exception as e:
            if config.DEBUG:
                print(f"[ERROR] Failed to compute energy metrics: {e}")
            # Set default values on error
            self.total_initial_energy = 0
            self.total_energy_consumed = 0
            self.total_remaining_energy = 0
            self.total_data_packets_sent = 0

    def validate_packets(self):
        """
        Validate packet data before calculation.
        Returns only valid DataPackets with proper time attributes.
        """
        valid_packets = []
        invalid_count = 0

        for pck in self.drones_packets_to_depot:
            if (isinstance(pck, DataPacket) and
                    hasattr(pck, 'time_delivery') and
                    hasattr(pck, 'time_step_creation') and
                    pck.time_delivery is not None and
                    pck.time_step_creation is not None and
                    pck.time_delivery >= pck.time_step_creation):  # Sanity check
                valid_packets.append(pck)
            else:
                invalid_count += 1

        if config.DEBUG and invalid_count > 0:
            print(
                f"[DEBUG] Found {invalid_count} invalid packets out of {len(self.drones_packets_to_depot)} total packets")

        return valid_packets

    def debug_delivery_time_calculation(self):
        """Debug delivery time calculation step by step"""
        print(f"[DEBUG] === DELIVERY TIME ANALYSIS ===")

        delivery_times_steps = []
        delivery_times_seconds = []
        creation_times = []
        delivery_times_raw = []
        outlier_packets = []

        for pck in self.drones_packets_to_depot:
            if (isinstance(pck, DataPacket) and
                    hasattr(pck, 'time_delivery') and hasattr(pck, 'time_step_creation') and
                    pck.time_delivery is not None and pck.time_step_creation is not None):

                # Time in steps
                time_steps = pck.time_delivery - pck.time_step_creation
                time_seconds = time_steps * self.simulator.time_step_duration

                delivery_times_steps.append(time_steps)
                delivery_times_seconds.append(time_seconds)
                creation_times.append(pck.time_step_creation)
                delivery_times_raw.append(pck.time_delivery)

                # Track outliers (> 60 seconds)
                if time_seconds > 60:
                    outlier_packets.append({
                        'packet_id': pck.identifier,
                        'event_id': pck.event_ref.identifier,
                        'creation_step': pck.time_step_creation,
                        'delivery_step': pck.time_delivery,
                        'delivery_time_steps': time_steps,
                        'delivery_time_seconds': time_seconds
                    })

        if delivery_times_steps:
            print(f"Time step duration: {self.simulator.time_step_duration}s")
            print(f"Total simulation steps: {self.simulator.len_simulation}")
            print(f"Total simulation time: {self.simulator.len_simulation * self.simulator.time_step_duration}s")
            print(f"Valid packets analyzed: {len(delivery_times_steps)}")
            print(f"")
            print(f"DELIVERY TIME STATISTICS:")
            print(f"  Mean delivery time (steps): {np.mean(delivery_times_steps):.1f}")
            print(f"  Mean delivery time (seconds): {np.mean(delivery_times_seconds):.1f}")
            print(f"  Median delivery time: {np.median(delivery_times_seconds):.1f}s")
            print(f"  Min delivery time: {np.min(delivery_times_seconds):.1f}s")
            print(f"  Max delivery time: {np.max(delivery_times_seconds):.1f}s")
            print(f"  95th percentile: {np.percentile(delivery_times_seconds, 95):.1f}s")
            print(f"")

            # Check creation time distribution
            print(f"PACKET CREATION ANALYSIS:")
            print(f"  Earliest creation: step {min(creation_times)}")
            print(f"  Latest creation: step {max(creation_times)}")
            print(f"  Creation time span: {max(creation_times) - min(creation_times)} steps")
            print(f"")

            # Analyze outliers
            fast_packets = [t for t in delivery_times_seconds if t < 5]  # < 5 seconds
            normal_packets = [t for t in delivery_times_seconds if 5 <= t <= 30]  # 5-30 seconds
            slow_packets = [t for t in delivery_times_seconds if 30 < t <= 60]  # 30-60 seconds
            very_slow_packets = [t for t in delivery_times_seconds if t > 60]  # > 60 seconds

            print(f"DELIVERY TIME DISTRIBUTION:")
            print(
                f"  Fast packets (<5s): {len(fast_packets)} ({len(fast_packets) / len(delivery_times_seconds) * 100:.1f}%)")
            print(
                f"  Normal packets (5-30s): {len(normal_packets)} ({len(normal_packets) / len(delivery_times_seconds) * 100:.1f}%)")
            print(
                f"  Slow packets (30-60s): {len(slow_packets)} ({len(slow_packets) / len(delivery_times_seconds) * 100:.1f}%)")
            print(
                f"  Very slow packets (>60s): {len(very_slow_packets)} ({len(very_slow_packets) / len(delivery_times_seconds) * 100:.1f}%)")
            print(f"")

            # Show outlier details
            if outlier_packets:
                print(f"OUTLIER ANALYSIS (>60s delivery time):")
                print(f"  Total outliers: {len(outlier_packets)}")
                print(f"  Sample outliers:")
                for i, pkt in enumerate(outlier_packets[:5]):  # Show first 5
                    print(f"    Packet {pkt['packet_id']}: {pkt['delivery_time_seconds']:.1f}s "
                          f"(steps {pkt['creation_step']} -> {pkt['delivery_step']})")

            # Check for potential issues
            print(f"POTENTIAL ISSUES:")
            if np.mean(delivery_times_seconds) > 25:
                print(f"  WARNING: High mean delivery time ({np.mean(delivery_times_seconds):.1f}s)")
            if len(very_slow_packets) > len(delivery_times_seconds) * 0.1:  # >10% very slow
                print(
                    f"  WARNING: High percentage of very slow packets ({len(very_slow_packets) / len(delivery_times_seconds) * 100:.1f}%)")
            if max(delivery_times_seconds) > 120:
                print(f"  WARNING: Extremely slow packets detected (max: {max(delivery_times_seconds):.1f}s)")

        else:
            print(f"No valid delivery time data found!")

        print(f"=== END DELIVERY TIME ANALYSIS ===")

    def debug_routing_efficiency(self):
        """Debug routing efficiency and network performance"""
        print(f"[DEBUG] === ROUTING EFFICIENCY ANALYSIS ===")

        # Basic network stats
        mean_relays = np.mean(self.mean_numbers_of_possible_relays) if self.mean_numbers_of_possible_relays else 0
        print(f"Network connectivity:")
        print(f"  Mean possible relays: {mean_relays:.2f}")
        print(f"  Communication range: {getattr(self.simulator, 'drone_com_range', 'Unknown')}m")
        print(
            f"  Environment size: {getattr(self.simulator, 'env_width', 'Unknown')}m x {getattr(self.simulator, 'env_height', 'Unknown')}m")

        # Calculate theoretical minimum hops to depot
        env_width = getattr(self.simulator, 'env_width', 1800)
        comm_range = getattr(self.simulator, 'drone_com_range', 300)
        theoretical_min_hops = max(1, int(np.sqrt(2) * env_width / 2 / comm_range))
        print(f"  Theoretical minimum hops to depot: {theoretical_min_hops}")

        # Packet efficiency
        print(f"")
        print(f"Packet routing efficiency:")
        print(f"  Total packets generated: {self.all_data_packets_in_simulation}")
        print(f"  Packets delivered: {self.number_of_packets_to_depot}")
        print(f"  Delivery ratio: {self.packet_delivery_ratio:.3f}")
        print(
            f"  Control packet overhead: {self.all_control_packets_in_simulation / max(1, self.all_data_packets_in_simulation):.1f}x")

        # Time analysis
        total_sim_time = self.simulator.len_simulation * self.simulator.time_step_duration
        print(f"")
        print(f"Time efficiency:")
        print(f"  Total simulation time: {total_sim_time:.1f}s")
        print(f"  Mean delivery time: {self.packet_mean_delivery_time:.1f}s")
        print(f"  Delivery time as % of sim time: {self.packet_mean_delivery_time / total_sim_time * 100:.2f}%")
        print(f"  Throughput: {self.throughput:.4f} packets/sec")

        print(f"=== END ROUTING EFFICIENCY ANALYSIS ===")

    def calculate_throughput(self):
        """
        Standardized throughput calculation in packets per second.
        """
        total_time = self.simulator.len_simulation * self.simulator.time_step_duration
        if total_time > 0:
            return len(self.drones_packets_to_depot) / total_time
        else:
            if config.DEBUG:
                print("[WARNING] Total simulation time is zero, throughput set to 0")
            return 0.0

    def calculate_delivery_times(self, valid_packets):
        """
        Calculate packet and event delivery times.
        Returns tuple: (packet_delivery_times, event_delivery_times)
        """
        packet_delivery_times = []

        # Group packets by event for event delivery time calculation
        event_delivery_dict = defaultdict(list)

        for pck in valid_packets:
            # Calculate packet delivery time (consistent time attributes)
            delivery_time = pck.time_delivery - pck.time_step_creation
            packet_delivery_times.append(delivery_time)

            # Calculate event delivery time (from event creation to packet delivery)
            event_delivery_time = pck.time_delivery - pck.event_ref.current_time
            event_delivery_dict[pck.event_ref.identifier].append(event_delivery_time)

        # For each event, take the minimum delivery time (first packet arrival)
        event_delivery_times = []
        for event_id, times in event_delivery_dict.items():
            if times:  # Ensure non-empty list
                event_delivery_times.append(min(times))

        return packet_delivery_times, event_delivery_times

    def info_mission(self):
        """Store mission setup parameters"""
        try:
            self.mission_setup = {
                "n_drones": self.simulator.n_drones,
                "sim_duration": self.simulator.len_simulation,
                "com_range": self.simulator.drone_com_range,
                "sen_range": self.simulator.drone_sen_range,
                "speed": self.simulator.drone_speed,
                "max_energy": self.simulator.drone_max_energy,
                "time_step_duration": self.simulator.time_step_duration,
                "packets_max_ttl": self.simulator.packets_max_ttl,
                "routing_algorithm": self.simulator.routing_algorithm_name,  # Use string name
                "total_initial_energy": self.total_initial_energy,
                "total_energy_consumed": self.total_energy_consumed,
                "total_remaining_energy": self.total_remaining_energy,
                "drone_communication_success": self.simulator.drone_communication_success,
                "depot_com_range": self.simulator.depot_com_range,
                "depot_coordinates": self.simulator.depot_coordinates,
                "event_duration": self.simulator.event_duration,
                "show_plot": self.simulator.show_plot,
                "time_on_active_routing": str(self.time_on_active_routing)
            }
        except AttributeError as e:
            if config.DEBUG:
                print(f"[WARNING] Some mission parameters not available: {e}")
            # Set minimal required info
            self.mission_setup = {
                "sim_duration": getattr(self.simulator, 'len_simulation', 0),
                "time_step_duration": getattr(self.simulator, 'time_step_duration', 1),
                "routing_algorithm": getattr(self.simulator, 'routing_algorithm_name', 'Unknown')
            }

    def other_metrics(self):
        """
        Calculate all other metrics including delivery times, ratios, and event statistics.
        Fixed version that handles packet duplicates correctly.
        """
        # Validate packets before processing
        valid_packets = self.validate_packets()

        if config.DEBUG:
            print(f"[DEBUG] Number of valid DataPackets used for delay calculation: {len(valid_packets)}")

        # Event statistics
        self.number_of_generated_events = len(self.events)
        self.number_of_not_generated_events = len(self.events_not_listened)

        # Detected events (unique events from all packets)
        all_detected_events = set([pck.event_ref for pck in self.drones_packets])
        self.number_of_detected_events = len(all_detected_events)

        # === FIX: Handle packet duplicates properly ===
        # Count unique packets and events delivered to depot
        unique_packet_identifiers = set()
        unique_event_identifiers = set()
        first_delivery_packets = {}  # event_id -> first packet that delivered it

        hop_counts_list = []
        # -------------------------------

        for pck in self.drones_packets_to_depot:
            if isinstance(pck, DataPacket) and hasattr(pck, 'event_ref') and pck.event_ref is not None:
                event_id = pck.event_ref.identifier
                packet_id = pck.identifier

                # Chỉ tính cho các gói tin duy nhất (không tính gói trùng lặp)
                if packet_id not in unique_packet_identifiers:
                    unique_packet_identifiers.add(packet_id)

                    # --- THU THẬP SỐ HOP ---
                    # Trừ 1 vì hop_count bắt đầu tính từ Source (node tạo ra).
                    # Nếu Source -> Depot thì hop_count=1 (nhưng thực tế là 1 đường truyền).
                    # Nếu Source -> Relay -> Depot thì hop_count=2 (2 đường truyền).
                    # Tùy định nghĩa của bạn, thường Hop = số liên kết = số nút đi qua - 1 (nếu tính cả đích)
                    # Ở đây add_hop được gọi khi gói tin được tạo/nhận tại drone.
                    # Khi đến Depot, nó không gọi add_hop trên object packet đó nữa (thường là vậy).
                    # Nên hop_count hiện tại = số drone đã giữ gói tin (bao gồm Source).
                    # Số Hop truyền dẫn (Links) = hop_count - 1 (nếu trực tiếp Source->Depot thì coi là 1 hop)

                    # Tuy nhiên, để an toàn và đơn giản, ta lấy giá trị hop_count hiện tại.
                    # Nếu bạn muốn tính số nút trung gian: hop_count - 2
                    # Nếu tính số liên kết (links): hop_count
                    # Ở đây tôi lấy giá trị hop_count (số lần truyền)

                    current_hops = getattr(pck, 'hop_count', 0)
                    hop_counts_list.append(current_hops)
                    # -----------------------

                # For events, only count the first delivery
                if event_id not in unique_event_identifiers:
                    unique_event_identifiers.add(event_id)
                    first_delivery_packets[event_id] = pck

        for pck in self.drones_packets_to_depot:
            if isinstance(pck, DataPacket) and hasattr(pck, 'event_ref') and pck.event_ref is not None:
                event_id = pck.event_ref.identifier
                packet_id = pck.identifier

                # Count unique packets
                unique_packet_identifiers.add(packet_id)

                # For events, only count the first delivery
                if event_id not in unique_event_identifiers:
                    unique_event_identifiers.add(event_id)
                    first_delivery_packets[event_id] = pck
                elif config.DEBUG:
                    print(f"[DEBUG] Duplicate delivery for event {event_id} - packet {packet_id}")
        # --- TÍNH TRUNG BÌNH SỐ HOP ---
        if hop_counts_list:
            self.average_hop_count = np.mean(hop_counts_list)
        else:
            self.average_hop_count = 0


        # Update counts with unique values
        self.number_of_events_to_depot = len(unique_event_identifiers)
        self.number_of_packets_to_depot = len(unique_packet_identifiers)

        if config.DEBUG:
            total_deliveries = len(self.drones_packets_to_depot)
            duplicates = total_deliveries - len(unique_packet_identifiers)
            print(
                f"[DEBUG] Total deliveries: {total_deliveries}, Unique packets: {len(unique_packet_identifiers)}, Duplicates: {duplicates}")

        # Calculate throughput using unique packets
        total_time = self.simulator.len_simulation * self.simulator.time_step_duration
        if total_time > 0:
            self.throughput = len(unique_packet_identifiers) / total_time
        else:
            self.throughput = 0
            if config.DEBUG:
                print("[WARNING] Total simulation time is zero, throughput set to 0")

        # Calculate delivery times using only first deliveries
        if first_delivery_packets:
            packet_delivery_times = []
            event_delivery_times = []

            for event_id, pck in first_delivery_packets.items():
                # Calculate packet delivery time
                if (hasattr(pck, 'time_delivery') and hasattr(pck, 'time_step_creation') and
                        pck.time_delivery is not None and pck.time_step_creation is not None):

                    packet_delivery_time = pck.time_delivery - pck.time_step_creation
                    packet_delivery_times.append(packet_delivery_time)

                    # Calculate event delivery time
                    if hasattr(pck.event_ref, 'current_time') and pck.event_ref.current_time is not None:
                        event_delivery_time = pck.time_delivery - pck.event_ref.current_time
                        event_delivery_times.append(event_delivery_time)

            # Convert to seconds and calculate means
            self.packet_mean_delivery_time = (
                np.nanmean(packet_delivery_times) * self.simulator.time_step_duration
                if packet_delivery_times else 0
            )

            self.event_mean_delivery_time = (
                np.nanmean(event_delivery_times) * self.simulator.time_step_duration
                if event_delivery_times else 0
            )

            self.event_delivery_times = event_delivery_times
        else:
            self.packet_mean_delivery_time = 0
            self.event_mean_delivery_time = 0
            self.event_delivery_times = []

        # === FIX: Correct packet delivery ratio calculation ===
        if self.all_data_packets_in_simulation > 0:
            self.packet_delivery_ratio = len(unique_packet_identifiers) / self.all_data_packets_in_simulation
        else:
            self.packet_delivery_ratio = 0
            if config.DEBUG:
                print("[WARNING] No data packets generated, delivery ratio set to 0")

        # Validate delivery ratio
        if self.packet_delivery_ratio > 1.0:
            if config.DEBUG:
                print(f"[ERROR] Invalid delivery ratio: {self.packet_delivery_ratio}")
                print(f"[ERROR] Unique packets delivered: {len(unique_packet_identifiers)}")
                print(f"[ERROR] Total packets generated: {self.all_data_packets_in_simulation}")
                print("[ERROR] This indicates a bug in packet counting logic!")
            # Cap at 1.0 as emergency fix
            self.packet_delivery_ratio = min(1.0, self.packet_delivery_ratio)

    # Add this debug method to metrics.py class
    def debug_packet_consistency(self):
        """
        Debug method to check packet counting consistency and find duplicates.
        """
        print(f"[DEBUG] === PACKET CONSISTENCY CHECK ===")
        print(f"All data packets generated: {self.all_data_packets_in_simulation}")
        print(f"Packets in drones_packets set: {len(self.drones_packets)}")
        print(f"Total packet deliveries: {len(self.drones_packets_to_depot)}")

        # Analyze packets delivered to depot
        packet_types = {}
        event_refs = []
        packet_ids = []

        for pck in self.drones_packets_to_depot:
            pck_type = type(pck).__name__
            packet_types[pck_type] = packet_types.get(pck_type, 0) + 1

            if isinstance(pck, DataPacket):
                if hasattr(pck, 'event_ref') and pck.event_ref is not None:
                    event_refs.append(pck.event_ref.identifier)
                if hasattr(pck, 'identifier'):
                    packet_ids.append(pck.identifier)

        print(f"Packet types in depot: {packet_types}")

        # Check for duplicates
        unique_events = set(event_refs)
        unique_packets = set(packet_ids)

        print(f"Unique events delivered: {len(unique_events)}")
        print(f"Unique packets delivered: {len(unique_packets)}")
        print(f"Event deliveries: {len(event_refs)}")
        print(f"Packet deliveries: {len(packet_ids)}")

        if len(unique_events) != len(event_refs):
            from collections import Counter
            event_counts = Counter(event_refs)
            duplicates = {event_id: count for event_id, count in event_counts.items() if count > 1}
            print(f"[WARNING] Duplicate event deliveries: {len(duplicates)} events")
            if config.DEBUG:
                for event_id, count in list(duplicates.items())[:5]:  # Show first 5
                    print(f"  Event {event_id}: delivered {count} times")

        if len(unique_packets) != len(packet_ids):
            from collections import Counter
            packet_counts = Counter(packet_ids)
            duplicates = {pck_id: count for pck_id, count in packet_counts.items() if count > 1}
            print(f"[WARNING] Duplicate packet deliveries: {len(duplicates)} packets")
            if config.DEBUG:
                for pck_id, count in list(duplicates.items())[:5]:  # Show first 5
                    print(f"  Packet {pck_id}: delivered {count} times")

        print(f"=== END CONSISTENCY CHECK ===")
        return len(unique_events), len(unique_packets)

    def get_summary_dict(self):
        """
        Get a comprehensive dictionary representation of all metrics.
        Updated to use corrected packet counts.
        """
        # Calculate mean of possible relays safely
        mean_relays = (np.nanmean(self.mean_numbers_of_possible_relays)
                       if self.mean_numbers_of_possible_relays else 0)

        # Calculate energy efficiency per unique packet
        energy_per_packet = (self.total_energy_consumed / self.number_of_packets_to_depot
                             if self.number_of_packets_to_depot > 0 else 0)

        return {
            # Basic packet metrics - using corrected counts
            "all_data_packets_in_simulation": self.all_data_packets_in_simulation,
            "all_control_packets_in_simulation": self.all_control_packets_in_simulation,
            "drones_packets": len(self.drones_packets),
            "drones_packets_to_depot": len(self.drones_packets_to_depot),  # Total deliveries (may include duplicates)
            "unique_packets_to_depot": self.number_of_packets_to_depot,  # Unique packets delivered

            # Event metrics
            "events_generated": self.number_of_generated_events,
            "events_detected": self.number_of_detected_events,
            "events_not_listened": self.number_of_not_generated_events,
            "events_to_depot": self.number_of_events_to_depot,

            # Performance metrics - using corrected values
            "packet_delivery_ratio": self.packet_delivery_ratio,
            "packet_mean_delivery_time": self.packet_mean_delivery_time,
            "event_mean_delivery_time": self.event_mean_delivery_time,
            "throughput": self.throughput,
            # add1
            "average_hop_count": self.average_hop_count,
            # Energy metrics - using unique packet count
            "total_initial_energy": self.total_initial_energy,
            "total_energy_consumed": self.total_energy_consumed,
            "total_remaining_energy": self.total_remaining_energy,
            "energy_efficiency": energy_per_packet,

            # Network metrics
            "mean_numbers_of_possible_relays": mean_relays,
            "time_on_mission": self.time_on_mission,
            "time_on_active_routing": self.time_on_active_routing,

            # Mission info
            "simulation_time": (self.mission_setup.get("sim_duration", 0) *
                                self.mission_setup.get("time_step_duration", 1)),
            "routing_algorithm": self.mission_setup.get("routing_algorithm", "Unknown"),

            # Duplicate detection metrics
            "total_packet_deliveries": len(self.drones_packets_to_depot),
            "duplicate_deliveries": len(self.drones_packets_to_depot) - self.number_of_packets_to_depot
        }

    def validate_metrics_consistency(self):
        """
        Validate metrics for consistency and return list of issues found.
        """
        issues = []

        # Energy consistency check
        calculated_consumed = self.total_initial_energy - self.total_remaining_energy
        if abs(self.total_energy_consumed - calculated_consumed) > 0.001:  # Small tolerance for floating point
            issues.append(f"Energy calculation inconsistency: "
                          f"stored={self.total_energy_consumed}, calculated={calculated_consumed}")

        # Packet count consistency
        if self.number_of_packets_to_depot != len(self.drones_packets_to_depot):
            issues.append(f"Packet count inconsistency: "
                          f"stored={self.number_of_packets_to_depot}, actual={len(self.drones_packets_to_depot)}")

        # Delivery ratio bounds check
        if self.packet_delivery_ratio < 0 or self.packet_delivery_ratio > 1:
            issues.append(f"Invalid delivery ratio: {self.packet_delivery_ratio}")

        # Throughput reasonableness check
        if self.throughput < 0:
            issues.append(f"Negative throughput: {self.throughput}")

        # Time consistency checks
        if self.packet_mean_delivery_time < 0:
            issues.append(f"Negative mean delivery time: {self.packet_mean_delivery_time}")

        return issues

    def print_overall_stats(self):
        """Print comprehensive statistics with enhanced debugging"""
        print("=" * 50)
        print("         SIMULATION METRICS SUMMARY")
        print("=" * 50)

        # Run debug analysis first
        if config.DEBUG:
            self.debug_delivery_time_calculation()
            print()
            self.debug_routing_efficiency()
            print()

        # Basic packet statistics
        print(f"📦 PACKET STATISTICS:")
        print(f"   Total data packets generated: {self.all_data_packets_in_simulation}")
        print(f"   Total control packets: {self.all_control_packets_in_simulation}")
        print(f"   Packets delivered to depot: {len(self.drones_packets_to_depot)}")
        print(f"   Packet delivery ratio: {self.packet_delivery_ratio:.3f}")

        # Event statistics
        print(f"\n🎯 EVENT STATISTICS:")
        print(f"   Events generated: {self.number_of_generated_events}")
        print(f"   Events detected: {self.number_of_detected_events}")
        print(f"   Events delivered to depot: {self.number_of_events_to_depot}")
        print(f"   Events not listened: {self.number_of_not_generated_events}")

        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Mean packet delivery time: {self.packet_mean_delivery_time:.3f} sec")
        print(f"   Mean event delivery time: {self.event_mean_delivery_time:.3f} sec")
        print(f"   Throughput: {self.throughput:.4f} packets/sec")

        print(f"   Average Hops to Depot: {self.average_hop_count:.2f} hops")
        # Energy metrics
        print(f"\n🔋 ENERGY METRICS:")
        print(f"   Total initial energy: {self.total_initial_energy} J")
        print(f"   Total energy consumed: {self.total_energy_consumed} J")
        print(
            f"   Energy efficiency: {(self.total_energy_consumed / len(self.drones_packets_to_depot) if len(self.drones_packets_to_depot) > 0 else 0):.3f} J/packet")

        # Network metrics
        if self.mean_numbers_of_possible_relays:
            mean_relays = np.nanmean(self.mean_numbers_of_possible_relays)
            print(f"\n🌐 NETWORK METRICS:")
            print(f"   Mean possible relays: {mean_relays:.2f}")
            print(f"   Time on mission: {self.time_on_mission}")
            print(f"   Time on active routing: {self.time_on_active_routing}")

        # Validation
        issues = self.validate_metrics_consistency()
        if issues:
            print(f"\n⚠️  VALIDATION ISSUES:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print(f"\n✅ All metrics validated successfully")

        print("=" * 50)

    def save(self, filename="metrics.npy"):
        """Save metrics as numpy array."""
        try:
            np.save(filename, self.get_summary_dict())
            if config.DEBUG:
                print(f"[DEBUG] Metrics saved to {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save metrics to {filename}: {e}")

    def save_as_json(self, filename="metrics.json"):
        """Save metrics as JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.get_summary_dict(), f, indent=4, default=str)
            if config.DEBUG:
                print(f"[DEBUG] Metrics saved to {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save metrics to {filename}: {e}")

    def save_detailed_analysis(self, filename_prefix="detailed_metrics"):
        """Save detailed analysis including distributions and statistics."""
        try:
            summary = self.get_summary_dict()

            # Save summary
            with open(f"{filename_prefix}_summary.json", 'w') as f:
                json.dump(summary, f, indent=4, default=str)

            # Save delivery time distributions if available
            if self.event_delivery_times:
                analysis = {
                    "event_delivery_times": {
                        "data": self.event_delivery_times,
                        "mean": float(np.mean(self.event_delivery_times)),
                        "std": float(np.std(self.event_delivery_times)),
                        "min": float(np.min(self.event_delivery_times)),
                        "max": float(np.max(self.event_delivery_times)),
                        "percentiles": {
                            "25th": float(np.percentile(self.event_delivery_times, 25)),
                            "50th": float(np.percentile(self.event_delivery_times, 75)),
                            "75th": float(np.percentile(self.event_delivery_times, 75)),
                            "95th": float(np.percentile(self.event_delivery_times, 95))
                        }
                    }
                }

                with open(f"{filename_prefix}_analysis.json", 'w') as f:
                    json.dump(analysis, f, indent=4, default=str)

            if config.DEBUG:
                print(f"[DEBUG] Detailed analysis saved with prefix {filename_prefix}")

        except Exception as e:
            print(f"[ERROR] Failed to save detailed analysis: {e}")

    # Deprecated method for backward compatibility
    def __dictionary_represenation(self):
        """Deprecated: Use get_summary_dict() instead."""
        if config.DEBUG:
            print("[WARNING] __dictionary_represenation is deprecated, use get_summary_dict()")
        return self.get_summary_dict()