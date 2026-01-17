import numpy as np
import multiprocessing
import pandas as pd
import os
from datetime import datetime
#from fix_simulator import auto_patch
#auto_patch()
from src.simulation.simulator import Simulator
from src.utilities.policies import *
from src.utilities import config

#CLAUDE
def run_simulation(params):
    """
    Chạy một simulation với các tham số đã cho và trả về metrics.

    Args:
        params: tuple (n_drones, drone_speed, seed)

    Returns:
        tuple: (n_drones, drone_speed, routing_algorithm, seed, delivery_ratio,
                mean_delivery_time, total_energy_consumed, energy_efficiency, throughput)
    """
    n_drones, drone_speed, seed = params

    try:
        # Khởi tạo simulator
        sim = Simulator(
            n_drones=n_drones,
            seed=seed,
            drone_speed=drone_speed
        )

        # Chạy simulation
        print(f"[INFO] Starting simulation: Drones={n_drones}, Speed={drone_speed}, Seed={seed}")
        sim.run()

        # Tính toán metrics
        sim.metrics.compute_energy_metrics(sim.drones)
        sim.metrics.other_metrics()

        # Lấy routing algorithm name
        routing_algorithm = sim.routing_algorithm_name

        # Lấy các metrics cần thiết
        delivery_ratio = sim.metrics.packet_delivery_ratio
        mean_delivery_time = sim.metrics.packet_mean_delivery_time
        total_energy_consumed = sim.metrics.total_energy_consumed

        # Tính energy efficiency (J/packet) - sử dụng unique packets
        if sim.metrics.number_of_packets_to_depot > 0:
            energy_efficiency = total_energy_consumed / sim.metrics.number_of_packets_to_depot
        else:
            energy_efficiency = 0.0

        throughput = sim.metrics.throughput

        # Debug info
        if config.DEBUG:
            print(f"[DEBUG] Simulation completed for Drones={n_drones}, Speed={drone_speed}, Seed={seed}")
            print(f"[DEBUG] Metrics: DR={delivery_ratio:.3f}, MDT={mean_delivery_time:.3f}, "
                  f"Energy={total_energy_consumed:.1f}, Efficiency={energy_efficiency:.3f}, "
                  f"Throughput={throughput:.4f}")

        # Đóng simulator
        sim.close()

        return (n_drones, drone_speed, routing_algorithm, seed, delivery_ratio,
                mean_delivery_time, total_energy_consumed, energy_efficiency, throughput)

    except Exception as e:
        print(f"[ERROR] Simulation failed for params {params}: {e}")
        # Trả về kết quả lỗi với các giá trị mặc định
        return (n_drones, drone_speed, "ERROR", seed, 0.0, 0.0, 0.0, 0.0, 0.0)


def save_results(results, filename_prefix="simulation_results"):
    """
    Lưu kết quả simulation vào các file khác nhau.

    Args:
        results: list of tuples với simulation results
        filename_prefix: prefix cho tên file
    """
    # Tạo DataFrame với các cột được yêu cầu
    columns = [
        "Drone",
        "Drone_Speed",
        "routing_algorithm",
        "Seed",
        "Delivery_Ratio",
        "Mean_Delivery_Time",
        "Total_Energy_Consumed",
        "Energy_Efficiency",
        "Throughput"
    ]

    df = pd.DataFrame(results, columns=columns)

    # Tạo timestamp cho filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Tạo thư mục results nếu chưa tồn tại
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Lưu file CSV
    csv_filename = f"{results_dir}/{filename_prefix}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"[INFO] Results saved to CSV: {csv_filename}")

    # Lưu file numpy (để backup)
    npy_filename = f"{results_dir}/{filename_prefix}_{timestamp}.npy"
    np.save(npy_filename, np.array(results, dtype=object))
    print(f"[INFO] Results saved to NPY: {npy_filename}")

    # Lưu summary statistics
    summary_filename = f"{results_dir}/{filename_prefix}_summary_{timestamp}.txt"
    with open(summary_filename, 'w') as f:
        f.write("SIMULATION RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total simulations: {len(df)}\n")
        f.write(f"Routing algorithm: {df['routing_algorithm'].iloc[0]}\n")
        f.write(f"Drone counts: {sorted(df['Drone'].unique())}\n")
        f.write(f"Drone speeds: {sorted(df['Drone_Speed'].unique())}\n")
        f.write(f"Seeds: {sorted(df['Seed'].unique())}\n\n")

        f.write("METRICS SUMMARY:\n")
        f.write("-" * 30 + "\n")

        # Summary statistics cho các metrics quan trọng
        numeric_columns = ['Delivery_Ratio', 'Mean_Delivery_Time', 'Total_Energy_Consumed',
                           'Energy_Efficiency', 'Throughput']

        for col in numeric_columns:
            if col in df.columns:
                f.write(f"\n{col}:\n")
                f.write(f"  Mean: {df[col].mean():.4f}\n")
                f.write(f"  Std:  {df[col].std():.4f}\n")
                f.write(f"  Min:  {df[col].min():.4f}\n")
                f.write(f"  Max:  {df[col].max():.4f}\n")

        # Group by metrics
        f.write(f"\nGROUP BY DRONE COUNT:\n")
        f.write("-" * 30 + "\n")
        grouped = df.groupby('Drone')[numeric_columns].mean()
        f.write(grouped.to_string())

        f.write(f"\n\nGROUP BY DRONE SPEED:\n")
        f.write("-" * 30 + "\n")
        grouped = df.groupby('Drone_Speed')[numeric_columns].mean()
        f.write(grouped.to_string())

    print(f"[INFO] Summary saved to: {summary_filename}")

    return csv_filename, summary_filename


def validate_results(results):
    """
    Kiểm tra tính hợp lệ của kết quả simulation.

    Args:
        results: list of tuples với simulation results

    Returns:
        tuple: (valid_results, invalid_count, issues)
    """
    valid_results = []
    invalid_count = 0
    issues = []

    for result in results:
        n_drones, drone_speed, routing_alg, seed, delivery_ratio, mean_delivery_time, \
            total_energy, energy_efficiency, throughput = result

        is_valid = True
        result_issues = []

        # Kiểm tra delivery ratio
        if delivery_ratio < 0 or delivery_ratio > 1.0:
            is_valid = False
            result_issues.append(f"Invalid delivery ratio: {delivery_ratio}")

        # Kiểm tra mean delivery time
        if mean_delivery_time < 0:
            is_valid = False
            result_issues.append(f"Negative delivery time: {mean_delivery_time}")

        # Kiểm tra energy values
        if total_energy < 0 or energy_efficiency < 0:
            is_valid = False
            result_issues.append(f"Negative energy values: total={total_energy}, efficiency={energy_efficiency}")

        # Kiểm tra throughput
        if throughput < 0:
            is_valid = False
            result_issues.append(f"Negative throughput: {throughput}")

        # Kiểm tra routing algorithm
        if routing_alg == "ERROR":
            is_valid = False
            result_issues.append("Simulation failed")

        if is_valid:
            valid_results.append(result)
        else:
            invalid_count += 1
            issues.append(f"Simulation (Drones={n_drones}, Speed={drone_speed}, Seed={seed}): {result_issues}")

    return valid_results, invalid_count, issues


def main():
    """
    Hàm chính để chạy các simulation và lưu kết quả.
    """
    print("=" * 60)
    print("   UAV NETWORK SIMULATION WITH QMAR ROUTING")
    print("=" * 60)

    # Cấu hình simulation parameters
    drones = range(30,100, 10)  #
    drone_speeds = range(12, 22, 20)  #
    seeds = range(1, 10)  # [1, 2, 3]

    print(f"[INFO] Simulation Configuration:")
    print(f"  Drone counts: {list(drones)}")
    print(f"  Drone speeds: {list(drone_speeds)}")
    print(f"  Seeds: {list(seeds)}")
    print(f"  Routing algorithm: {config.ROUTING_ALGORITHM}")
    print(f"  Total simulations: {len(drones) * len(drone_speeds) * len(seeds)}")

    # Tạo danh sách tham số cho các simulation
    simulation_params = [
        (drone, drone_speed, seed)
        for drone in drones
        for drone_speed in drone_speeds
        for seed in seeds
    ]

    print(f"\n[INFO] Starting {len(simulation_params)} simulations...")

    # Chạy simulations song song
    try:
        if len(simulation_params) > 1:
            # Sử dụng multiprocessing cho multiple simulations
            with multiprocessing.Pool() as pool:
                results = pool.map(run_simulation, simulation_params)
        else:
            # Chạy single simulation (để debug dễ hơn)
            results = [run_simulation(simulation_params[0])]

        print(f"\n[INFO] All simulations completed!")

    except Exception as e:
        print(f"[ERROR] Failed to run simulations: {e}")
        return

    # Validate results
    valid_results, invalid_count, issues = validate_results(results)

    if invalid_count > 0:
        print(f"\n[WARNING] Found {invalid_count} invalid simulation results:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")

    if not valid_results:
        print("[ERROR] No valid simulation results to save!")
        return

    print(f"\n[INFO] Valid simulations: {len(valid_results)}/{len(results)}")

    # Lưu kết quả
    try:
        csv_file, summary_file = save_results(valid_results, "QMAR_simulation_results")

        print(f"\n[SUCCESS] Simulation completed successfully!")
        print(f"  CSV file: {csv_file}")
        print(f"  Summary file: {summary_file}")

        # In preview của kết quả
        df = pd.DataFrame(valid_results, columns=[
            "Drone", "Drone_Speed", "routing_algorithm", "Seed",
            "Delivery_Ratio", "Mean_Delivery_Time", "Total_Energy_Consumed",
            "Energy_Efficiency", "Throughput"
        ])

        print(f"\n[INFO] Results Preview:")
        print(df.head(10).to_string(index=False))

        if len(df) > 10:
            print(f"... and {len(df) - 10} more rows")

        # Quick statistics
        print(f"\n[INFO] Quick Statistics:")
        print(f"  Average Delivery Ratio: {df['Delivery_Ratio'].mean():.3f}")
        print(f"  Average Delivery Time: {df['Mean_Delivery_Time'].mean():.3f} ms")
        print(f"  Average Energy Efficiency: {df['Energy_Efficiency'].mean():.3f} J/packet")
        print(f"  Average Throughput: {df['Throughput'].mean():.4f} packets/sec")

    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")

        # Fallback: save as simple CSV
        fallback_filename = f"simulation_results_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_fallback = pd.DataFrame(valid_results)
        df_fallback.to_csv(fallback_filename, index=False, header=False)
        print(f"[INFO] Fallback results saved to: {fallback_filename}")


if __name__ == "__main__":
    main()