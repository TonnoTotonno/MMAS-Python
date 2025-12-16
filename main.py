import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from mmas.algo import (
            warm_up_numba, Hyperparameters, MMAS_Solver
)
from mmas.parser import (
            load_optimal_values, save_tour_file,
            parse_tsplib_to_matrix
)

from mmas.utilities import (
            get_duration_matrix, extract_points_from_geojson,
            get_route_geometry, plot_route_vector
)

if __name__ == "__main__":
    base_dir = Path.cwd()
    dataset_dir = base_dir / "datasets"
    res_dir = base_dir / "aco_results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load optimal values map
    optimal_values_map = load_optimal_values(dataset_dir)

    # --- PARAMS (Optimized for Quality) ---
    PARAMS = {
        "n_ants": 100,
        "max_iterations": 1000,
        "alpha": 1.0,
        "beta": 2.25,
        "rho": 0.98,
        "seed": 42
    }

    warm_up_numba()

    instances = {
        "tsp": (
            "berlin52",
            "eil101", "kroA100", "pr124", "pcb442"
            ),
        "atsp": ("ftv33", "ftv44", "ftv64")
    }

    csv_path = res_dir / "results.csv"
    file_exists = csv_path.exists()

    with open(csv_path, mode='a', newline='') as csv_file:
        # Added 'optimal_val' and 'gap_pct' to headers
        fieldnames = ["instance", "type", "best_length", "optimal_val", "gap_pct", "time_seconds"] + \
            list(PARAMS.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        warm_up_numba()

        print(f"{'='*50}\nMODULAR ACO (Symmetric & Asymmetric)\n{'='*50}")

        for tsp_type, names in instances.items():
            for name in names:
                fpath = dataset_dir / tsp_type / f"{name}.{tsp_type}"

                if not fpath.exists():
                    print(f"[SKIP] {name} not found in {fpath.parent}")
                    continue

                print(f"\n>>> Processing: {name}")

                try:
                    # 1. Parse
                    data = parse_tsplib_to_matrix(fpath)
                    dist_matrix = data['matrix']
                    p_type = data['type']
                    coords = data.get('coords')

                    is_sym = (p_type == "symmetric")

                    print(f"    Type: {p_type.upper()}")
                    print(f"    Size: {dist_matrix.shape[0]} cities")

                    # 2. Optimize
                    if is_sym:
                        params = Hyperparameters(**PARAMS)
                    else:
                        params = Hyperparameters(**PARAMS, nn_size=50)
                    solver = MMAS_Solver(dist_matrix, params, symmetric=is_sym)

                    # Run logic
                    best_tour, best_len, duration = solver.optimize(verbose=True)

                    # 3. Calculate Optimality Gap
                    opt_val = optimal_values_map.get(name)
                    gap_pct = None
                    if opt_val is not None and opt_val > 0:
                        gap = (best_len - opt_val) / opt_val
                        gap_pct = round(gap * 100, 2)
                        print(f"    Optimal: {opt_val} | Gap: {gap_pct}%")
                    else:
                        print(f"    Optimal: Not Found")

                    # 4. Save Tour File
                    tour_path = res_dir / f"{name}.tour"
                    save_tour_file(tour_path, best_tour, best_len)
                    print(f"    Saved tour: {tour_path.name}")

                    # 5. Save Plot (if coords exist)
                    if coords is not None:
                        plt.figure(figsize=(10, 8))
                        # Close the loop by appending the first city to the end
                        path_indices = np.append(best_tour, best_tour[0])

                        plt.plot(coords[path_indices, 0], coords[path_indices, 1],
                                 c='blue', zorder=1, linewidth=1, label=f"Len: {best_len:.2f}")
                        plt.scatter(coords[:, 0], coords[:, 1], c='red', s=10, zorder=2)

                        # Mark start
                        plt.scatter(coords[best_tour[0], 0], coords[best_tour[0], 1],
                                    c='green', s=50, marker='*', zorder=3, label="Start")

                        plt.title(f"ACO Tour: {name} (Gap: {gap_pct if gap_pct is not None else 'N/A'}%)")
                        plt.legend()

                        plot_path = res_dir / f"{name}.png"

                        # Create an ordered array of coordinates based on the tour
                        ordered_coords = coords[best_tour]
                        # Close the loop (add the first city to the end)
                        ordered_coords = np.vstack([ordered_coords, ordered_coords[0]])

                        # Save to a space-separated .dat file
                        dat_path = res_dir / f"{name}_tour.dat"
                        np.savetxt(dat_path, ordered_coords, header="x y", comments="", fmt="%.4f")


                        # plot_path = res_dir / f"{name}.pgf"
                        plt.savefig(plot_path)
                        plt.close() # Important to free memory and avoid overlaps
                        print(f"    Saved plot: {plot_path.name}")

                    # 6. Log to CSV
                    row = {
                        "instance": name,
                        "type": p_type,
                        "best_length": f"{best_len:.2f}",
                        "optimal_val": opt_val if opt_val else "N/A",
                        "gap_pct": gap_pct if gap_pct is not None else "N/A",
                        "time_seconds": f"{duration:.2f}",
                        **PARAMS
                    }
                    writer.writerow(row)
                    csv_file.flush()

                except Exception as e:
                    print(f"    [ERROR] {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\n{'='*50}\nOPTIMIZING GENOA ROUTE\n{'='*50}")

    # 1. Load Genoa Data
    geojson_path = base_dir / 'datasets' / 'geojson' / 'genoa.geojson'
    with open(geojson_path, 'r', encoding='utf-8') as f:
        json_string = f.read()

    coords_array, names_list = extract_points_from_geojson(json_string)
    print(f"Loaded {len(coords_array)} locations in Genoa.")

    dist_matrix = get_duration_matrix(coords_array)

    genoa_params = Hyperparameters(
        n_ants=100, 
        max_iterations=1000, 
        alpha=1.0, 
        beta=2.25, 
        rho=0.98, 
        seed=42
    )

    solver = MMAS_Solver(dist_matrix, genoa_params, symmetric=False)

    print("Running ACO optimization...")
    best_tour, best_len, duration = solver.optimize(verbose=False)
    print(f"Optimization finished. Est. Time: {best_len:.2f} seconds")
    print(f"Execution time {duration:.2f} seconds")
    print(best_tour)

    genoa_tour_path = res_dir / "unige_mmas.tour"

    save_tour_file(genoa_tour_path, best_tour, best_len)
    print(f"Saved Genoa tour to: {genoa_tour_path}")

