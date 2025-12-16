# Max-Min Ant System (MMAS) for TSPThis project implements a high-performance Max-Min Ant System (MMAS) solver for the Traveling Salesperson Problem (TSP). It is designed to handle both standard academic benchmarks (TSPLIB) and real-world road network optimization via OSRM.

The core algorithm is accelerated using **Numba** to ensure computational efficiency, making it capable of solving complex instances quickly.

## Key Features* **Algorithm:** Max-Min Ant System (MMAS) with `min_max` pheromone clamping.

* **Performance:** Critical paths (distance calculation, roulette selection, pheromone updates) are JIT-compiled using **Numba** (parallel/fastmath enabled).
* **Local Search:** Implements 2-opt local search heuristic to refine solutions.
* **Input Formats:**
* Parses standard TSPLIB files (`.tsp`, `.atsp`).
* Supports GeoJSON inputs for real-world coordinate extraction.


* **Real-World Routing:** Integrates with OSRM (Open Source Routing Machine) to calculate duration matrices for actual street networks.

## Project Structure* `mmas/`: Core package containing the source code.
* `algo.py`: Implementation of the MMAS algorithm, Numba kernels, and `Hyperparameters` configuration.
* `parser.py`: Utilities for parsing TSPLIB files and loading optimal solutions.
* `utilities.py`: Tools for OSRM API interaction (Table and Route services).


* `datasets/`: Directory for TSPLIB instances and GeoJSON data.
* `main.py`: Entry point for executing the solver.

##Installation

This project requires **Python 3.12+**.

**Dependencies:**

* `numpy`
* `numba`
* `requests`

**Using pip:**

```bash
pip install numpy numba requests

```

**Using uv:**

```bash
uv sync

```

## UsageTo run the solver, execute the main script:

```bash
uv run python main.py

```

### Configuration

The solver behavior is controlled via the `Hyperparameters` class in `mmas/algo.py`. Default settings include:

* **n_ants**: 50
* **max_iterations**: 1000
* **alpha**: 1.0 (Pheromone importance)
* **beta**: 2.0 (Heuristic importance)
* **rho**: 0.98 (Evaporation rate)
* **nn_size**: 20 (Nearest-neighbor candidate list size)

### OSRM Setup

For real-world routing (calculating travel times between coordinates), the project connects to an OSRM server.

* **Default Host:** `localhost:5000`
* **Configuration:** Modify the `HOST` variable in `mmas/utilities.py` to point to your specific OSRM instance or a public demo server (e.g., `router.project-osrm.org`).

## Outputs

The solver outputs:

1. **Tour Files:** Standard TSPLIB format files containing the optimized sequence of nodes.
2. **Console Logs:** Execution time, best tour length found, and optimization progress.

