import numpy as np
from numba import njit, prange
import time
from dataclasses import dataclass
from typing import Tuple, Optional

# ============================================================
# 1) Configuration
# ============================================================

@dataclass(frozen=True)
class Hyperparameters:
    n_ants: int = 50
    max_iterations: int = 1000
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.98  # persistence
    nn_size: int = 20
    seed: Optional[int] = None

    def __post_init__(self):

        if not (0.0 <= self.rho < 1.0):
            raise ValueError(
                f"Rho (persistence) must be in range [0, 1). Got {self.rho}. "
                "Rho=1.0 causes division by zero in MMAS limits calculation."
            )

        if self.n_ants <= 0:
            raise ValueError(f"n_ants must be > 0. Got {self.n_ants}")

        if self.max_iterations < 0:
            raise ValueError("max_iterations cannot be negative.")

        if self.nn_size < 1:
            raise ValueError(f"nn_size must be >= 1. Got {self.nn_size}")

# ============================================================
# 2) Numba kernels (hot path)
# ============================================================

@njit(fastmath=True, parallel=True, nogil=True)
def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix for city coordinates (N,2) -> (N,N)."""
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            dist[i, j] = dist[j, i] = d
    return dist


@njit(fastmath=True, parallel=True, nogil=True)
def compute_choice_info(
    pheromone: np.ndarray,
    dists: np.ndarray,
    alpha: float,
    beta: float
) -> np.ndarray:
    """
    Precompute numerator for transition probabilities:
      (tau_ij^alpha) * (eta_ij^beta), where eta = 1/dist.
    """
    n = len(pheromone)
    choice = np.zeros((n, n), dtype=np.float64)

    for i in prange(n):
        for j in range(n):
            if i != j:
                heuristic = 1.0 / (dists[i, j] + 1e-10)
                choice[i, j] = (pheromone[i, j] ** alpha) * (heuristic ** beta)

    return choice

@njit(fastmath=True, nogil=True)
def _roulette_from_candidates(
    current: int,
    candidates: np.ndarray,
    visited: np.ndarray,
    choice_info: np.ndarray
) -> int:
    """
    Selects the next city using roulette wheel selection.

    Optimized to avoid temporary array allocations and fancy indexing,
    improving cache locality and hot-loop performance.
    """
    m = candidates.size
    sum_w = 0.0

    # --- PASS 1: Calculate total probability weight ---
    # We iterate once just to get the sum, avoiding array allocation.
    for i in range(m):
        city = candidates[i]
        if not visited[city]:
            sum_w += choice_info[current, city]

    # --- PASS 2: Roulette Wheel Selection ---
    if sum_w > 0.0:
        # Pick a random point in the cumulative distribution
        r = np.random.random() * sum_w
        
        for i in range(m):
            city = candidates[i]
            if not visited[city]:
                weight = choice_info[current, city]
                r -= weight
                # If r drops below zero, this is our chosen city
                if r <= 0.0:
                    return city
        
        # Floating-point precision safety:
        # If the loop finishes without picking (due to tiny rounding errors),
        # strictly return the last valid candidate found.
        for i in range(m - 1, -1, -1):
            city = candidates[i]
            if not visited[city]:
                return city

    # --- FALLBACK: Deterministic Selection ---
    # Reached if sum_w is 0.0 (all weights are zero) or list exhausted.
    # We simply pick the first unvisited candidate available.
    for i in range(m):
        city = candidates[i]
        if not visited[city]:
            return city

    # No move possible (all candidates visited)
    return -1

@njit(fastmath=True, nogil=True)
def construct_tour(
    start_node: int,
    n_cities: int,
    nn_list: np.ndarray,
    all_cities: np.ndarray,
    choice_info: np.ndarray
) -> np.ndarray:
    """
    Construct a full tour for a single ant.
    """

    visited = np.zeros(n_cities, dtype=np.bool_)
    tour = np.empty(n_cities, dtype=np.int32)

    current = start_node
    visited[current] = True
    tour[0] = current

    for step in range(1, n_cities):
        # NN restricted candidate set
        nxt = _roulette_from_candidates(current, nn_list[current], visited, choice_info)

        # Global fallback if NN list exhausted
        if nxt == -1:
            nxt = _roulette_from_candidates(current, all_cities, visited, choice_info)

        # Safety: should not happen, but keep it robust
        if nxt == -1:
            for city in range(n_cities):
                if not visited[city]:
                    nxt = city
                    break

        tour[step] = nxt
        visited[nxt] = True
        current = nxt

    return tour


@njit(fastmath=True, parallel=True, nogil=True)
def run_colony_step(
    n_ants: int,
    distances: np.ndarray,
    nn_list: np.ndarray,
    all_cities: np.ndarray,
    choice_info: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel construction of tours + tour length evaluation for all ants.
    """
    n_cities = len(distances)
    tours = np.empty((n_ants, n_cities), dtype=np.int32)
    lengths = np.empty(n_ants, dtype=np.float64)

    for k in prange(n_ants):
        start = np.random.randint(0, n_cities)
        tour = construct_tour(start, n_cities, nn_list, all_cities, choice_info)

        l = 0.0
        for i in range(n_cities):
            l += distances[tour[i], tour[(i + 1) % n_cities]]

        tours[k] = tour
        lengths[k] = l

    return tours, lengths


@njit(fastmath=True, parallel=True, nogil=True)
def update_pheromone(
    pheromone: np.ndarray,
    best_tour: np.ndarray,
    best_len: float,
    rho: float,
    limits: Tuple[float, float],
    symmetric: bool
) -> np.ndarray:
    """
    Applies the MMAS (Max-Min Ant System) pheromone update rule:
      1. Global evaporation (persistence) with [min, max] clamping.
      2. Pheromone deposit on the edges of the best tour found.
    """
    tau_min, tau_max = limits
    n = len(pheromone)

    # Safety: Prevent division by zero if best_len is extremely small
    deposit = 1.0 / best_len if best_len > 1e-10 else 0.0

    # --------------------------------------------------------
    # 1) Global Evaporation (Parallel)
    # --------------------------------------------------------
    # We treat the NxN matrix as a single long 1D array (.ravel())
    # This improves load balancing across threads and CPU cache usage.
    flat_view = pheromone.ravel()

    for i in prange(flat_view.size):

        val = flat_view[i] * rho

        # Clamp value within [tau_min, tau_max]
        val = max(val, tau_min)
        val = min(val, tau_max)

        flat_view[i] = val

    # Deposit on Best Tour (Sequential)
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]  # Modulo operator wraps back to start

        # Calculate new pheromone level
        new_val = pheromone[u, v] + deposit

        # Clamp Upper Bound only.
        # (Lower bound is guaranteed because we added a positive deposit 
        # to a value that was already >= tau_min).
        if new_val > tau_max:
            new_val = tau_max

        pheromone[u, v] = new_val

        if symmetric:
            pheromone[v, u] = new_val

    return pheromone

@njit
def rotate_tour_to_zero(tour: np.ndarray) -> np.ndarray:
    """
    Rotate tour so that city 0 is first (canonical form).
    """
    idx = -1
    for i in range(tour.size):
        if tour[i] == 0:
            idx = i
            break
    if idx == -1:
        return tour
    return np.concatenate((tour[idx:], tour[:idx]))


# ============================================================
# 3) Cold-path helpers
# ============================================================

def build_nn_list(distances: np.ndarray, nn_size: int) -> np.ndarray:
    """Nearest-neighbor indices for each city (shape: N x k)."""
    n = distances.shape[0]
    k = min(nn_size, n - 1)
    nn = np.empty((n, k), dtype=np.int32)
    for i in range(n):
        nn[i] = np.argsort(distances[i])[1 : k + 1]
    return nn

@njit(fastmath=True, nogil=True)
def local_search_swap(tour: np.ndarray, dist: np.ndarray, max_passes: int = 10) -> Tuple[np.ndarray, float]:
    n = len(tour)
    # Calculate initial length
    current_len = 0.0
    for i in range(n):
        current_len += dist[tour[i], tour[(i + 1) % n]]
    
    # SAFETY VALVE: Only allow a few full passes over the tour
    # This prevents the loop from running forever on messy initial tours.
    passes = 0
    
    improved = True
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        for i in range(n):
            for j in range(i + 1, n):
                
                u, v = tour[i], tour[j]
                
                # Pre-compute indices for neighbors
                i_prev = (i - 1 + n) % n
                i_next = (i + 1) % n
                j_prev = (j - 1 + n) % n
                j_next = (j + 1) % n
                
                p1, n1 = tour[i_prev], tour[i_next]
                p2, n2 = tour[j_prev], tour[j_next]

                # Delta Calculation
                removed = 0.0
                added = 0.0

                if i_next == j: # Adjacent
                    removed = dist[p1, u] + dist[u, v] + dist[v, n2]
                    added   = dist[p1, v] + dist[v, u] + dist[u, n2]
                elif i == 0 and j == n - 1: # Wrap-around
                    removed = dist[p2, v] + dist[v, u] + dist[u, n1]
                    added   = dist[p2, u] + dist[u, v] + dist[v, n1]
                else: # Non-adjacent
                    removed = dist[p1, u] + dist[u, n1] + dist[p2, v] + dist[v, n2]
                    added   = dist[p1, v] + dist[v, n1] + dist[p2, u] + dist[u, n2]

                delta = added - removed

                if delta < -1e-8:
                    tour[i] = v
                    tour[j] = u
                    current_len += delta
                    improved = True
                    # First Improvement break
                    break 
            
            if improved: 
                break

    return tour, current_len

# ============================================================
# 4) Solver (orchestration)
# ============================================================

class MMAS_Solver:
    """
    Max-Min Ant System (MMAS) solver for TSP.
    """

    def __init__(
        self,
        distances: np.ndarray,
        params: Optional[Hyperparameters] = None,
        symmetric: bool = True
    ):
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("Distance matrix must be square (N x N).")

        if params is None:
            self.params = Hyperparameters()
        else:
            self.params = params

        self.symmetric = symmetric

        self.distances = distances.astype(np.float64, copy=False)
        self.n_cities = self.distances.shape[0]

        # Structures used in kernels
        self.nn_list = build_nn_list(self.distances, self.params.nn_size)
        self.all_cities = np.arange(self.n_cities, dtype=np.int32)

        # State
        self.best_tour: Optional[np.ndarray] = None
        self.best_length: float = np.inf

        # Pheromone bounds/state (MMAS)
        self.tau_max: float = 1.0
        self.tau_min: float = self.tau_max / 1000.0
        self.pheromone = np.full((self.n_cities, self.n_cities), self.tau_max, dtype=np.float64)

    def _update_limits_from_best(self) -> None:
        self.tau_max = 1.0 / ((1.0 - self.params.rho) * self.best_length)
        self.tau_min = max(self.tau_max / 1000.0, 1e-10)

    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float, float]:
        if self.params.seed is not None:
            np.random.seed(self.params.seed)

        t0 = time.time()

        for _ in range(self.params.max_iterations):
            choice_info = compute_choice_info(
                self.pheromone, self.distances, self.params.alpha, self.params.beta
            )

            tours, lengths = run_colony_step(
                self.params.n_ants, self.distances, self.nn_list, self.all_cities, choice_info
            )

            min_idx = int(np.argmin(lengths))
            best_len_it = float(lengths[min_idx])
            best_tour_it = tours[min_idx].copy()
            best_tour_it, best_len_it = local_search_swap(best_tour_it, self.distances)

            if best_len_it < self.best_length:
                self.best_length = best_len_it
                self.best_tour = best_tour_it.copy()
                self._update_limits_from_best()

            # MMAS pheromone update uses current global best
            self.pheromone = update_pheromone(
                self.pheromone,
                self.best_tour,
                self.best_length,
                self.params.rho,
                (self.tau_min, self.tau_max),
                self.symmetric,
            )

        exec_time = time.time() - t0
        final_tour = rotate_tour_to_zero(self.best_tour)

        if verbose:
            print(f"Done in {exec_time:.2f}s. Best Length: {self.best_length:.2f}")

        return final_tour, self.best_length, exec_time


# ============================================================
# 5) Warm-up
# ============================================================

def warm_up_numba() -> None:
    """Force JIT compilation so benchmarks don't include compile time."""
    print("Warm-up: compiling Numba kernels...")
    coords = np.random.rand(10, 2)
    dist = compute_distance_matrix(coords)
    solver = MMAS_Solver(dist, Hyperparameters(n_ants=5, max_iterations=2))
    solver.optimize(verbose=False)
    print("Warm-up: complete.\n")

if __name__ == "__main__":
    pass
