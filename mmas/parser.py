import numpy as np
import re
from pathlib import Path
from typing import Dict

# ============================================================
# 0) I/O Helpers
# ============================================================

def parse_tsplib_to_matrix(filepath: Path) -> Dict:
    """
    Parses a TSPLIB file (.tsp or .atsp) and returns a dictionary containing:
    - 'matrix': The N x N distance matrix (np.float64)
    - 'type': 'symmetric' or 'asymmetric'
    - 'coords': The N x 2 coordinate array (np.float64) if available, else None
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Detect dimension
    dim_match = re.search(r"DIMENSION\s*:\s*(\d+)", content)
    if not dim_match:
        raise ValueError("Could not find DIMENSION in file.")
    n = int(dim_match.group(1))

    # Detect edge weight type
    type_match = re.search(r"EDGE_WEIGHT_TYPE\s*:\s*(\w+)", content)
    weight_type = type_match.group(1) if type_match else "UNKNOWN"

    file_ext = filepath.suffix.lower()
    problem_type = "asymmetric" if file_ext == '.atsp' else "symmetric"

    # ==========================================
    # CASE A: Coordinates (EUC_2D, CEIL_2D, etc.)
    # ==========================================
    if "EUC_2D" in weight_type or "NODE_COORD_SECTION" in content:
        coords = np.zeros((n, 2), dtype=np.float64)
        lines = content.split('\n')
        start_reading = False
        idx = 0

        for line in lines:
            if "NODE_COORD_SECTION" in line:
                start_reading = True
                continue
            if "EOF" in line:
                break
            if start_reading:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # TSPLIB usually: NodeID X Y
                    coords[idx, 0] = float(parts[1])
                    coords[idx, 1] = float(parts[2])
                    idx += 1
                    if idx >= n: break

        # Compute Euclidean Distance Matrix
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = np.sqrt(np.sum((coords[i] - coords[j])**2))
                    # Standard TSPLIB often uses nint(), keeping float for general use
                    dist_matrix[i, j] = d

        return {
            'matrix': dist_matrix,
            'type': 'symmetric',
            'coords': coords
        }

    # ==========================================
    # CASE B: Explicit Matrix (FULL_MATRIX)
    # ==========================================
    elif "EDGE_WEIGHT_SECTION" in content:
        raw_data = []
        lines = content.split('\n')
        start_reading = False

        for line in lines:
            if "EDGE_WEIGHT_SECTION" in line:
                start_reading = True
                continue
            if "EOF" in line or "DISPLAY_DATA_SECTION" in line:
                break
            if start_reading:
                tokens = [float(x) for x in line.strip().split() if x]
                raw_data.extend(tokens)

        matrix = np.array(raw_data, dtype=np.float64)
        if matrix.size != n * n:
             raise ValueError(f"Matrix size mismatch. Expected {n*n}, got {matrix.size}")

        return {
            'matrix': matrix.reshape((n, n)),
            'type': problem_type,
            'coords': None # Explicit matrix usually implies no coords
        }

    else:
        raise ValueError(f"Unsupported TSPLIB format: {weight_type}")


def load_optimal_values(base_path: Path) -> Dict[str, float]:
    """
    Reads bestSolutions.txt from ./tsp and ./atsp folders.
    Returns a dict { 'instance_name': optimal_value }.
    """
    solutions = {}
    paths = [
        base_path / "tsp" / "bestSolutions.txt",
        base_path / "atsp" / "bestSolutions.txt"
    ]

    for p in paths:
        if p.exists():
            with open(p, 'r') as f:
                for line in f:
                    if ':' in line:
                        parts = line.strip().split(':')
                        name = parts[0].strip()
                        try:
                            val = float(parts[1].strip())
                            solutions[name] = val
                        except ValueError:
                            pass
    return solutions


def save_tour_file(filepath: Path, tour: np.ndarray, length: float):
    """Saves the tour in standard TSPLIB format."""
    n = len(tour)
    with open(filepath, 'w') as f:
        f.write(f"NAME : {filepath.stem}\n")
        f.write(f"TYPE : TOUR\n")
        f.write(f"DIMENSION : {n}\n")
        f.write(f"TOUR_LENGTH : {length}\n")
        f.write(f"SECTION : TOUR_SECTION\n")
        for city_idx in tour:
            # TSPLIB uses 1-based indexing for output
            f.write(f"{city_idx + 1}\n")
        f.write("-1\n")
        f.write("EOF\n")

if __name__ == "__main__":
    pass
