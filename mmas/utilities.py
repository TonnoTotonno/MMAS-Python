import numpy as np
import requests
from typing import Literal
import json

# OSRM server address
#HOST = "router.project-osrm.org" # just for demos
HOST = "localhost:5000"

def get_duration_matrix(
    coordinates: np.ndarray, 
    profile: Literal["driving", "walking", "foot"] = "foot",
    host: str = HOST
) -> np.ndarray:
    """
    Fetches the N x N duration matrix (in seconds) for a set of coordinates 
    using the OSRM Table service.

    Args:
        coordinates (np.ndarray): Shape (N, 2) array of [Longitude, Latitude].
        profile (str): 'driving', 'walking', or 'foot'.
        host (str): OSRM server address.

    Returns:
        np.ndarray: N x N matrix where M[i][j] is the travel time from i to j in seconds.
    """
    
    # 1. format profile
    valid_profiles = {
        "driving": "driving",
        "walking": "walking",
        "foot": "walking"
    }
    if profile not in valid_profiles:
        raise ValueError(f"Invalid profile. Choose from: {list(valid_profiles.keys())}")
    
    service_profile = valid_profiles[profile]

    # 2. Format Coordinates into a single string "lon,lat;lon,lat;..."
    # Note: OSRM public demo server limits requests to ~100 coordinates.
    coords_formatted = [f"{lon},{lat}" for lon, lat in coordinates]
    coordinates_str = ";".join(coords_formatted)

    # 3. Construct URL
    url = f"http://{host}/table/v1/{service_profile}/{coordinates_str}"

    # We only need durations (default), but being explicit is good.
    params = {
        "annotations": "duration"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # 'durations' is a list of lists (matrix)
        matrix = np.array(data["durations"], dtype=float)

        # OSRM might return None for unreachable points; replace with infinity
        # or a very large number to discourage the ACO from using that edge.
        matrix[np.isnan(matrix)] = 1e9 

        return matrix

    except requests.RequestException as e:
        print(f"OSRM Table API Error: {e}")
        # Return an empty matrix or handle appropriately
        return np.zeros((len(coordinates), len(coordinates)))

def extract_points_from_geojson(geojson_str: str):
    """
    Parses a GeoJSON string, sorts features by 'id' (int), 
    and returns coordinates and names as aligned arrays.
    
    Returns:
        coords (np.ndarray): Shape (N, 2) -> [[lon, lat], ...]
        names (list): List of strings matching the coords order.
    """
    data = json.loads(geojson_str)
    
    # Sort features by ID (converting string id "23" to int 23 for proper sorting)
    # If some features don't have an ID, use a fallback (e.g., -1)
    sorted_features = sorted(
        data['features'], 
        key=lambda f: int(f['properties'].get('id', -1))
    )
    
    coords = []
    names = []
    
    for feature in sorted_features:
        # GeoJSON is [Lon, Lat]
        c = feature['geometry']['coordinates']
        n = feature['properties'].get('name', 'Unknown')
        
        coords.append(c)
        names.append(n)
        
    return np.array(coords), names

def get_path_duration(
    coordinates: np.ndarray, 
    profile: Literal["driving", "walking", "foot"] = "walking",
    host: str = HOST
) -> float:
    """
    Calculates the total duration of a path given a sequence of coordinates.

    Args:
        coordinates (np.ndarray): A NumPy array of shape (N, 2). 
                                  Format must be [LONGITUDE, LATITUDE].
        profile (str): Transport mode. Options: 'driving', 'walking', or 'foot'.

    Returns:
        float: Total duration in seconds. Returns -1.0 if an error occurs.
    """

    valid_profiles = {
        "driving": "driving",
        "walking": "walking",
        "foot": "walking" # Common alias
    }

    if profile not in valid_profiles:
        raise ValueError(f"Invalid profile. Choose from: {list(valid_profiles.keys())}")

    service_profile = valid_profiles[profile]

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Coordinates array must have shape (N, 2)")

    coords_formatted = [f"{lon},{lat}" for lon, lat in coordinates]
    coordinates_str = ";".join(coords_formatted)

    url = f"http://{host}/route/v1/{service_profile}/{coordinates_str}"

    # Optimization: 'overview=false' reduces JSON size (no geometry returned)
    params = {
        "overview": "false", 
        "steps": "false"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # 'duration' at route level is the sum of all legs
        return data["routes"][0]["duration"]

    except (requests.RequestException, KeyError, IndexError) as e:
        print(f"OSRM Error: {e}")
        return -1.0


if __name__ == "__main__":
    pass
