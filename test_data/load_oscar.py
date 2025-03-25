from pathlib import Path

import cv2
from netCDF4 import Dataset
import numpy as np
import math

INPUT_FILE = "oscar_currents_final_20220101.nc"

OUTPUT_PATH = Path(".")
MIN_INCLINATION = 0.01
RESIZE_FACTOR = 3

def normalize_to_uint8(data):
    return (np.iinfo(np.uint8).max * ((data - np.min(data)) / np.ptp(data))).astype(
        np.uint8
    )


with Dataset(INPUT_FILE, "r", format="NETCDF4") as data:
    print(data)

    u = data.variables["u"][:].filled(0)[0, :, :]
    v = data.variables["v"][:].filled(0)[0, :, :]

    u = cv2.rotate(u, cv2.ROTATE_90_COUNTERCLOCKWISE)
    v = cv2.rotate(v, cv2.ROTATE_90_COUNTERCLOCKWISE)

    angles = np.arctan2(u, v)
    magnitude = np.hypot(u, v)

    angles = (angles + math.pi/2) % math.tau

    mapping_angle = angles + math.pi  # center around math.pi (128) so we avoid negative values
    mapping_angle = ((mapping_angle / math.tau) * 255).astype(np.uint8)

    mapping_non_flat = np.zeros_like(magnitude, dtype=np.uint8)
    mapping_non_flat[magnitude > MIN_INCLINATION] = 255  # uint8

    print(np.min(magnitude), np.max(magnitude))
    magnitude = np.clip(magnitude, 0, 1)
    mapping_distance = ~normalize_to_uint8(magnitude)  # uint8

    resize_dimensions = (int(mapping_angle.shape[1] * RESIZE_FACTOR), int(mapping_angle.shape[0] * RESIZE_FACTOR))
    mapping_angle = cv2.resize(mapping_angle, resize_dimensions)
    mapping_non_flat = cv2.resize(mapping_non_flat, resize_dimensions)
    mapping_distance = cv2.resize(mapping_distance, resize_dimensions)

    mapping_max_segments = np.full_like(mapping_angle, 255)

    cv2.imwrite(str(Path(OUTPUT_PATH, "map_angle.png")), mapping_angle)
    cv2.imwrite(str(Path(OUTPUT_PATH, "map_non_flat.png")), mapping_non_flat)
    cv2.imwrite(str(Path(OUTPUT_PATH, "map_distance.png")), mapping_distance)
    cv2.imwrite(str(Path(OUTPUT_PATH, "map_max_segments.png")), mapping_max_segments)
