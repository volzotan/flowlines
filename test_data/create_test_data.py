from pathlib import Path
import numpy as np
import rasterio
import cv2
import math
from scipy import ndimage


ELEVATION_FILE = Path("gebco_crop.tif")
OUTPUT_PATH = Path(".")
RESIZE_SIZE = (2000, 2000)
MIN_INCLINATION = 0.01


def normalize_to_uint8(data):
    return (np.iinfo(np.uint8).max * ((data - np.min(data)) / np.ptp(data))).astype(
        np.uint8
    )


def get_slope(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes angle (in rad) and magnitude of the given 2D array of values
    """

    r, c = np.shape(data)
    Y, X = np.mgrid[0:r, 0:c]
    dY, dX = np.gradient(data)

    angles = np.arctan2(dY, dX)
    magnitude = np.hypot(dY, dX)

    return (angles, magnitude)


data = None
with rasterio.open(str(ELEVATION_FILE)) as dataset:
    data = dataset.read()[0]
    data = cv2.resize(data, RESIZE_SIZE)

data[data > 0] = 0  # bathymetry data only

angles, inclination = get_slope(data)

mapping_angle = angles  # float

mapping_non_flat = np.zeros_like(inclination, dtype=np.uint8)
mapping_non_flat[inclination > MIN_INCLINATION] = 255  # uint8

mapping_distance = normalize_to_uint8(data)  # uint8

WINDOW_SIZE = 25
MAX_WIN_VAR = 40000
win_mean = ndimage.uniform_filter(data.astype(float), (WINDOW_SIZE, WINDOW_SIZE))
win_sqr_mean = ndimage.uniform_filter(data.astype(float) ** 2, (WINDOW_SIZE, WINDOW_SIZE))
win_var = win_sqr_mean - win_mean**2
win_var = np.clip(win_var, 0, MAX_WIN_VAR)
win_var = win_var * -1 + MAX_WIN_VAR
win_var = normalize_to_uint8(win_var)

mapping_max_segments = win_var
# mapping_max_segments = np.full_like(angles, int(255 / 2))

mapping_angle = cv2.blur(mapping_angle, (10, 10))
mapping_distance = cv2.blur(mapping_distance, (10, 10))
mapping_max_segments = cv2.blur(mapping_max_segments, (10, 10))

cv2.imwrite(
    str(Path(OUTPUT_PATH, "map_angle.png")),
    normalize_to_uint8(mapping_angle / math.tau),
)
cv2.imwrite(str(Path(OUTPUT_PATH, "map_non_flat.png")), mapping_non_flat)
cv2.imwrite(str(Path(OUTPUT_PATH, "map_distance.png")), mapping_distance)
cv2.imwrite(str(Path(OUTPUT_PATH, "map_max_segments.png")), mapping_max_segments)
