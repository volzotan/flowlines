from pathlib import Path
import numpy as np
import rasterio
import cv2
import math
from scipy import ndimage


ELEVATION_FILE = Path("gebco_crop.tif")
# ELEVATION_FILE = Path("slope_test_2.png")

OUTPUT_PATH = Path(".")
RESIZE_SIZE = (2000, 2000)
MIN_INCLINATION = 0.01

BLUR = 20


def blur(img: np.ndarray) -> np.ndarray:
    return cv2.blur(img, (BLUR, BLUR))


def normalize_to_uint8(data):
    return (np.iinfo(np.uint8).max * ((data - np.min(data)) / np.ptp(data))).astype(
        np.uint8
    )


def get_slope(data: np.ndarray, sampling_step: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes angle (in rad) and magnitude of the given 2D array of values
    """
    test_slice = data[::sampling_step, ::sampling_step]
    dY, dX = np.gradient(test_slice)  # order! Y X

    angles = np.arctan2(dY, dX)
    magnitude = np.hypot(dY, dX)

    if sampling_step > 1:
        angles = cv2.resize(angles, data.shape)
        magnitude = cv2.resize(magnitude, data.shape)

    return (angles, magnitude)


data = None
with rasterio.open(str(ELEVATION_FILE)) as dataset:
    data = dataset.read()[0]
    data = cv2.resize(data, RESIZE_SIZE)

data[data > 0] = 0  # bathymetry data only

angles, inclination = get_slope(data, 4)

# follow the flowlines
# angles = (angles + math.pi/2) % math.tau

mapping_angle = angles  # float
mapping_angle += math.pi  # center around math.pi (128) so we avoid negative values
mapping_angle = ((mapping_angle / math.tau) * 255).astype(np.uint8)

mapping_non_flat = np.zeros_like(inclination, dtype=np.uint8)
mapping_non_flat[inclination > MIN_INCLINATION] = 255  # uint8

mapping_distance = normalize_to_uint8(data)  # uint8

WINDOW_SIZE = 25
MAX_WIN_VAR = 40000
win_mean = ndimage.uniform_filter(data.astype(float), (WINDOW_SIZE, WINDOW_SIZE))
win_sqr_mean = ndimage.uniform_filter(
    data.astype(float) ** 2, (WINDOW_SIZE, WINDOW_SIZE)
)
win_var = win_sqr_mean - win_mean**2
win_var = np.clip(win_var, 0, MAX_WIN_VAR)
win_var = win_var * -1 + MAX_WIN_VAR
win_var = normalize_to_uint8(win_var)

mapping_max_segments = win_var
# mapping_max_segments = np.full_like(angles, int(255 / 2))

mapping_angle = blur(mapping_angle)
mapping_distance = blur(mapping_distance)
mapping_max_segments = blur(
    mapping_max_segments,
)

cv2.imwrite(str(Path(OUTPUT_PATH, "map_angle.png")), mapping_angle)
cv2.imwrite(str(Path(OUTPUT_PATH, "map_non_flat.png")), mapping_non_flat)
cv2.imwrite(str(Path(OUTPUT_PATH, "map_distance.png")), mapping_distance)
cv2.imwrite(str(Path(OUTPUT_PATH, "map_max_segments.png")), mapping_max_segments)
