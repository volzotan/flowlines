import flowlines_py as flowlines
import datetime
from pathlib import Path

import numpy as np
import cv2

INPUT_DIR = Path("..", "test_data")

timer_start = datetime.datetime.now()

map_distance = np.zeros([1000, 1000], dtype=np.uint8)
map_angle = np.zeros([1000, 1000], dtype=np.uint8)
map_max_length = np.zeros([1000, 1000], dtype=np.uint8)
map_flat = np.full([1000, 1000], 255, dtype=np.uint8)

map_distance = cv2.imread(str(Path(INPUT_DIR, "map_distance.png")), cv2.IMREAD_GRAYSCALE)
map_angle = cv2.imread(str(Path(INPUT_DIR, "map_angle.png")), cv2.IMREAD_GRAYSCALE)
map_max_length = cv2.imread(str(Path(INPUT_DIR, "map_max_length.png")), cv2.IMREAD_GRAYSCALE)
map_flat = cv2.imread(str(Path(INPUT_DIR, "map_flat.png")), cv2.IMREAD_GRAYSCALE)

print(f"reading image time: {(datetime.datetime.now()-timer_start).total_seconds():5.2f}s")
timer_start = datetime.datetime.now()

config = flowlines.FlowlinesConfig()
config.line_distance = [20.0, 50.0]

result = flowlines.hatch([1000, 1000], config, map_distance, map_angle, map_max_length, map_flat)

print(f"data transfer + rust computation time: {(datetime.datetime.now()-timer_start).total_seconds():5.2f}s")

