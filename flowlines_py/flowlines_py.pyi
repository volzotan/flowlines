

class FlowlinesConfig:
    """
    A configuration object encapsulating all settings as class attributes
    """

    line_distance: tuple[float, float]
    line_distance_end_factor: float
    line_step_distance: float
    line_max_length: tuple[float, float]
    max_angle_discontinuity: float
    starting_point_init_distance: tuple[float, float]
    seedpoint_extraction_skip_line_segments: int
    max_iterations: int

    def __init__() -> None: ...

def hatch(
    config: FlowlinesConfig,
    map_distance: np.ndarray,
    map_angle: np.ndarray,
    map_max_length: np.ndarray,
    map_non_flat: np.ndarray,
) -> list[list[tuple[float, float]]]:
    """
    Calculate hatching flowlines with the Rust implementation of the algorithm by Jobart and Lefer
    presented in "Creating Evenly-Spaced Streamlines of Arbitrary Density"

    :param config: settings in a FlowlinesConfig instance
    :param map_distance: numpy uint8 ndarray, mapping the config.line_distance values to a pixel location
    :param map_angle: numpy uint8 ndarray, providing the orientation for the line to follow at each pixel location
    :param map_max_length: numpy uint8 ndarray, mapping the config.line_max_length values to a pixel location
    :param map_non_flat: numpy uint8 ndarray, mapping a pixel location to either 0 (flat) or 255 (non flat) terrain
    """