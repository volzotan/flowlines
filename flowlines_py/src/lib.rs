use flowlines_rs;
use geo::Point;
use image::{GrayImage, ImageBuffer};
use numpy::ndarray::ArrayViewD;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::{
    Bound, FromPyObject, PyResult, pymodule,
    types::{PyAnyMethods, PyDictMethods, PyModule},
};
use std::collections::VecDeque;
use std::error::Error;

fn copy_array_into_grayimage(arr: PyReadonlyArrayDyn<u8>) -> Option<GrayImage> {
    let x: ArrayViewD<'_, u8> = arr.as_array();
    let vec = x.as_standard_layout().iter().copied().collect();
    ImageBuffer::from_raw(x.shape()[1] as u32, x.shape()[0] as u32, vec)
}

#[pyclass(get_all, set_all)]
struct FlowlinesConfig {
    line_distance: [f64; 2],
    line_distance_end_factor: f64,
    line_step_distance: f64,
    line_max_length: [f64; 2],
    max_angle_discontinuity: f64,
    starting_point_init_distance: [i32; 2],
    seedpoint_extraction_skip_line_segments: usize,
    max_iterations: u32,
}

#[pymethods]
impl FlowlinesConfig {
    #[new]
    pub fn new() -> Self {
        let c = flowlines_rs::FlowlinesConfig::default();
        FlowlinesConfig::from(c)
    }
}

impl Into<flowlines_rs::FlowlinesConfig> for &FlowlinesConfig {
    fn into(self) -> flowlines_rs::FlowlinesConfig {
        flowlines_rs::FlowlinesConfig {
            line_distance: self.line_distance,
            line_distance_end_factor: self.line_distance_end_factor,
            line_step_distance: self.line_step_distance,
            line_max_length: self.line_max_length,
            max_angle_discontinuity: self.max_angle_discontinuity,
            starting_point_init_distance: self.starting_point_init_distance,
            seedpoint_extraction_skip_line_segments: self.seedpoint_extraction_skip_line_segments,
            max_iterations: self.max_iterations,
        }
    }
}

impl From<flowlines_rs::FlowlinesConfig> for FlowlinesConfig {
    fn from(c: flowlines_rs::FlowlinesConfig) -> Self {
        FlowlinesConfig {
            line_distance: c.line_distance,
            line_distance_end_factor: c.line_distance_end_factor,
            line_step_distance: c.line_step_distance,
            line_max_length: c.line_max_length,
            max_angle_discontinuity: c.max_angle_discontinuity,
            starting_point_init_distance: c.starting_point_init_distance,
            seedpoint_extraction_skip_line_segments: c.seedpoint_extraction_skip_line_segments,
            max_iterations: c.max_iterations,
        }
    }
}

#[pyfunction]
fn hatch<'py>(
    config: &FlowlinesConfig,
    map_distance: PyReadonlyArrayDyn<'py, u8>,
    map_angle: PyReadonlyArrayDyn<'py, u8>,
    map_max_length: PyReadonlyArrayDyn<'py, u8>,
    map_non_flat: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Vec<Vec<[f64; 2]>>> {

    let config: flowlines_rs::FlowlinesConfig = config.into();
    let map_distance = copy_array_into_grayimage(map_distance).expect("could not read map_distance");
    let map_angle = copy_array_into_grayimage(map_angle).expect("could not read map_angle");
    let map_max_length = copy_array_into_grayimage(map_max_length).expect("could not read map_max_length");
    let map_non_flat = copy_array_into_grayimage(map_non_flat).expect("could not read map_non_flat");

    let hatcher = flowlines_rs::FlowlinesHatcher::new(
        &config,
        &map_distance,
        &map_angle,
        &map_max_length,
        &map_non_flat,
    );

    let lines = hatcher.hatch().unwrap();
    let lines_vec: Vec<Vec<[f64; 2]>> = lines
        .iter()
        .map(|ls: &VecDeque<Point>| ls.iter().map(|p: &Point| [p.x(), p.y()]).collect())
        .collect();

    Ok(lines_vec)
}

#[pymodule]
fn flowlines_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FlowlinesConfig>()?;
    m.add_function(wrap_pyfunction!(hatch, m)?)?;
    Ok(())
}
