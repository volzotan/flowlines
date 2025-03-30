use geo::Point;
use image::imageops::grayscale;
use image::{GenericImageView, GrayImage, ImageBuffer, ImageReader, Luma};
use numpy::ndarray::ArrayViewD;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::{
    Bound, FromPyObject, PyResult, pymodule,
    types::{PyAnyMethods, PyDictMethods, PyModule},
};
use std::collections::VecDeque;

use flowlines;

fn load_grayscale_image(path: &str) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let img = ImageReader::open(path)
        .expect("Could not load image")
        .decode()
        .expect("Could not decode image");

    grayscale(&img)
}

fn copy_array_into_grayimage(arr: PyReadonlyArrayDyn<u8>) -> Option<GrayImage> {
    let x: ArrayViewD<'_, u8> = arr.as_array();
    let (height, width) = (x.shape()[0] as u32, x.shape()[1] as u32);
    let vec = x.as_standard_layout().iter().copied().collect();
    ImageBuffer::from_raw(width, height, vec)
}

#[pyclass]
struct PyFlowlinesConfig(flowlines::FlowlinesConfig);

#[pymethods]
impl PyFlowlinesConfig {
    #[new]
    pub fn new() -> Self {
        PyFlowlinesConfig {
            0: Default::default(),
        }
    }
}

#[pyfunction]
fn hatch<'py>(
    config: &PyFlowlinesConfig,
    map_distance: PyReadonlyArrayDyn<'py, u8>,
    map_angle: PyReadonlyArrayDyn<'py, u8>,
    map_max_length: PyReadonlyArrayDyn<'py, u8>,
    map_non_flat: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Vec<Vec<[f64; 2]>>> {
    // let map_distance = load_grayscale_image("test_data/map_distance.png");
    // let map_angle = load_grayscale_image("test_data/map_angle.png");
    // let map_max_length = load_grayscale_image("test_data/map_max_length.png");
    // let map_non_flat = load_grayscale_image("test_data/map_non_flat.png");

    let map_distance =
        copy_array_into_grayimage(map_distance).expect("could not read map_distance");
    let map_angle = copy_array_into_grayimage(map_angle).expect("could not read map_angle");
    let map_max_length =
        copy_array_into_grayimage(map_max_length).expect("could not read map_max_length");
    let map_non_flat =
        copy_array_into_grayimage(map_non_flat).expect("could not read map_non_flat");

    let hatcher = flowlines::FlowlinesHatcher::new(
        &config.0,
        &map_distance,
        &map_angle,
        &map_max_length,
        &map_non_flat,
    );
    let lines: Vec<VecDeque<Point>> = hatcher.hatch().unwrap();

    let lines_vec: Vec<Vec<[f64; 2]>> = lines
        .iter()
        .map(|ls: &VecDeque<Point>| ls.iter().map(|p: &Point| [p.x(), p.y()]).collect())
        .collect();

    Ok(lines_vec)
}

#[pymodule]
fn pyflowlines(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFlowlinesConfig>()?;
    m.add_function(wrap_pyfunction!(hatch, m)?)?;
    Ok(())
}
