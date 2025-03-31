use std::borrow::Cow;
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
use std::f64::consts::PI;
use flowlines_rs;

fn copy_array_into_grayimage(arr: PyReadonlyArrayDyn<u8>) -> Option<GrayImage> {
    let x: ArrayViewD<'_, u8> = arr.as_array();
    let vec = x.as_standard_layout().iter().copied().collect();
    ImageBuffer::from_raw(x.shape()[1] as u32, x.shape()[0] as u32, vec)
}

fn copy_array_into_grayimage2(arr: PyReadonlyArrayDyn<u8>) -> Option<GrayImage> {
    let x: ArrayViewD<'_, u8> = arr.as_array();

    let raw_data = x.as_slice().expect("Array must be contiguous");
    let data: Cow<[u8]> = Cow::Borrowed(raw_data);

    // Create an ImageBuffer using the borrowed data
    ImageBuffer::from_raw(x.shape()[1] as u32, x.shape()[0] as u32, data.into())
}

// fn array_into_imageview(arr: PyReadonlyArrayDyn<u8>) -> Option<&dyn GenericImageView<Pixel = Luma<u8>>> {
//     let x: ArrayViewD<'_, u8> = arr.as_array();
//
//     // let raw_data = x.as_slice().expect("Array must be contiguous");
//     let raw_data = arr.as_slice().expect("Array must be contiguous");
//     let image: &dyn GenericImageView<Pixel = Luma<u8>> = raw_data;
//     println!("{:?}", image.get_pixel(10, 10));
//     Some(image)
// }

#[pyclass]
struct PyFlowlinesConfig2 {
    foo: String,
    bar: f64,
}

#[pymethods]
impl PyFlowlinesConfig2 {
    #[new]
    pub fn new() -> Self {
        Default::default()
    }


}

impl Default for PyFlowlinesConfig2 {
    fn default() -> PyFlowlinesConfig2 {
        PyFlowlinesConfig2 {
            foo: String::from("default"),
            bar: 0.0
        }
    }
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
    config: &FlowlinesConfig,
    map_distance: PyReadonlyArrayDyn<'py, u8>,
    map_angle: PyReadonlyArrayDyn<'py, u8>,
    map_max_length: PyReadonlyArrayDyn<'py, u8>,
    map_non_flat: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Vec<Vec<[f64; 2]>>> {

    let map_distance = copy_array_into_grayimage(map_distance).expect("could not read map_distance");
    let map_angle = copy_array_into_grayimage(map_angle).expect("could not read map_angle");
    let map_max_length = copy_array_into_grayimage(map_max_length).expect("could not read map_max_length");
    let map_non_flat = copy_array_into_grayimage(map_non_flat).expect("could not read map_non_flat");

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
    m.add_class::<PyFlowlinesConfig2>()?;

    m.add_class::<PyFlowlinesConfig>()?;
    m.add_function(wrap_pyfunction!(hatch, m)?)?;
    Ok(())
}
