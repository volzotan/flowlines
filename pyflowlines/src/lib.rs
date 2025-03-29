use flowlines::{FlowlinesConfig, FlowlinesHatcher};
use geo::Point;
use image::imageops::grayscale;
use image::{ImageBuffer, ImageReader, Luma};
use pyo3::prelude::*;
use std::collections::VecDeque;

fn load_grayscale_image(path: &str) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let img = ImageReader::open(path)
        .expect("Could not load image")
        .decode()
        .expect("Could not decode image");

    grayscale(&img)
}

#[pyfunction]
fn hatch(a: usize, b: usize) -> PyResult<String> {
    let map_distance = load_grayscale_image("test_data/map_distance.png");
    let map_angle = load_grayscale_image("test_data/map_angle.png");
    let map_max_length = load_grayscale_image("test_data/map_max_length.png");
    let map_non_flat = load_grayscale_image("test_data/map_non_flat.png");

    let hatcher = FlowlinesHatcher::new(
        FlowlinesConfig::default(),
        &map_distance,
        &map_angle,
        &map_max_length,
        &map_non_flat,
    );
    let lines: Vec<VecDeque<Point>> = hatcher.hatch().unwrap();

    return Ok(String::from("test"));
}

#[pymodule]
fn pyflowlines(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hatch, m)?)?;
    Ok(())
}
