use chrono::Utc;
use flowlines_rs::{FlowlinesConfig, FlowlinesHatcher};
use geo::Point;
use image::imageops::grayscale;
use image::{ImageBuffer, ImageReader, Luma, Rgb, imageops};
use imageproc::drawing::draw_antialiased_line_segment_mut;
use imageproc::pixelops::interpolate;
use std::collections::VecDeque;

fn load_grayscale_image(path: &str) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let img = ImageReader::open(path)
        .expect("Could not load image")
        .decode()
        .expect("Could not decode image");

    grayscale(&img)
}

fn draw_lines_on_image(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, lines: &[VecDeque<Point>]) {
    for line in lines {
        let point_vec = line;
        for i in 1..point_vec.len() {
            let start = point_vec[i - 1];
            let end = point_vec[i];

            draw_antialiased_line_segment_mut(
                image,
                (start.x() as i32, start.y() as i32),
                (end.x() as i32, end.y() as i32),
                Rgb::from([0, 0, 0]),
                interpolate,
            );
        }
    }
}

fn main() {
    let timer_start = Utc::now();

    let map_distance = load_grayscale_image("test_data/map_distance.png");
    let map_angle = load_grayscale_image("test_data/map_angle.png");
    let map_max_length = load_grayscale_image("test_data/map_max_length.png");
    let map_flat = load_grayscale_image("test_data/map_flat.png");

    let timer_diff = Utc::now() - timer_start;
    println!(
        "Image loading in {:.3}s",
        timer_diff.num_milliseconds() as f64 / 1_000.0
    );
    let timer_start = Utc::now();

    let map_angle = imageops::blur(&map_angle, 2.0);

    let width = map_distance.width() * 2;
    let height = map_distance.height() * 2;

    let mut config = FlowlinesConfig::default();
    config.starting_point_init_distance =
        [config.line_distance[0] * 1.5, config.line_distance[1] * 1.5];
    config.line_max_length = [400.0, 500.0];
    config.line_distance = [6.0, 12.0];

    let hatcher = FlowlinesHatcher::new(
        [width, height],
        &config,
        &map_distance,
        &map_angle,
        &map_max_length,
        &map_flat,
    );
    let lines: Vec<VecDeque<Point>> = hatcher.hatch().unwrap();

    let timer_diff = Utc::now() - timer_start;
    println!(
        "Hatching in {:.3}s",
        timer_diff.num_milliseconds() as f64 / 1_000.0
    );

    let mut img_output: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_pixel(width, height, Rgb([255, 255, 255]));
    draw_lines_on_image(&mut img_output, &lines);

    img_output
        .save("output.jpg")
        .expect("Failed to save output");
}
