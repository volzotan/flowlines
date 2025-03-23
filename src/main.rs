use std::collections::VecDeque;
use image::{GrayImage, ImageBuffer, ImageReader, Rgb, Pixel, Luma};
use image::imageops::grayscale;
use std::f64::consts::PI;
use geo::{Point};
use rstar::RTree;
use std::error::Error;
use imageproc::drawing::draw_antialiased_line_segment_mut;
use imageproc::pixelops::interpolate;
use chrono::Utc;

const MAX_ITERATIONS: u32 = 100_000;
const STARTING_POINT_INIT_DISTANCE_WIDTH: u32 = 1;
const STARTING_POINT_INIT_DISTANCE_HEIGHT: u32 = 1;
const SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS: usize = 10;

struct FlowlineHatcher<'a> {
    line_distance: [f64; 2],
    line_step_distance: f64,
    line_max_segments: [usize; 2],
    mapping_factor: f64,
    map_line_distance: &'a GrayImage,
    map_angle: &'a GrayImage,
    map_line_max_segments: &'a GrayImage,
    map_non_flat: &'a GrayImage,
    _bbox: [i32; 4]
}

impl<'a> FlowlineHatcher<'a> {

    fn new(map_line_distance: &'a GrayImage,
           map_angle: &'a GrayImage,
           map_line_max_segments: &'a GrayImage,
           map_non_flat: &'a GrayImage,
    ) -> Self {

        let bbox = [0, 0, map_line_distance.width() as i32, map_line_distance.height() as i32];

        FlowlineHatcher {
            line_distance: [5.0, 20.0],
            line_step_distance: 2.5,
            line_max_segments: [15, 30],
            mapping_factor: 1.0,
            map_line_distance,
            map_angle,
            map_line_max_segments,
            map_non_flat,
            _bbox: bbox,
        }
    }

    fn _map_angle(&self, x: i32, y: i32) -> f64 {
        PI/4.0
        // 0.0
    }

    fn _map_line_distance(&self, x: f64, y: f64) -> f64 {
        let pixel = self.map_line_distance.get_pixel(x as u32, y as u32);
        self.line_distance[0] + (self.line_distance[1] - self.line_distance[0]) * (pixel[0] as f64) / 255.0
    }

    fn _map_line_max_segments(&self, x: f64, y: f64) -> usize {
        self.line_max_segments[1]
    }

    fn _collision(&self, tree: &RTree<Point>, x: f64, y: f64) -> bool {
        match tree.nearest_neighbor(&Point::new(x, y)) {
            Some(p) => {
                let dist = ((p.x() - x).powi(2) + (p.y() - y).powi(2)).sqrt();
                return dist < self._map_line_distance(x, y)
            },
            None => false
        }
    }

    fn _next_point(&self, tree: &RTree<Point>, p: &Point, forwards: bool) -> Option<Point> {

        let x1 = p.x();
        let y1 = p.y();

        let rm_x1 = (x1 * self.mapping_factor) as i32;
        let rm_y1 = (y1 * self.mapping_factor) as i32;

        let a1 = self._map_angle(rm_y1, rm_x1);

        // if not self.non_flat[rm_y1, rm_x1] > 1:
        //     return None

        let mut dir: f64 = 1.0;
        if !forwards {
            dir = -1.0
        }

        let x2 = x1 + self.line_step_distance * a1.cos() * dir;
        let y2 = y1 + self.line_step_distance * a1.sin() * dir;

        if x2 < 0.0 || x2 >= self._bbox[2] as f64 || y2 < 0.0 || y2 >= self._bbox[3] as f64 {
            return None;
        }

        if self._collision(tree, x2, y2) {
            return None;
        }

        // TODO: MAX_ANGLE_DISCONTINUITY check

        return Some(Point::new(x2, y2));
    }

    fn _extract_seed_points(&self, line: &VecDeque<Point>) -> Vec<Point> {
        let mut num_seedpoints = 1;
        let mut seed_points: Vec<Point> = Vec::new();

        if line.len() > SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS {
            num_seedpoints = (line.len() - 1) / SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS;
        }

        for i in 0..num_seedpoints {
            let (x1, y1) = line[i * SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS].x_y();
            let (x2, y2) = line[i * SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS + 1].x_y();

            // midpoint
            let x3 = x1 + (x2 - x1) / 2.0;
            let y3 = y1 + (y2 - y1) / 2.0;

            let a1 = (y1 - y3).atan2(x1 - x3);

            let mut a2 = a1;
            if i % 2 == 0 {
                a2 += PI/2.0;
            } else {
                a2 -= PI/2.0;
            }

            let x4 = self._map_line_distance(x3, y3);
            let y4 = 0.0;

            let x5 = x4 * a2.cos() - y4 * a2.sin() + x3;
            let y5 = x4 * a2.sin() + y4 * a2.cos() + y3;

            if x5 < 0.0 || x5 >= self._bbox[2] as f64 || y5 < 0.0 || y5 >= self._bbox[3] as f64 {
                continue
            }

            seed_points.push(Point::new(x5, y5));
        }
        return seed_points;
    }

    fn hatch(&self) -> Result<Vec<VecDeque<Point>>, Box<dyn Error>> {
        let mut tree: RTree<Point> = RTree::new();

        let mut lines: Vec<VecDeque<Point>> = Vec::new();
        let mut starting_points: VecDeque<Point> = VecDeque::new();

        // generate initial starting points
        for x in 0..self.map_line_distance.width() / STARTING_POINT_INIT_DISTANCE_WIDTH {
            for y in 0..self.map_line_distance.height() / STARTING_POINT_INIT_DISTANCE_HEIGHT {
                starting_points.push_back(Point::new(
                    (x * STARTING_POINT_INIT_DISTANCE_WIDTH) as f64,
                    (y * STARTING_POINT_INIT_DISTANCE_HEIGHT) as f64
                ));
            }
        }

        println!("starting_points: {:?}", starting_points.len());

        for i in 0..MAX_ITERATIONS {

            if i >= MAX_ITERATIONS - 1 {
                println!("maximum iterations exceeded");
            }

            if starting_points.len() == 0 {
                break; // hatching completed
            }

            // valid starting point?
            let starting_point = starting_points.pop_front().unwrap();
            if self._collision(&tree, starting_point.x(), starting_point.y()) {
                continue;
            }

            let mut line: VecDeque<Point> = VecDeque::new();
            line.push_front(starting_point);

            // follow gradient upwards
            for _ in 0..self.line_max_segments[1] {
                match self._next_point(&tree, line.back().unwrap(), true) {
                    Some(point) => {
                        if line.len() < self._map_line_max_segments(point.x(), point.y()) {
                            line.push_back(point);
                        }
                    },
                    None => break,
                }
            }

            // follow gradient downwards
            for _ in 0..self.line_max_segments[1] {
                match self._next_point(&tree, line.front().unwrap(), false) {
                    Some(point) => {
                        if line.len() < self._map_line_max_segments(point.x(), point.y()) {
                            line.push_front(point);
                        }
                    },
                    None => break,
                }
            }

            // LineString must contain 0 or >= 2 points
            if line.len() < 2 {
                continue;
            }

            // seed points
            for p in self._extract_seed_points(&line) {
                starting_points.push_front(p);
            }

            // collision detection
            for p in &line {
                tree.insert(p.clone());
            }

            lines.push(line);
        }
        Ok(lines)
    }
}

// impl Default for FlowlineHatcher {
//     fn default() -> FlowlineHatcher {
//         FlowlineHatcher {
//             line_distance: 3.0,
//             line_step_distance: 0.3,
//             line_max_segments: 25,
//         }
//     }
// }


fn load_grayscale_image(path: &str) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let img = ImageReader::open(path)
        .expect("Could not load image")
        .decode()
        .expect("Could not decode image");

    grayscale(&img)
}

fn main() {
    let map_distance = load_grayscale_image("test_data/map_distance.png");
    let map_angle = load_grayscale_image("test_data/map_angle.png");
    let map_max_segments = load_grayscale_image("test_data/map_max_segments.png");
    let map_non_flat = load_grayscale_image("test_data/map_non_flat.png");

    let timer_start = Utc::now();
    let hatcher = FlowlineHatcher::new(
        &map_distance,
        &map_angle,
        &map_max_segments,
        &map_non_flat,
    );
    let lines: Vec<VecDeque<Point>> = hatcher.hatch().unwrap();
    let timer_diff = Utc::now() - timer_start;
    println!("Total time taken to run is {:.3}s", timer_diff.num_milliseconds() as f64 / 1_000.0);

    let mut img_output: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(map_distance.width(), map_distance.height());
    for line in lines {

        // for point in line.points() {
        //     let pixel = img_output.get_pixel_mut(point.x() as u32, point.y() as u32);
        //     *pixel = image::Rgb([255, 255, 255]);
        // }

        // let point_vec = line.into_points();
        let point_vec = line;
        for i in 1..point_vec.len() {
            let start = point_vec[i-1];
            let end = point_vec[i];
            // println!("{:?} -> {:?}", start, end);

            draw_antialiased_line_segment_mut(
                &mut img_output,
                (start.x() as i32, start.y() as i32),
                (end.x() as i32, end.y() as i32),
                Rgb::from([255, 255, 255]),
                interpolate
            );
        }
    }

    img_output.save("output.jpg").expect("Failed to save output");
}
