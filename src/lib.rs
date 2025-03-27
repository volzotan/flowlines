use chrono::Utc;
use geo::Point;
use image::imageops::grayscale;
use image::{GrayImage, ImageBuffer, ImageReader, Luma, Rgb};
use imageproc::drawing::draw_antialiased_line_segment_mut;
use imageproc::pixelops::interpolate;
use rstar::RTree;
use std::collections::VecDeque;
use std::error::Error;
use std::f64::consts::{PI, TAU};

const MAX_ITERATIONS: u32 = 1_000_000;
const SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS: usize = 3;

pub struct FlowlinesConfig {
    line_distance: [f64; 2],
    line_distance_end_factor: f64,
    line_step_distance: f64,
    line_max_segments: [u32; 2],
    max_angle_discontinuity: f64,
    starting_point_init_distance: [i32; 2],
    seedpoint_extraction_skip_line_segments: usize,
    max_iterations: u32,
}

impl Default for FlowlinesConfig {
    fn default() -> FlowlinesConfig {
        FlowlinesConfig {
            line_distance: [3.0, 6.0],
            line_distance_end_factor: 0.75,
            line_step_distance: 0.5,
            line_max_segments: [100, 1000], //[60, 100],
            max_angle_discontinuity: PI / 2.0,
            starting_point_init_distance: [5, 5],
            seedpoint_extraction_skip_line_segments: SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS,
            max_iterations: MAX_ITERATIONS,
        }
    }
}

pub struct FlowlinesHatcher<'a> {
    config: FlowlinesConfig,
    mapping_factor: u32,
    map_line_distance: &'a GrayImage,
    map_angle: &'a GrayImage,
    map_line_max_segments: &'a GrayImage,
    map_non_flat: &'a GrayImage,
    _bbox: [i32; 4],
}

impl<'a> FlowlinesHatcher<'a> {
    pub fn new(
        config: FlowlinesConfig,
        map_line_distance: &'a GrayImage,
        map_angle: &'a GrayImage,
        map_line_max_segments: &'a GrayImage,
        map_non_flat: &'a GrayImage,
    ) -> Self {
        let bbox = [
            0,
            0,
            map_line_distance.width() as i32,
            map_line_distance.height() as i32,
        ];

        FlowlinesHatcher {
            config,
            mapping_factor: 1,
            map_line_distance,
            map_angle,
            map_line_max_segments,
            map_non_flat,
            _bbox: bbox,
        }
    }

    fn _map_angle(&self, x: u32, y: u32) -> f64 {
        let angle = self.map_angle.get_pixel(x, y)[0] as f64 / 255.0 * PI * 2.0;
        angle - PI // supplied u8 image is centered around 128 to deal with negative values
    }

    fn _map_line_distance(&self, x: f64, y: f64) -> f64 {
        let pixel = self.map_line_distance.get_pixel(x as u32, y as u32)[0] as f64;
        let diff = self.config.line_distance[1] - self.config.line_distance[0];
        self.config.line_distance[0] + diff * pixel / 255.0
    }

    fn _map_line_max_segments(&self, x: f64, y: f64) -> usize {
        let pixel = self.map_line_max_segments.get_pixel(x as u32, y as u32)[0] as f64;
        let diff = (self.config.line_max_segments[1] - self.config.line_max_segments[0]) as f64;
        (self.config.line_max_segments[0] as f64 + diff * pixel / 255.0) as usize
    }

    fn _collision(&self, tree: &RTree<Point>, x: f64, y: f64, factor: f64) -> bool {
        tree.locate_within_distance(
            Point::new(x, y),
            (self._map_line_distance(x, y) * factor).powi(2),
        )
        .count()
            > 0
    }

    fn _next_point(&self, tree: &RTree<Point>, p: &Point, forwards: bool) -> Option<Point> {
        let x1 = p.x();
        let y1 = p.y();

        let rm_x1 = (x1 * self.mapping_factor as f64) as u32;
        let rm_y1 = (y1 * self.mapping_factor as f64) as u32;
        let a1 = self._map_angle(rm_x1, rm_y1);

        if self.map_non_flat.get_pixel(rm_x1, rm_y1)[0] == 0 {
            return None;
        }

        let mut dir: f64 = 1.0;
        if !forwards {
            dir = -1.0
        }

        let x2 = x1 + self.config.line_step_distance * a1.cos() * dir;
        let y2 = y1 + self.config.line_step_distance * a1.sin() * dir;

        if x2 < 0.0 || x2 >= self._bbox[2] as f64 || y2 < 0.0 || y2 >= self._bbox[3] as f64 {
            return None;
        }

        if self._collision(tree, x2, y2, self.config.line_distance_end_factor) {
            return None;
        }

        if self.config.max_angle_discontinuity > 0.0 {
            let rm_x2 = (x2 * self.mapping_factor as f64) as u32;
            let rm_y2 = (y2 * self.mapping_factor as f64) as u32;
            let a2 = self._map_angle(rm_x2, rm_y2);

            if (a2 - a1).abs() > self.config.max_angle_discontinuity {
                return None;
            }
        }

        Some(Point::new(x2, y2))
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
                a2 += PI / 2.0;
            } else {
                a2 -= PI / 2.0;
            }

            let x4 = self._map_line_distance(x3, y3);
            let y4 = 0.0;

            let x5 = x4 * a2.cos() - y4 * a2.sin() + x3;
            let y5 = x4 * a2.sin() + y4 * a2.cos() + y3;

            if x5 < 0.0 || x5 >= self._bbox[2] as f64 || y5 < 0.0 || y5 >= self._bbox[3] as f64 {
                continue;
            }

            seed_points.push(Point::new(x5, y5));
        }
        return seed_points;
    }

    fn _generate_starting_points(&self) -> VecDeque<Point> {
        let mut starting_points: VecDeque<Point> = VecDeque::new();

        for x in 0..(self._bbox[2] - self._bbox[0]) / self.config.starting_point_init_distance[0] {
            for y in
                0..(self._bbox[3] - self._bbox[1]) / self.config.starting_point_init_distance[1]
            {
                starting_points.push_back(Point::new(
                    (self._bbox[0] + x * self.config.starting_point_init_distance[0]) as f64,
                    (self._bbox[1] + y * self.config.starting_point_init_distance[1]) as f64,
                ));
            }
        }

        return starting_points;
    }

    pub fn hatch(&self) -> Result<Vec<VecDeque<Point>>, Box<dyn Error>> {
        let mut tree: RTree<Point> = RTree::new();
        let mut lines: Vec<VecDeque<Point>> = Vec::new();
        let mut starting_points: VecDeque<Point> = self._generate_starting_points();
        println!("starting_points: {:?}", starting_points.len());

        for i in 0..self.config.max_iterations {
            if i >= self.config.max_iterations - 1 {
                println!("maximum iterations exceeded");
            }

            if starting_points.len() == 0 {
                break; // hatching completed
            }

            // valid starting point?
            let starting_point = starting_points.pop_front().unwrap();
            if self._collision(&tree, starting_point.x(), starting_point.y(), 1.0) {
                continue;
            }

            let mut line: VecDeque<Point> = VecDeque::new();
            line.push_front(starting_point);

            // follow gradient upwards
            for _ in 0..self.config.line_max_segments[1] {
                match self._next_point(&tree, line.back().unwrap(), true) {
                    Some(point) => {
                        if line.len() > self._map_line_max_segments(point.x(), point.y()) {
                            break;
                        }

                        line.push_back(point);
                    }
                    None => break,
                }
            }

            // follow gradient downwards
            for _ in 0..self.config.line_max_segments[1] {
                match self._next_point(&tree, line.front().unwrap(), false) {
                    Some(point) => {
                        if line.len() > self._map_line_max_segments(point.x(), point.y()) {
                            break;
                        }

                        line.push_front(point);
                    }
                    None => break,
                }
            }

            // line should be >= 2 points
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_angle() {
        let map_distance = GrayImage::new(100, 100);
        let map_angle = GrayImage::from_pixel(100, 100, Luma([127]));
        let map_max_segments = GrayImage::new(100, 100);
        let map_non_flat = GrayImage::new(100, 100);
        let hatcher = FlowlinesHatcher::new(
            FlowlinesConfig::default(),
            &map_distance,
            &map_angle,
            &map_max_segments,
            &map_non_flat,
        );

        assert_eq!(
            hatcher._map_angle(50, 50),
            ((127.0 / 255.0) * TAU) - PI,
            "_map_angle() expects a u8 image mapping values from [0, 255] -> [-PI, +PI]"
        )
    }

    #[test]
    fn test_generate_starting_points() {
        let map_distance = GrayImage::new(100, 100);
        let map_angle = GrayImage::from_pixel(100, 100, Luma([127]));
        let map_max_segments = GrayImage::new(100, 100);
        let map_non_flat = GrayImage::new(100, 100);

        let mut config = FlowlinesConfig::default();
        config.starting_point_init_distance = [20, 20];

        let hatcher = FlowlinesHatcher::new(
            config,
            &map_distance,
            &map_angle,
            &map_max_segments,
            &map_non_flat,
        );

        let starting_points = hatcher._generate_starting_points();

        assert_eq!(
            starting_points.len(),
            ((100 / 20) as u32).pow(2) as usize,
            "incorrect number of starting points"
        )
    }
}
