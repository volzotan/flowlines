use geo::Point;
use image::GrayImage;
use rstar::RTree;
use std::collections::VecDeque;
use std::error::Error;
use std::f64::consts::{PI, TAU};

const MAX_ITERATIONS: u32 = 100_000_000;
const SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS: usize = 3;

pub struct FlowlinesConfig {
    pub line_distance: [f64; 2],
    pub line_distance_end_factor: f64,
    pub line_step_distance: f64,
    pub line_max_length: [f64; 2],
    pub max_angle_discontinuity: f64,
    pub starting_point_init_distance: [f64; 2],
    pub seedpoint_extraction_skip_line_segments: usize,
    pub max_iterations: u32,
}

impl Default for FlowlinesConfig {
    fn default() -> FlowlinesConfig {
        FlowlinesConfig {
            line_distance: [3.0, 6.0],
            line_distance_end_factor: 0.75,
            line_step_distance: 0.5,
            line_max_length: [100.0, 1000.0], //[60, 100],
            max_angle_discontinuity: PI / 2.0,
            starting_point_init_distance: [5.0, 5.0],
            seedpoint_extraction_skip_line_segments: SEEDPOINT_EXTRACTION_SKIP_LINE_SEGMENTS,
            max_iterations: MAX_ITERATIONS,
        }
    }
}

pub struct FlowlinesHatcher<'a> {
    config: &'a FlowlinesConfig,
    map_line_distance: &'a GrayImage,
    map_angle: &'a GrayImage,
    map_line_max_length: &'a GrayImage,
    map_non_flat: &'a GrayImage,
    mapping_factor: u32,
    bbox: [i32; 4],
}

impl<'a> FlowlinesHatcher<'a> {
    pub fn new(
        config: &'a FlowlinesConfig,
        map_line_distance: &'a GrayImage,
        map_angle: &'a GrayImage,
        map_line_max_length: &'a GrayImage,
        map_non_flat: &'a GrayImage,
    ) -> Self {

        let bbox = [
            0,
            0,
            map_line_distance.width() as i32,
            map_line_distance.height() as i32,
        ];

        assert!(config.line_distance[0] > 0.0);
        assert!(config.line_distance[1] > 0.0);
        assert!(config.line_distance[0] <= config.line_distance[1]);
        assert!(config.line_distance_end_factor > 0.0);
        assert!(config.line_step_distance > 0.0);
        assert!(config.line_max_length[0] > 0.0);
        assert!(config.line_max_length[1] > 0.0);
        assert!(config.line_max_length[0] <= config.line_max_length[1]);
        assert!(config.starting_point_init_distance[0] > 0.0);
        assert!(config.starting_point_init_distance[1] > 0.0);

        FlowlinesHatcher {
            config,
            map_line_distance,
            map_angle,
            map_line_max_length,
            map_non_flat,
            mapping_factor: 1,
            bbox,
        }
    }

    fn map_angle(&self, x: u32, y: u32) -> f64 {
        let angle = self.map_angle.get_pixel(x, y)[0] as f64 / 255.0 * TAU;
        angle - PI // supplied u8 image is centered around 128 to deal with negative values
    }

    fn map_line_distance(&self, x: f64, y: f64) -> f64 {
        let pixel = self.map_line_distance.get_pixel(x as u32, y as u32)[0] as f64;
        let diff = self.config.line_distance[1] - self.config.line_distance[0];
        self.config.line_distance[0] + diff * pixel / 255.0
    }

    fn map_line_max_length(&self, x: f64, y: f64) -> f64 {
        let pixel = self.map_line_max_length.get_pixel(x as u32, y as u32)[0] as f64;
        let diff = (self.config.line_max_length[1] - self.config.line_max_length[0]) as f64;
        self.config.line_max_length[0] as f64 + diff * pixel / 255.0
    }

    fn collision(&self, tree: &RTree<Point>, x: f64, y: f64, factor: f64) -> bool {
        tree.locate_within_distance(
            Point::new(x, y),
            (self.map_line_distance(x, y) * factor).powi(2),
        )
        .count()
            > 0
    }

    fn next_point(&self, tree: &RTree<Point>, p: &Point, forwards: bool) -> Option<Point> {
        let x1 = p.x();
        let y1 = p.y();

        let rm_x1 = (x1 * self.mapping_factor as f64) as u32;
        let rm_y1 = (y1 * self.mapping_factor as f64) as u32;
        let a1 = self.map_angle(rm_x1, rm_y1);

        if self.map_non_flat.get_pixel(rm_x1, rm_y1)[0] == 0 {
            return None;
        }

        let mut dir: f64 = 1.0;
        if !forwards {
            dir = -1.0
        }

        let x2 = x1 + self.config.line_step_distance * a1.cos() * dir;
        let y2 = y1 + self.config.line_step_distance * a1.sin() * dir;

        if x2 < 0.0 || x2 >= self.bbox[2] as f64 || y2 < 0.0 || y2 >= self.bbox[3] as f64 {
            return None;
        }

        if self.collision(tree, x2, y2, self.config.line_distance_end_factor) {
            return None;
        }

        if self.config.max_angle_discontinuity > 0.0 {
            let rm_x2 = (x2 * self.mapping_factor as f64) as u32;
            let rm_y2 = (y2 * self.mapping_factor as f64) as u32;
            let a2 = self.map_angle(rm_x2, rm_y2);

            if (a2 - a1).abs() > self.config.max_angle_discontinuity {
                return None;
            }
        }

        Some(Point::new(x2, y2))
    }

    fn extract_seed_points(&self, line: &VecDeque<Point>) -> Vec<Point> {
        let mut num_seedpoints = 1;
        let mut seed_points: Vec<Point> = Vec::new();

        if line.len() > self.config.seedpoint_extraction_skip_line_segments {
            num_seedpoints = (line.len() - 1) / self.config.seedpoint_extraction_skip_line_segments;
        }

        for i in 0..num_seedpoints {
            let (x1, y1) = line[i * self.config.seedpoint_extraction_skip_line_segments].x_y();
            let (x2, y2) = line[i * self.config.seedpoint_extraction_skip_line_segments + 1].x_y();

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

            let x4 = self.map_line_distance(x3, y3);
            let y4 = 0.0;

            let x5 = x4 * a2.cos() - y4 * a2.sin() + x3;
            let y5 = x4 * a2.sin() + y4 * a2.cos() + y3;

            if x5 < 0.0 || x5 >= self.bbox[2] as f64 || y5 < 0.0 || y5 >= self.bbox[3] as f64 {
                continue;
            }

            seed_points.push(Point::new(x5, y5));
        }
        return seed_points;
    }

    fn generate_starting_points(&self) -> VecDeque<Point> {
        let mut starting_points: VecDeque<Point> = VecDeque::new();

        let width = self.bbox[2] - self.bbox[0];
        let height = self.bbox[3] - self.bbox[1];

        for x in 0..(width as f64 / self.config.starting_point_init_distance[0]) as i32 {
            for y in 0..(height as f64 / self.config.starting_point_init_distance[1]) as i32 {
                starting_points.push_back(Point::new(
                    (self.bbox[0] as f64) + (x as f64) * self.config.starting_point_init_distance[0],
                    (self.bbox[1] as f64) + (y as f64) * self.config.starting_point_init_distance[1],
                ));
            }
        }

        return starting_points;
    }

    pub fn hatch(&self) -> Result<Vec<VecDeque<Point>>, Box<dyn Error>> {
        let mut tree: RTree<Point> = RTree::new();
        let mut lines: Vec<VecDeque<Point>> = Vec::new();
        let mut starting_points: VecDeque<Point> = self.generate_starting_points();

        // println!("starting_points: {:?}", starting_points.len());

        for i in 0..self.config.max_iterations {
            if i >= self.config.max_iterations - 1 {
                println!("maximum iterations exceeded");
            }

            // hatching completed?
            if starting_points.len() == 0 {
                break;
            }

            // valid starting point?
            let starting_point = starting_points.pop_front().unwrap();
            if self.collision(&tree, starting_point.x(), starting_point.y(), 1.0) {
                continue;
            }

            let mut line: VecDeque<Point> = VecDeque::new();
            line.push_front(starting_point);

            // follow gradient upwards
            for _ in 0..(self.config.line_max_length[1] / self.config.line_step_distance) as u32 {
                match self.next_point(&tree, line.back().unwrap(), true) {
                    Some(point) => {
                        if (line.len() as f64 * self.config.line_step_distance)
                            > self.map_line_max_length(point.x(), point.y())
                        {
                            break;
                        }

                        line.push_back(point);
                    }
                    None => break,
                }
            }

            // follow gradient downwards
            for _ in 0..(self.config.line_max_length[1] / self.config.line_step_distance) as u32 {
                match self.next_point(&tree, line.front().unwrap(), false) {
                    Some(point) => {
                        if (line.len() as f64 * self.config.line_step_distance)
                            > self.map_line_max_length(point.x(), point.y())
                        {
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

            // get net seed points
            for p in self.extract_seed_points(&line) {
                starting_points.push_front(p);
            }

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
        let map_max_length = GrayImage::new(100, 100);
        let map_non_flat = GrayImage::new(100, 100);
        let config = FlowlinesConfig::default();

        let hatcher = FlowlinesHatcher::new(
            &config,
            &map_distance,
            &map_angle,
            &map_max_length,
            &map_non_flat,
        );

        assert_eq!(
            hatcher.map_angle(50, 50),
            ((127.0 / 255.0) * TAU) - PI,
            "_map_angle() expects a u8 image mapping values from [0, 255] -> [-PI, +PI]"
        );
    }

    #[test]
    fn test_generate_starting_points() {
        let (width, height)  = (200, 100);
        let map_distance = GrayImage::new(width, height);
        let map_angle = GrayImage::from_pixel(width, height, Luma([127]));
        let map_max_length = GrayImage::new(width, height);
        let map_non_flat = GrayImage::new(width, height);
        let mut config = FlowlinesConfig::default();

        config.starting_point_init_distance = [20.0, 20.0];
        let hatcher = FlowlinesHatcher::new(
            &config,
            &map_distance,
            &map_angle,
            &map_max_length,
            &map_non_flat,
        );
        let starting_points = hatcher.generate_starting_points();

        assert_eq!(
            ((width as f64 / config.starting_point_init_distance[0]) * (height as f64 / config.starting_point_init_distance[1])) as usize,
            starting_points.len(),
            "incorrect number of starting points"
        );

        config.starting_point_init_distance = [0.5, 0.5];
        let hatcher = FlowlinesHatcher::new(
            &config,
            &map_distance,
            &map_angle,
            &map_max_length,
            &map_non_flat,
        );
        let starting_points = hatcher.generate_starting_points();

        assert_eq!(
            (width * height) as usize,
            starting_points.len(),
            "incorrect number of starting points when distance is < 1.0 (smaller than a pixel)"
        );
    }
}
