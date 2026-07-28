use crate::spatial::world::PhysicsWorld;

pub struct SensorConfig {
    pub ray_count: usize,
    pub max_dist: f32,
    pub fov_h: f32,
    pub fov_v: f32,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            ray_count: 64,
            max_dist: 20.0,
            fov_h: std::f32::consts::PI * 0.75,
            fov_v: std::f32::consts::PI * 0.5,
        }
    }
}

impl Clone for SensorConfig {
    fn clone(&self) -> Self {
        Self {
            ray_count: self.ray_count,
            max_dist: self.max_dist,
            fov_h: self.fov_h,
            fov_v: self.fov_v,
        }
    }
}

pub struct RaySensor {
    pub index: usize,
    pub direction: [f32; 3],
    pub config: SensorConfig,
}

impl RaySensor {
    pub fn new(index: usize, config: &SensorConfig) -> Self {
        let total = config.ray_count as f32;
        let i = index as f32;
        let cols = total.sqrt().ceil();
        let row = (index as f32 / cols).floor();
        let h_angle = -config.fov_h / 2.0 + (i / total) * config.fov_h;
        let v_angle = -config.fov_v / 2.0 + (row / cols) * config.fov_v;

        let dir = [
            h_angle.sin() * v_angle.cos(),
            v_angle.sin(),
            h_angle.cos() * v_angle.cos(),
        ];

        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let dir = if len > 0.0 {
            [dir[0] / len, dir[1] / len, dir[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        };

        Self {
            index,
            direction: dir,
            config: config.clone(),
        }
    }

    pub fn cast(&self, world: &PhysicsWorld) -> (f64, Option<String>) {
        let origin = world.camera_pos();
        let origin_f32 = [origin[0] as f32, origin[1] as f32, origin[2] as f32];
        let (dist, hit) = world.ray_cast(&origin_f32, &self.direction, self.config.max_dist);
        (dist as f64, hit)
    }

    pub fn cast_raw(&self, origin: &[f32; 3], world: &PhysicsWorld) -> (f32, Option<String>) {
        world.ray_cast(origin, &self.direction, self.config.max_dist)
    }
}

pub struct SphericalSensor {
    pub directions: Vec<[f32; 3]>,
    pub max_dist: f32,
}

impl SphericalSensor {
    pub fn new(num_rays: usize, max_dist: f32) -> Self {
        let mut directions = Vec::with_capacity(num_rays);
        let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;

        for i in 0..num_rays {
            let theta = 2.0 * std::f32::consts::PI * i as f32 / golden_ratio;
            let phi = (1.0 - 2.0 * (i as f32 + 0.5) / num_rays as f32).acos();
            directions.push([
                theta.cos() * phi.sin(),
                phi.cos(),
                theta.sin() * phi.sin(),
            ]);
        }

        Self { directions, max_dist }
    }

    pub fn cast_all(&self, origin: &[f64; 3], room: &super::room::Room) -> Vec<f32> {
        let o = [origin[0] as f32, origin[1] as f32, origin[2] as f32];
        let eps = 0.35;
        self.directions.iter().map(|dir| {
            let ro = [o[0] + dir[0] * eps, o[1] + dir[1] * eps, o[2] + dir[2] * eps];
            room.cast_ray(&ro, dir, self.max_dist)
        }).collect()
    }
}
