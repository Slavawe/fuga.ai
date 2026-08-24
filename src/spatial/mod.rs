pub mod controller;
pub mod room;
pub mod sensor;
pub mod world;

pub struct SpatialObservation {
    pub depth_map: Vec<f64>,
    pub agent_pos: [f64; 3],
    pub agent_vel: [f64; 3],
}

pub struct SpatialPerception {
    pub world: world::PhysicsWorld,
    pub sensors: Vec<sensor::RaySensor>,
    pub buf: Vec<f64>,
}

impl SpatialPerception {
    pub fn new(dim: usize) -> Self {
        let world = world::PhysicsWorld::new();
        let sensor_config = sensor::SensorConfig::default();
        let sensors = (0..sensor_config.ray_count)
            .map(|i| sensor::RaySensor::new(i, &sensor_config))
            .collect();
        Self {
            world,
            sensors,
            buf: vec![0.0; dim],
        }
    }

    pub fn step(&mut self, dt: f64) -> SpatialObservation {
        self.world.step(dt);
        let pos = self.world.camera_pos();
        let vel = self.world.camera_vel();

        let depth_map: Vec<f64> = self
            .sensors
            .iter()
            .map(|s| {
                let o = [pos[0] as f32, pos[1] as f32, pos[2] as f32];
                let (d, _) = s.cast_raw(&o, &self.world);
                d as f64
            })
            .collect();

        SpatialObservation {
            depth_map,
            agent_pos: pos,
            agent_vel: vel,
        }
    }

    pub fn encode_obs(&mut self, obs: &SpatialObservation) -> &[f64] {
        let dim = self.buf.len();
        self.buf.fill(0.0);

        let rays = obs.depth_map.len();
        for (i, &d) in obs.depth_map.iter().enumerate() {
            let idx = (i as f64 / rays as f64 * dim as f64) as usize % dim;
            let norm_d = (d / 20.0).min(1.0);
            self.buf[idx] = norm_d;
        }

        let pos_idx = (dim / 4) % dim;
        self.buf[pos_idx] = (obs.agent_pos[0] / 10.0 + 1.0) * 0.5;
        self.buf[(pos_idx + 1) % dim] = (obs.agent_pos[1] / 10.0 + 1.0) * 0.5;
        self.buf[(pos_idx + 2) % dim] = (obs.agent_pos[2] / 10.0 + 1.0) * 0.5;

        &self.buf
    }
}
