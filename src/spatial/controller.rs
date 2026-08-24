use crate::core::hypervector::Hypervector;
use crate::core::wave_cube::WaveCube;

pub struct RoomController {
    pub cube: WaveCube<3, 4>,
    pub attractor: [f64; 3],
    pub target_vel: f64,
    pub k_attract: f64,
    pub k_repel: f64,
    pub k_damp: f64,
    pub wall_threshold: f64,
    pub half_extent: f64,
    phase_history: Vec<f64>,
    coherence_history: Vec<f64>,
}

impl RoomController {
    pub fn new(dim: usize, half_extent: f64) -> Self {
        Self {
            cube: WaveCube::<3, 4>::new(dim),
            attractor: [0.0, 0.0, 0.0],
            target_vel: 0.5,
            k_attract: 0.3,
            k_repel: 5.0,
            k_damp: 1.2,
            wall_threshold: 3.0,
            half_extent,
            phase_history: Vec::with_capacity(200),
            coherence_history: Vec::with_capacity(200),
        }
    }

    pub fn compute(
        &mut self,
        pos: [f64; 3],
        vel: [f64; 3],
        distances: &[f32],
        step_id: usize,
    ) -> [f32; 3] {
        let dim = self.cube.dim;
        let room = self.half_extent;

        let mut encoded = vec![0.0; dim];
        for (i, &d) in distances.iter().enumerate() {
            let idx = (i as f64 / distances.len() as f64 * dim as f64) as usize % dim;
            encoded[idx] = (d as f64 / room).min(1.0);
        }

        let hv = Hypervector::from_i8_bits(
            dim,
            &encoded
                .iter()
                .map(|&x| if x > 0.5 { 1i8 } else { -1i8 })
                .collect::<Vec<_>>(),
        );
        let cell = (step_id % 4, (step_id / 4) % 4, (step_id / 16) % 4);
        self.cube.write_cell(cell.0, cell.1, cell.2, &hv);

        if step_id % 10 == 0 {
            self.cube.wave_flow_x(1);
            self.cube.wave_flow_y(1);
            self.cube.wave_flow_z(1);
        }

        let coherence = self.cube.coherence();
        self.phase_history.push(coherence);
        self.coherence_history.push(coherence);
        if self.phase_history.len() > 100 {
            self.phase_history.remove(0);
        }
        if self.coherence_history.len() > 500 {
            self.coherence_history.remove(0);
        }

        let dx = self.attractor[0] - pos[0];
        let dz = self.attractor[2] - pos[2];
        let dist_to_target = (dx * dx + dz * dz).sqrt();

        let wall_dists = [
            room - pos[0],
            room + pos[0],
            room - pos[1],
            room + pos[1],
            room - pos[2],
            room + pos[2],
        ];
        let min_wall = wall_dists.iter().copied().fold(f64::INFINITY, f64::min);

        let attract_scale = if min_wall < 1.5 {
            (min_wall / 1.5).max(0.1)
        } else if dist_to_target < 0.5 {
            dist_to_target / 0.5
        } else if dist_to_target > 3.0 {
            1.5
        } else {
            1.0
        };

        let desired_vx = dx * 0.4 * attract_scale;
        let desired_vz = dz * 0.4 * attract_scale;

        let max_speed = if min_wall < 1.0 { 0.5 } else { 1.5 };
        let dspeed = (desired_vx * desired_vx + desired_vz * desired_vz).sqrt();
        let (desired_vx, desired_vz) = if dspeed > max_speed {
            (
                desired_vx / dspeed * max_speed,
                desired_vz / dspeed * max_speed,
            )
        } else {
            (desired_vx, desired_vz)
        };

        let mut fx = (desired_vx - vel[0]) * 2.0;
        let mut fz = (desired_vz - vel[2]) * 2.0;
        let mut fy = -vel[1] * 3.0;

        let wall_normals: [[f64; 3]; 6] = [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ];

        for (wd, n) in wall_dists.iter().zip(wall_normals.iter()) {
            if *wd < self.wall_threshold {
                let strength = self.k_repel * (1.0 - *wd / self.wall_threshold);
                let hard = if *wd < 0.8 {
                    self.k_repel * 20.0 * (1.0 - *wd / 0.8).max(0.0)
                } else {
                    0.0
                };
                let s = (strength + hard).min(10.0);
                fx += n[0] * s;
                fy += n[1] * s;
                fz += n[2] * s;
            }
        }

        fx -= vel[0] * self.k_damp;
        fz -= vel[2] * self.k_damp;

        let max_f = 8.0;
        let mag = (fx * fx + fy * fy + fz * fz).sqrt();
        if mag > max_f {
            fx = fx / mag * max_f;
            fy = fy / mag * max_f;
            fz = fz / mag * max_f;
        }

        [fx as f32, fy as f32, fz as f32]
    }

    pub fn set_target(&mut self, x: f64, z: f64) {
        self.attractor = [x, 0.0, z];
    }

    pub fn phase_stability(&self) -> f64 {
        if self.phase_history.len() < 10 {
            return 1.0;
        }
        let recent: Vec<f64> = self.phase_history.iter().rev().take(10).copied().collect();
        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let var: f64 = recent.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        (1.0 - var.sqrt() * 30.0).max(0.0).min(1.0)
    }

    pub fn coherence(&self) -> f64 {
        if self.coherence_history.is_empty() {
            return 0.5;
        }
        self.coherence_history[self.coherence_history.len() - 1]
    }

    pub fn entropy(&self) -> f64 {
        self.cube.global_entropy()
    }
}
