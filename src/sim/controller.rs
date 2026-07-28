use crate::core::wave_cube::WaveCube;
use crate::core::hypervector::Hypervector;

pub struct CubicController<const S: usize> {
    pub cube: WaveCube<3, S>,
    pub setpoint: f64,
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    prev_error: f64,
    integral: f64,
    phase_history: Vec<f64>,
}

impl<const S: usize> CubicController<S> {
    pub fn new(dim: usize) -> Self {
        Self {
            cube: WaveCube::<3, S>::new(dim),
            setpoint: 0.0,
            kp: 0.5,
            ki: 0.1,
            kd: 0.05,
            prev_error: 0.0,
            integral: 0.0,
            phase_history: Vec::with_capacity(64),
        }
    }

    fn encode_float(&self, val: f64) -> Hypervector {
        let bits = val.to_bits();
        let mut hv = Hypervector::new(self.cube.dim);
        for i in 0..hv.words.len().min(8) {
            hv.words[i] = bits.wrapping_mul((i as u64 + 1) * 0x9E3779B97F4A7C15);
        }
        let p = bits as f64 / u64::MAX as f64;
        let threshold = (p * self.cube.dim as f64) as usize;
        for i in 0..threshold.min(self.cube.dim) {
            let w = i / 64;
            let b = i % 64;
            if w < hv.words.len() {
                hv.words[w] ^= 1u64 << b;
            }
        }
        hv
    }

    fn decode_phase(&self, sensor_hv: &Hypervector, setpoint_hv: &Hypervector) -> f64 {
        1.0 - sensor_hv.similarity(setpoint_hv)
    }

    pub fn update(&mut self, measurement: f64, dt: f64) -> f64 {
        let error = self.setpoint - measurement;
        let derivative = (error - self.prev_error) / dt.max(1e-9);
        self.prev_error = error;

        let raw = self.kp * error + self.ki * self.integral + self.kd * derivative;
        let output = raw.clamp(-0.5, 0.5);

        if (raw - output).abs() < 0.01 {
            self.integral += error * dt;
        }

        let sensor_hv = self.encode_float(measurement);
        let set_hv = self.encode_float(self.setpoint);
        let x = ((error.abs() * 10.0) as usize).min(S - 1);
        let y = ((self.integral.abs() * 5.0) as usize).min(S - 1);
        let z = ((derivative.abs() * 2.0) as usize).min(S - 1);
        self.cube.write_cell(x, y, z, &sensor_hv);

        let resonance = self.cube.cell(x, y, z);
        let phase_error = self.decode_phase(&resonance, &set_hv);
        self.phase_history.push(phase_error);
        if self.phase_history.len() > 64 {
            self.phase_history.remove(0);
        }

        if self.phase_history.len() % 20 == 0 {
            self.cube.wave_flow_x(1);
            self.cube.wave_flow_y(1);
            self.cube.wave_flow_z(1);
        }

        output
    }

    pub fn phase_stability(&self) -> f64 {
        if self.phase_history.len() < 4 {
            return 1.0;
        }
        let recent: Vec<f64> = self.phase_history.iter().rev().take(10).copied().collect();
        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance: f64 = recent.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        (1.0 - variance.sqrt()).max(0.0)
    }
}
