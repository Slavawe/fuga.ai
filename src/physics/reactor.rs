use std::f64::consts::PI;

pub struct ControlRod {
    pub position: f64, // 0.0 = fully inserted, 1.0 = withdrawn
    pub worth: f64,    // total reactivity worth when fully withdrawn (pcm)
}

impl ControlRod {
    pub fn reactivity(&self) -> f64 {
        self.position * self.worth
    }
}

pub struct ReactorCore {
    pub n: f64,   // normalized neutron population (power level)
    pub c: f64,   // delayed neutron precursor concentration
    pub t: f64,   // fuel temperature (K above ambient)
    pub rho: f64, // net reactivity (pcm)

    // kinetics parameters
    pub beta: f64,          // delayed neutron fraction
    pub lambda: f64,        // neutron generation time (s)
    pub decay: f64,         // precursor decay constant (s^-1)
    pub power_coeff: f64,   // heating per unit power (K/s per n)
    pub cooling_coeff: f64, // cooling rate (1/s)
    pub alpha_t: f64,       // Doppler temperature coefficient (pcm/K)
    pub rho_ext: f64,       // external reactivity (pcm)

    pub rods: Vec<ControlRod>,
    pub time: f64,
}

impl Default for ReactorCore {
    fn default() -> Self {
        let mut rods = Vec::new();
        rods.push(ControlRod {
            position: 0.5,
            worth: 2000.0,
        });
        rods.push(ControlRod {
            position: 0.5,
            worth: 2000.0,
        });
        Self {
            n: 1e-6,
            c: 0.0,
            t: 0.0,
            rho: 0.0,
            beta: 0.0065,
            lambda: 1e-4,
            decay: 0.08,
            power_coeff: 50.0,
            cooling_coeff: 0.1,
            alpha_t: -2.0,
            rho_ext: 0.0,
            rods,
            time: 0.0,
        }
    }
}

impl ReactorCore {
    pub fn step(&mut self, dt: f64) {
        let rho_rods: f64 = self.rods.iter().map(|r| r.reactivity()).sum();
        self.rho = self.rho_ext + rho_rods + self.alpha_t * self.t;
        let beta = self.beta;
        let lam = self.lambda;
        let rho = self.rho * 1e-5;
        let dn = ((rho - beta) / lam * self.n + self.decay * self.c) * dt;
        let dc = (beta / lam * self.n - self.decay * self.c) * dt;
        let dtemp = (self.power_coeff * self.n - self.cooling_coeff * self.t) * dt;
        self.n = (self.n + dn).max(0.0);
        self.c = (self.c + dc).max(0.0);
        self.t = (self.t + dtemp).max(0.0);
        self.time += dt;
    }

    pub fn rod_position(&mut self, idx: usize, pos: f64) {
        if idx < self.rods.len() {
            self.rods[idx].position = pos.clamp(0.0, 1.0);
        }
    }

    pub fn rod_set(&mut self, positions: &[f64]) {
        for (i, &p) in positions.iter().enumerate() {
            self.rod_position(i, p);
        }
    }

    pub fn scram(&mut self) {
        for rod in &mut self.rods {
            rod.position = 0.0;
        }
    }
}

pub struct ReactorGrid {
    pub n: usize,
    pub flux: Vec<f64>,
}

impl ReactorGrid {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            flux: vec![0.0; n * n * n],
        }
    }

    pub fn set_flux_bessel(&mut self, core: &ReactorCore) {
        let n = self.n;
        let half = n as f64 / 2.0;
        let r0 = n as f64 * 0.35;
        for z in 0..n {
            let fz = ((z as f64 - half) / r0).abs();
            let z_env = if fz < 1.0 { (PI * fz / 2.0).cos() } else { 0.0 };
            for y in 0..n {
                let fy = ((y as f64 - half) / r0).abs();
                let y_env = if fy < 1.0 { (PI * fy / 2.0).cos() } else { 0.0 };
                for x in 0..n {
                    let fx = ((x as f64 - half) / r0).abs();
                    let x_env = if fx < 1.0 { (PI * fx / 2.0).cos() } else { 0.0 };
                    let f = x_env * y_env * z_env;
                    let i = z * n * n + y * n + x;
                    self.flux[i] = f * core.n;
                }
            }
        }
    }
}
