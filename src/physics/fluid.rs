pub struct FluidField {
    pub density: Vec<f32>,
    pub velocity_x: Vec<f32>,
    pub velocity_z: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub diffusion: f32,
    pub viscosity: f32,
}

impl FluidField {
    pub fn new(w: usize, h: usize) -> Self {
        let n = w * h;
        Self {
            density: vec![0.0; n],
            velocity_x: vec![0.0; n],
            velocity_z: vec![0.0; n],
            width: w,
            height: h,
            diffusion: 0.001,
            viscosity: 0.0001,
        }
    }

    pub fn add_source(&mut self, x: f32, z: f32, amount: f32) {
        let i = ((x / 6.0 + 0.5) * self.width as f32) as usize;
        let j = ((z / 6.0 + 0.5) * self.height as f32) as usize;
        if i < self.width && j < self.height {
            let idx = j * self.width + i;
            self.density[idx] += amount;
        }
    }

    pub fn step(&mut self, dt: f32) {
        let w = self.width;
        let h = self.height;
        let mut new_d = self.density.clone();
        let mut new_vx = self.velocity_x.clone();
        let mut new_vz = self.velocity_z.clone();

        for i in 1..w-1 {
            for j in 1..h-1 {
                let idx = j * w + i;

                let dxx = self.density[idx+1] + self.density[idx-1] - 2.0 * self.density[idx];
                let dzz = self.density[idx+w] + self.density[idx-w] - 2.0 * self.density[idx];
                new_d[idx] += dt * (self.diffusion * (dxx + dzz)
                    - self.velocity_x[idx] * (self.density[idx+1] - self.density[idx-1]) / 2.0
                    - self.velocity_z[idx] * (self.density[idx+w] - self.density[idx-w]) / 2.0);

                let vxx = self.velocity_x[idx+1] + self.velocity_x[idx-1] - 2.0 * self.velocity_x[idx];
                let vxz = self.velocity_x[idx+w] + self.velocity_x[idx-w] - 2.0 * self.velocity_x[idx];
                new_vx[idx] += dt * (self.viscosity * (vxx + vxz));

                let vzx = self.velocity_z[idx+1] + self.velocity_z[idx-1] - 2.0 * self.velocity_z[idx];
                let vzz = self.velocity_z[idx+w] + self.velocity_z[idx-w] - 2.0 * self.velocity_z[idx];
                new_vz[idx] += dt * (self.viscosity * (vzx + vzz));
            }
        }

        self.density = new_d;
        self.velocity_x = new_vx;
        self.velocity_z = new_vz;
    }

    pub fn density_at(&self, x: f32, z: f32) -> f32 {
        let i = ((x / 6.0 + 0.5) * self.width as f32) as isize;
        let j = ((z / 6.0 + 0.5) * self.height as f32) as isize;
        if i >= 0 && i < self.width as isize && j >= 0 && j < self.height as isize {
            self.density[(j * self.width as isize + i) as usize]
        } else {
            0.0
        }
    }
}

pub fn color_from_density(d: f32) -> u32 {
    let f = (d * 20.0).min(1.0);
    let c = (f * 200.0) as u32;
    (c << 16) | (c << 8) | 0xFF
}
