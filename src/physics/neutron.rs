pub struct NeutronDiffusion {
    pub grid: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub diffusivity: f64,
    pub absorption: f64,
    pub source: Vec<f64>,
}

impl NeutronDiffusion {
    pub fn new(w: usize, h: usize, d: usize) -> Self {
        let n = w * h * d;
        let mut source = vec![0.0; n];
        let center = |i: usize, j: usize, k: usize| -> bool {
            i >= w / 3
                && i < 2 * w / 3
                && j >= h / 3
                && j < 2 * h / 3
                && k >= d / 3
                && k < 2 * d / 3
        };
        for i in 0..w {
            for j in 0..h {
                for k in 0..d {
                    let idx = (i * h + j) * d + k;
                    if center(i, j, k) {
                        source[idx] = 1.0;
                    }
                }
            }
        }
        Self {
            grid: vec![1e-6; n],
            width: w,
            height: h,
            depth: d,
            diffusivity: 1.0 / 3.0,
            absorption: 0.05,
            source,
        }
    }

    pub fn step(&mut self, dt: f64) {
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let mut next = self.grid.clone();
        let dx = 1.0 / w as f64;

        for i in 0..w {
            for j in 0..h {
                for k in 0..d {
                    let idx = (i * h + j) * d + k;
                    let mut lap = 0.0;

                    if i > 0 {
                        lap += self.grid[((i - 1) * h + j) * d + k];
                    } else {
                        lap += self.grid[idx];
                    }
                    if i < w - 1 {
                        lap += self.grid[((i + 1) * h + j) * d + k];
                    } else {
                        lap += self.grid[idx];
                    }
                    if j > 0 {
                        lap += self.grid[(i * h + (j - 1)) * d + k];
                    } else {
                        lap += self.grid[idx];
                    }
                    if j < h - 1 {
                        lap += self.grid[(i * h + (j + 1)) * d + k];
                    } else {
                        lap += self.grid[idx];
                    }
                    if k > 0 {
                        lap += self.grid[(i * h + j) * d + (k - 1)];
                    } else {
                        lap += self.grid[idx];
                    }
                    if k < d - 1 {
                        lap += self.grid[(i * h + j) * d + (k + 1)];
                    } else {
                        lap += self.grid[idx];
                    }

                    lap = (lap - 6.0 * self.grid[idx]) / (dx * dx);
                    let dphi_dt = self.diffusivity * lap - self.absorption * self.grid[idx]
                        + self.source[idx];
                    next[idx] = (self.grid[idx] + dt * dphi_dt).max(0.0);
                }
            }
        }
        self.grid = next;
    }

    pub fn flux_at(&self, x: f32, y: f32, z: f32) -> f32 {
        let i = ((x + 1.5) / 3.0 * self.width as f32) as usize;
        let j = ((y.max(0.0).min(3.0)) / 3.0 * self.height as f32) as usize;
        let k = ((z + 1.5) / 3.0 * self.depth as f32) as usize;
        if i < self.width && j < self.height && k < self.depth {
            self.grid[(i * self.height + j) * self.depth + k] as f32
        } else {
            0.0
        }
    }
}

pub fn color_from_flux(flux: f32) -> u32 {
    let f = (flux * 255.0).min(255.0) as u32;
    let r = f.min(100) * 2;
    let g = (f / 2).min(200);
    let b = (255 - f / 2).max(0);
    (r.min(255) << 16) | (g.min(255) << 8) | b.min(255)
}
