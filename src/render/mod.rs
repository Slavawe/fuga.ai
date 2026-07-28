use minifb::{Key, Window, WindowOptions};

const W: usize = 960;
const H: usize = 640;

pub struct Camera {
    pub pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self { pos: [0.0, 2.0, 6.0], yaw: 0.0, pitch: -0.2 }
    }
}

pub struct Render3D {
    pub window: Window,
    pub framebuf: Vec<u32>,
    pub camera: Camera,
    zbuf: Vec<f32>,
}

impl Render3D {
    pub fn new(title: &str) -> Self {
        let window = Window::new(title, W, H, WindowOptions::default())
            .expect("Failed to create window");
        let framebuf = vec![0; W * H];
        let zbuf = vec![f32::INFINITY; W * H];
        Self { window, framebuf, camera: Camera::default(), zbuf }
    }

    pub fn clear(&mut self, color: u32) {
        self.framebuf.fill(color);
        self.zbuf.fill(f32::INFINITY);
    }

    pub fn project(&self, p: &[f32; 3]) -> Option<(f32, f32, f32)> {
        let dx = p[0] - self.camera.pos[0];
        let dy = p[1] - self.camera.pos[1];
        let dz = p[2] - self.camera.pos[2];

        let cos_yaw = self.camera.yaw.cos();
        let sin_yaw = self.camera.yaw.sin();
        let cos_pitch = self.camera.pitch.cos();
        let sin_pitch = self.camera.pitch.sin();

        let x = cos_yaw * dx + sin_yaw * dz;
        let z = -sin_yaw * dx + cos_yaw * dz;
        let y = sin_pitch * x + cos_pitch * dy;
        let x = cos_pitch * x - sin_pitch * dy;

        if z <= 0.1 { return None; }
        let fov = 800.0;
        let sx = (W as f32 / 2.0) + x * fov / z;
        let sy = (H as f32 / 2.0) - y * fov / z;

        if sx < 0.0 || sx >= W as f32 || sy < 0.0 || sy >= H as f32 { return None; }
        Some((sx, sy, z))
    }

    pub fn draw_line(&mut self, a: &[f32; 3], b: &[f32; 3], color: u32) {
        let Some(pa) = self.project(a) else { return };
        let Some(pb) = self.project(b) else { return };
        let (x1, y1, _) = pa;
        let (x2, y2, _) = pb;
        let dx = (x2 - x1).abs();
        let dy = -(y2 - y1).abs();
        let sx = if x1 < x2 { 1.0 } else { -1.0 };
        let sy = if y1 < y2 { 1.0 } else { -1.0 };
        let mut err = dx + dy;
        let (mut x, mut y) = (x1, y1);
        loop {
            let ix = x as usize;
            let iy = y as usize;
            if ix < W && iy < H {
                let i = iy * W + ix;
                self.framebuf[i] = color;
            }
            if (x - x2).abs() < 0.5 && (y - y2).abs() < 0.5 { break; }
            let e2 = 2.0 * err;
            if e2 >= dy { err += dy; x += sx; }
            if e2 <= dx { err += dx; y += sy; }
        }
    }

    pub fn draw_ray(&mut self, origin: &[f32; 3], dir: &[f32; 3], length: f32, color: u32) {
        let end = [
            origin[0] + dir[0] * length,
            origin[1] + dir[1] * length,
            origin[2] + dir[2] * length,
        ];
        self.draw_line(origin, &end, color);
    }

    pub fn draw_dot(&mut self, p: &[f32; 3], size: f32, color: u32) {
        let Some(proj) = self.project(p) else { return };
        let (sx, sy, _) = proj;
        let r = (size * 2.0).max(1.0) as usize;
        let cx = sx as usize;
        let cy = sy as usize;
        for dy in -(r as i32)..=r as i32 {
            for dx in -(r as i32)..=r as i32 {
                if dx * dx + dy * dy <= (r * r) as i32 {
                    let x = cx as i32 + dx;
                    let y = cy as i32 + dy;
                    if x >= 0 && x < W as i32 && y >= 0 && y < H as i32 {
                        self.framebuf[(y as usize) * W + (x as usize)] = color;
                    }
                }
            }
        }
    }

    pub fn draw_arrow(&mut self, from: &[f32; 3], to: &[f32; 3], color: u32) {
        self.draw_line(from, to, color);
        let dx = to[0] - from[0];
        let dy = to[1] - from[1];
        let dz = to[2] - from[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt().max(0.01);
        let ux = dx / len;
        let uy = dy / len;
        let uz = dz / len;
        let hx = 0.2;
        let hy = 0.2;
        let tip = |s: f32| {
            let bx = -ux * hx + s * hy * (-uy);
            let by = -uy * hx + s * hy * ux;
            let bz = -uz * hx;
            [to[0] + bx, to[1] + by, to[2] + bz]
        };
        self.draw_line(to, &tip(1.0), color);
        self.draw_line(to, &tip(-1.0), color);
    }

    pub fn draw_cube_wire(&mut self, cx: f32, cy: f32, cz: f32, s: f32, color: u32) {
        let h = s / 2.0;
        let corners = [
            [cx-h, cy-h, cz-h], [cx+h, cy-h, cz-h], [cx+h, cy+h, cz-h], [cx-h, cy+h, cz-h],
            [cx-h, cy-h, cz+h], [cx+h, cy-h, cz+h], [cx+h, cy+h, cz+h], [cx-h, cy+h, cz+h],
        ];
        let edges = [
            (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7),
        ];
        for &(i, j) in &edges {
            self.draw_line(&corners[i], &corners[j], color);
        }
    }

    pub fn draw_sphere_wire(&mut self, cx: f32, cy: f32, cz: f32, r: f32, color: u32) {
        let segs = 12;
        for i in 0..segs {
            let a1 = i as f32 * 2.0 * std::f32::consts::PI / segs as f32;
            let a2 = (i + 1) as f32 * 2.0 * std::f32::consts::PI / segs as f32;
            self.draw_line(
                &[cx + r * a1.cos(), cy + r * a1.sin(), cz],
                &[cx + r * a2.cos(), cy + r * a2.sin(), cz], color,
            );
            self.draw_line(
                &[cx + r * a1.cos(), cy, cz + r * a1.sin()],
                &[cx + r * a2.cos(), cy, cz + r * a2.sin()], color,
            );
        }
    }

    pub fn draw_ground_grid(&mut self, color: u32) {
        for i in -5..=5 {
            let i = i as f32;
            self.draw_line(&[i, 0.0, -5.0], &[i, 0.0, 5.0], color);
            self.draw_line(&[-5.0, 0.0, i], &[5.0, 0.0, i], color);
        }
    }

    pub fn is_open(&self) -> bool { self.window.is_open() && !self.window.is_key_down(Key::Escape) }
    pub fn update(&mut self) { self.window.update_with_buffer(&self.framebuf, W, H).ok(); }
}
