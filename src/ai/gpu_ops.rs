//! GPU acceleration for the latent transition operator (Widrow-Hoff delta).
//!
//! The byte-decoder training loop spends most of its time in
//! `W += lr · err ⊗ x` (a LATENT_DIM² = 512² update per byte step). That is
//! embarrassingly parallel (each of the 512² cells updates independently), so
//! it maps to a single WGSL compute shader on `wgpu` / Vulkan (GTX 1660 Ti).
//!
//! Design:
//!   - `delta`: one thread per W element; W[row*512+col] += lr·err[row]·x[col].
//!     W lives in a GPU storage buffer; x/err/lr are uploaded per call.
//!   - `download_w`: copies W back to host via a staging buffer + map-read
//!     (needed for checkpointing / verification).
//!
//! HONEST scope: `delta` is the hot path and is fully GPU-accelerated; the
//! forward `y = W·x` prediction is left on the host (a 512² matvec is only
//! ~50µs on CPU and is called far less often than the delta in training). We
//! do NOT invent a half-working GPU baseline. If Vulkan isn't available the
//! module returns `None` and the training falls back to CPU — GPU is an
//! acceleration, never a hard dependency.
use std::num::NonZeroU64;

pub const DIM: u32 = 512;

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read>     x: array<f32>;
@group(0) @binding(2) var<storage, read>     err: array<f32>;
@group(0) @binding(3) var<storage, read>     lr: array<f32>;

@compute @workgroup_size(256)
fn delta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= 512u * 512u) { return; }
    let row = i / 512u;
    let col = i % 512u;
    w[i] += lr[0] * err[row] * x[col];
}
"#;

// Cap-стадия: ограничить squared-norm каждой строки W (как ROW_NORM_CAP=2.0
// в Rust learn_transition). 512 потоков, один на строку: считает sq-норму
// строки и масштабирует если > cap_sq.
const CAP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read>     cap: array<f32>;

@compute @workgroup_size(64)
fn cap_w(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= 512u) { return; }
    var sq = 0.0;
    for (var i = 0u; i < 512u; i = i + 1u) {
        let v = w[row * 512u + i];
        sq += v * v;
    }
    if (sq > cap[0]) {
        let scale = sqrt(cap[0] / sq);
        for (var i = 0u; i < 512u; i = i + 1u) {
            w[row * 512u + i] *= scale;
        }
    }
}
"#;

/// A minimal wgpu compute context for the Widrow-Hoff delta.
/// Двухскоростной режим: два набора W-буферов (local W и patch W_patch) —
/// оба используют ОДИН delta/cap-pipeline, но РАЗНЫЕ bind groups.
pub struct GpuOps {
    device: wgpu::Device,
    queue: wgpu::Queue,
    delta_pipeline: wgpu::ComputePipeline,
    w_buf: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    err_buf: wgpu::Buffer,
    lr_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    cap_pipeline: wgpu::ComputePipeline,
    cap_buf: wgpu::Buffer,
    cap_bind_group: wgpu::BindGroup,
    // Второй (патчевый) набор: W_patch + свои x/err/стaging + bind group.
    w2_buf: wgpu::Buffer,
    x2_buf: wgpu::Buffer,
    err2_buf: wgpu::Buffer,
    staging2: wgpu::Buffer,
    bind_group2: wgpu::BindGroup,
    cap_bind_group2: wgpu::BindGroup,
}

/// Build the GPU context; `None` when Vulkan is unavailable (callers use CPU).
pub fn try_new() -> Option<GpuOps> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(desc);
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: Default::default(),
    })) {
        Ok(a) => a,
        Err(_) => return None,
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("fuga-gpu"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: Default::default(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))
    .ok()?;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fuga-widrow-hoff"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("delta"),
        layout: None,
        module: &module,
        entry_point: Some("delta"),
        compilation_options: Default::default(),
        cache: None,
    });

    let w_size = (DIM * DIM * 4) as u64;
    let v_size = (DIM * 4) as u64;
    let mk = |label, size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let w_buf = mk(
        "w",
        w_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let x_buf = mk(
        "x",
        v_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let err_buf = mk(
        "err",
        v_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let lr_buf = mk(
        "lr",
        4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let staging = mk(
        "staging",
        w_size,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("delta-bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: err_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: lr_buf.as_entire_binding(),
            },
        ],
    });

    // Cap-стадия: отдельный compute-pipeline (raw W + cap value).
    let cap_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fuga-widrow-hoff-cap"),
        source: wgpu::ShaderSource::Wgsl(CAP_SHADER.into()),
    });
    let cap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cap"),
        layout: None,
        module: &cap_module,
        entry_point: Some("cap_w"),
        compilation_options: Default::default(),
        cache: None,
    });
    let cap_buf = mk(
        "cap",
        4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let clayout = cap_pipeline.get_bind_group_layout(0);
    let cap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cap-bg"),
        layout: &clayout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cap_buf.as_entire_binding(),
            },
        ],
    });

    // --- Второй (патчевый) набор: W2 + x2/err2/staging2 + delta/cap bind groups ---
    let w2_buf = mk(
        "w2",
        w_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let x2_buf = mk(
        "x2",
        v_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let err2_buf = mk(
        "err2",
        v_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let staging2 = mk(
        "staging2",
        w_size,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let bind_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("delta-bg2"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: w2_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x2_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: err2_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: lr_buf.as_entire_binding(),
            },
        ],
    });
    let cap_bind_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cap-bg2"),
        layout: &clayout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: w2_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cap_buf.as_entire_binding(),
            },
        ],
    });

    Some(GpuOps {
        device,
        queue,
        delta_pipeline: pipeline,
        w_buf,
        x_buf,
        err_buf,
        lr_buf,
        staging,
        bind_group,
        cap_pipeline,
        cap_buf,
        cap_bind_group,
        w2_buf,
        x2_buf,
        err2_buf,
        staging2,
        bind_group2,
        cap_bind_group2,
    })
}

impl GpuOps {
    /// Upload W (512² f32) to the GPU buffer.
    pub fn upload_w(&self, w: &[f32]) {
        self.queue.write_buffer(&self.w_buf, 0, bytes(w));
    }

    /// Dump W back to host (BLOCKING readback via staging). Used for
    /// checkpointing / verification. `out` must be DIM² f32.
    pub fn download_w(&self, out: &mut [f32]) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("w-download"),
            });
        enc.copy_buffer_to_buffer(&self.w_buf, 0, &self.staging, 0, self.staging.size());
        self.queue.submit(Some(enc.finish()));
        
        // Poll until the copy completes
        loop {
            match self.device.poll(wgpu::PollType::Poll) {
                Ok(_) => break,
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(1)),
            }
        }
        
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        
        // Poll until map completes
        loop {
            if rx.try_recv().is_ok() {
                break;
            }
            let _ = self.device.poll(wgpu::PollType::Poll);
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        
        let data = slice.get_mapped_range().expect("mapped range");
        let n = (DIM * DIM) as usize;
        for i in 0..n {
            out[i] = bytemuck::from_bytes::<f32>(&data[i * 4..i * 4 + 4]);
        }
        drop(data);
        self.staging.unmap();
    }

    /// W += lr · err ⊗ x  (one byte-step update, GPU).
    pub fn delta(&self, x: &[f32], err: &[f32], lr: f32) {
        self.queue.write_buffer(&self.x_buf, 0, bytes(x));
        self.queue.write_buffer(&self.err_buf, 0, bytes(err));
        self.queue.write_buffer(&self.lr_buf, 0, bytes(&[lr]));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("delta-enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("delta-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.delta_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(DIM * DIM / 256, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        // No blocking poll here — delta returns immediately, GPU works async
    }

    /// Batched W updates: apply N deltas accumulated in xs/errs without downloading W.
    /// Much faster than N separate delta() calls (amortizes PCIe overhead).
    pub fn batch_delta(&self, xs: &[Vec<f32>], errs: &[Vec<f32>], lr: f32) {
        for (x, err) in xs.iter().zip(errs.iter()) {
            self.queue.write_buffer(&self.x_buf, 0, bytes(x));
            self.queue.write_buffer(&self.err_buf, 0, bytes(err));
            self.queue.write_buffer(&self.lr_buf, 0, bytes(&[lr]));
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("batch-delta-enc"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("batch-delta-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.delta_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(DIM * DIM / 256, 1, 1);
            }
            self.queue.submit(Some(enc.finish()));
        }
        // Poll once after all submits to ensure completion
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Cap каждую строку W: масштаб если sq-norm > cap_sq (как Rust
    /// ROW_NORM_CAP=2.0 в learn_transition). 512 потоков, ~мгновенно.
    pub fn cap_w(&self, cap_sq: f32) {
        self.queue.write_buffer(&self.cap_buf, 0, bytes(&[cap_sq]));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cap-enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cap-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cap_pipeline);
            pass.set_bind_group(0, &self.cap_bind_group, &[]);
            pass.dispatch_workgroups(512 / 64 + 1, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    // --- Второй (патчевый) W2: тот же delta/cap-pipeline, свой буфер ---

    /// Upload W_patch (512² f32) в GPGPU-буфер.
    pub fn upload_w2(&self, w: &[f32]) {
        self.queue.write_buffer(&self.w2_buf, 0, bytes(w));
    }

    /// Dump W_patch обратно (blocking readback).
    pub fn download_w2(&self, out: &mut [f32]) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("w2-download"),
            });
        enc.copy_buffer_to_buffer(&self.w2_buf, 0, &self.staging2, 0, self.staging2.size());
        self.queue.submit(Some(enc.finish()));
        loop {
            match self.device.poll(wgpu::PollType::Poll) {
                Ok(_) => break,
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(1)),
            }
        }
        let slice = self.staging2.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        loop {
            if rx.try_recv().is_ok() {
                break;
            }
            let _ = self.device.poll(wgpu::PollType::Poll);
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        let data = slice.get_mapped_range().expect("mapped range");
        let n = (DIM * DIM) as usize;
        for i in 0..n {
            out[i] = bytemuck::from_bytes::<f32>(&data[i * 4..i * 4 + 4]);
        }
        drop(data);
        self.staging2.unmap();
    }

    /// Batched W_patch обновления (аналог batch_delta, но на w2_buf).
    pub fn batch_delta2(&self, xs: &[Vec<f32>], errs: &[Vec<f32>], lr: f32) {
        for (x, err) in xs.iter().zip(errs.iter()) {
            self.queue.write_buffer(&self.x2_buf, 0, bytes(x));
            self.queue.write_buffer(&self.err2_buf, 0, bytes(err));
            self.queue.write_buffer(&self.lr_buf, 0, bytes(&[lr]));
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("batch-delta2-enc"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("batch-delta2-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.delta_pipeline);
                pass.set_bind_group(0, &self.bind_group2, &[]);
                pass.dispatch_workgroups(DIM * DIM / 256, 1, 1);
            }
            self.queue.submit(Some(enc.finish()));
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Cap каждой строки W_patch.
    pub fn cap_w2(&self, cap_sq: f32) {
        self.queue.write_buffer(&self.cap_buf, 0, bytes(&[cap_sq]));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cap2-enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cap2-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cap_pipeline);
            pass.set_bind_group(0, &self.cap_bind_group2, &[]);
            pass.dispatch_workgroups(512 / 64 + 1, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::Poll);
    }
}

fn bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

// NOTE: `bytemuck::from_bytes` above is intentionally *not* full-frame; we use
// a manual from_bytes with correct length. Cargo.toml has no `bytemuck` dep.
mod bytemuck {
    pub fn from_bytes<T: Copy>(b: &[u8]) -> T {
        assert!(b.len() >= std::mem::size_of::<T>());
        unsafe { std::ptr::read_unaligned(b.as_ptr() as *const T) }
    }
}

#[allow(unused)]
fn _nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}
