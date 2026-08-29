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

use crate::ai::latent_jepa::LATENT_DIM;
pub const DIM: u32 = LATENT_DIM as u32;

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read>     x: array<f32>;
@group(0) @binding(2) var<storage, read>     err: array<f32>;
@group(0) @binding(3) var<storage, read>     lr: array<f32>;

@compute @workgroup_size(256)
fn delta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= 1024u * 1024u) { return; }
    let row = i / 1024u;
    let col = i % 1024u;
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
    if (row >= 1024u) { return; }
    var sq = 0.0;
    for (var i = 0u; i < 1024u; i = i + 1u) {
        let v = w[row * 1024u + i];
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

// KAN-delta: один поток на (o,i) — обновляет 2 активных сплайн-узла
// по x[i]: c[o,i,k] += lr · err[o] · hat_k(x[i]).
// Точное соответствие kan.rs: hat(k) — треугольник на [GRID[k-1], GRID[k+1]]
// с пиком 1.0 в GRID[k]; k0 = сегмент слева от xi (по умолчанию 0).
const KAN_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> c: array<f32>;
@group(0) @binding(1) var<storage, read>     x: array<f32>;
@group(0) @binding(2) var<storage, read>     err: array<f32>;
@group(0) @binding(3) var<storage, read>     lr: array<f32>;

fn grid(k: u32) -> f32 {
    if (k == 0u) { return -1.0; }
    if (k == 1u) { return -0.6; }
    if (k == 2u) { return -0.2; }
    if (k == 3u) { return 0.2; }
    if (k == 4u) { return 0.6; }
    return 1.0;
}

fn hat(k: u32, xi: f32) -> f32 {
    let lo = grid(max(k, 1u) - 1u);
    let hi = grid(min(k + 1u, 5u));
    let xk = grid(k);
    let span = hi - lo;
    if (span <= 0.0) { return 0.0; }
    if (xi < lo || xi > hi) { return 0.0; }
    if (xi <= xk) {
        return (xi - lo) / max(xk - lo, 1e-8);
    } else {
        return (hi - xi) / max(hi - xk, 1e-8);
    }
}

@compute @workgroup_size(256)
fn kan_delta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= 512u * 512u) { return; }
    let o = idx / 512u;
    let i = idx % 512u;
    let xi = x[i];
    var k0 = 0u;
    for (var k = 1u; k < 6u; k = k + 1u) {
        if (xi >= grid(k - 1u) && xi <= grid(k)) { k0 = k - 1u; break; }
    }
    let base = (o * 512u + i) * 6u;
    c[base + k0] += lr[0] * err[o] * hat(k0, xi);
    if (k0 + 1u < 6u) {
        c[base + k0 + 1u] += lr[0] * err[o] * hat(k0 + 1u, xi);
    }
}
"#;

// RESIDUAL-шейдер: честный LMS-остаток E[X] = T − W·X, считанный НА GPU.
// Исправляет критический баг Hebb-накопления (err≈target): теперь W-дельта
// и KAN-дельта получают реальную ошибку предсказания, а не сырой таргет.
// Один поток на строку o: err[o] = t[o] − Σ_i W[o,i]·x[i].
const RESIDUAL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read>     x: array<f32>;
@group(0) @binding(2) var<storage, read_write> err: array<f32>;
@group(0) @binding(3) var<storage, read>       t: array<f32>;

@compute @workgroup_size(64)
fn residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= 1024u) { return; }
    let row = o * 1024u;
    var pred = 0.0f;
    for (var i = 0u; i < 1024u; i = i + 1u) {
        pred += w[row + i] * x[i];
    }
    err[o] = t[o] - pred;
}
"#;

// KAN-кап НА GPU (мягкий, как kan.rs после калибровки): один поток на строку
// o (512 строк), считает норму всех 512×6=3072 коэфф строки и применяет
// scale = sqrt(CAP/(CAP+sq)). Раньше кап шёл через download→CPU→upload
// (6MB×2 per cap — узкое место конвейера); теперь во-врéмя на GPU.
const KAN_CAP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> c: array<f32>;
@group(0) @binding(1) var<storage, read>     cap: array<f32>;

@compute @workgroup_size(64)
fn kan_cap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= 1024u) { return; }
    var sq = 0.0;
    for (var i = 0u; i < 1024u; i = i + 1u) {
        for (var k = 0u; k < 6u; k = k + 1u) {
            let v = c[(o * 1024u + i) * 6u + k];
            sq += v * v;
        }
    }
    // Синхронно с CPU kan.rs cap_outputs: min(1, sqrt(cap/sq)) — ниже порога
    // НЕ трогаем (плавный cap/(cap+sq) резал даже малые нормы каждый кап).
    let scale = min(1.0, sqrt(cap[0] / max(sq, 1e-8)));
    if (abs(scale - 1.0) > 1e-6) {
        for (var i = 0u; i < 512u; i = i + 1u) {
            for (var k = 0u; k < 6u; k = k + 1u) {
                c[(o * 512u + i) * 6u + k] *= scale;
            }
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
    residual_pipeline: wgpu::ComputePipeline,
    w_buf: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    err_buf: wgpu::Buffer,
    t_buf: wgpu::Buffer,
    lr_buf: wgpu::Buffer,
    lr_kan_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    residual_bind_group: wgpu::BindGroup,
    cap_pipeline: wgpu::ComputePipeline,
    cap_buf: wgpu::Buffer,
    cap_bind_group: wgpu::BindGroup,
    // Второй (патчевый) набор: W_patch + свои x/err/стaging + bind group.
    w2_buf: wgpu::Buffer,
    x2_buf: wgpu::Buffer,
    err2_buf: wgpu::Buffer,
    t2_buf: wgpu::Buffer,
    staging2: wgpu::Buffer,
    bind_group2: wgpu::BindGroup,
    residual_bind_group2: wgpu::BindGroup,
    cap_bind_group2: wgpu::BindGroup,
    // KAN-набор: c[o,i,k] (512×512×6 f32) + свой pipeline/bind group.
    kan_c_buf: wgpu::Buffer,
    kan_staging: wgpu::Buffer,
    kan_pipeline: wgpu::ComputePipeline,
    kan_bind_group: wgpu::BindGroup,
    kan_cap_pipeline: wgpu::ComputePipeline,
    kan_cap_bind_group: wgpu::BindGroup,
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
    let t_buf = mk(
        "t",
        v_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let lr_buf = mk(
        "lr",
        4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    // Отдельный lr для KAN-пайплайна: запись в один lr_buf до submit
    // приводила к тому, что W-дельта училась со скоростью lr_kan.
    let lr_kan_buf = mk(
        "lr_kan",
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

    // Residual-модель: честный LMS-остаток err = t − W·x на GPU.
    let residual_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fuga-residual"),
        source: wgpu::ShaderSource::Wgsl(RESIDUAL_SHADER.into()),
    });
    let residual_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("residual"),
        layout: None,
        module: &residual_module,
        entry_point: Some("residual"),
        compilation_options: Default::default(),
        cache: None,
    });
    let rlayout = residual_pipeline.get_bind_group_layout(0);
    let residual_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("residual-bg"),
        layout: &rlayout,
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
                resource: t_buf.as_entire_binding(),
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
    let t2_buf = mk(
        "t2",
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
    let residual_bind_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("residual-bg2"),
        layout: &rlayout,
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
                resource: t2_buf.as_entire_binding(),
            },
        ],
    });

    // --- KAN-набор: c[o,i,k] (512×512×6 f32) + pipeline + bind group ---
    let kan_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fuga-kan-delta"),
        source: wgpu::ShaderSource::Wgsl(KAN_SHADER.into()),
    });
    let kan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("kan-delta"),
        layout: None,
        module: &kan_mod,
        entry_point: Some("kan_delta"),
        compilation_options: Default::default(),
        cache: None,
    });
    let kan_c_size = (DIM as usize * DIM as usize * 6 * 4) as u64;
    let kan_c_buf = mk(
        "kan_c",
        kan_c_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let kan_staging = mk(
        "kan_staging",
        kan_c_size,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let kan_layout = kan_pipeline.get_bind_group_layout(0);
    let kan_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("kan-bg"),
        layout: &kan_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: kan_c_buf.as_entire_binding(),
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
                resource: lr_kan_buf.as_entire_binding(),
            },
        ],
    });

    // KAN-cap pipeline: c + cap_buf (тот же cap_buf, что у W-cap).
    let kan_cap_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fuga-kan-cap"),
        source: wgpu::ShaderSource::Wgsl(KAN_CAP_SHADER.into()),
    });
    let kan_cap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("kan-cap"),
        layout: None,
        module: &kan_cap_module,
        entry_point: Some("kan_cap"),
        compilation_options: Default::default(),
        cache: None,
    });
    let kan_cap_layout = kan_cap_pipeline.get_bind_group_layout(0);
    let kan_cap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("kan-cap-bg"),
        layout: &kan_cap_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: kan_c_buf.as_entire_binding(),
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
        residual_pipeline,
        w_buf,
        x_buf,
        err_buf,
        t_buf,
        lr_buf,
        lr_kan_buf,
        staging,
        bind_group,
        residual_bind_group,
        cap_pipeline,
        cap_buf,
        cap_bind_group,
        w2_buf,
        x2_buf,
        err2_buf,
        t2_buf,
        staging2,
        bind_group2,
        residual_bind_group2,
        cap_bind_group2,
        kan_c_buf,
        kan_staging,
        kan_pipeline,
        kan_bind_group,
        kan_cap_pipeline,
        kan_cap_bind_group,
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
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
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

    /// Честный LMS-остаток: pred = W·x, err = t − pred (на GPU).
    /// Вызывает RESIDUAL-шейдер для local-канала.
    pub fn residual(&self, x: &[f32], t: &[f32]) {
        self.queue.write_buffer(&self.x_buf, 0, bytes(x));
        self.queue.write_buffer(&self.t_buf, 0, bytes(t));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("residual-enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("residual-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.residual_pipeline);
            pass.set_bind_group(0, &self.residual_bind_group, &[]);
            pass.dispatch_workgroups(512 / 64 + 1, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Честный LMS-остаток для патчевого канала (W_patch · x2 → err2).
    pub fn residual2(&self, x: &[f32], t: &[f32]) {
        self.queue.write_buffer(&self.x2_buf, 0, bytes(x));
        self.queue.write_buffer(&self.t2_buf, 0, bytes(t));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("residual2-enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("residual2-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.residual_pipeline);
            pass.set_bind_group(0, &self.residual_bind_group2, &[]);
            pass.dispatch_workgroups(512 / 64 + 1, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Полный гибридный шаг: residual (err = t − W·x) → W-delta → KAN-delta.
    /// ВАЖНО: это честный LMS, а не Hebb-накопление (err≈target) — исправляет
    /// зацикливание генерации. KAN получает ТОТ ЖЕ остаток, что и W.
    pub fn hybrid_step(&self, x: &[f32], t: &[f32], lr_w: f32, lr_kan: f32) {
        self.queue.write_buffer(&self.x_buf, 0, bytes(x));
        self.queue.write_buffer(&self.t_buf, 0, bytes(t));
        self.queue.write_buffer(&self.lr_buf, 0, bytes(&[lr_w]));
        self.queue.write_buffer(&self.lr_kan_buf, 0, bytes(&[lr_kan]));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hybrid-step-enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("residual-pass"), // сначала честный остаток
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.residual_pipeline);
            pass.set_bind_group(0, &self.residual_bind_group, &[]);
            pass.dispatch_workgroups(512 / 64 + 1, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("delta-pass"), // W += lr_w · err ⊗ x
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.delta_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(DIM * DIM / 256, 1, 1);
        }
        if lr_kan > 0.0 {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kan-delta-pass"), // KAN на честном остатке
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kan_pipeline);
            pass.set_bind_group(0, &self.kan_bind_group, &[]);
            pass.dispatch_workgroups(DIM * DIM / 256, 1, 1);
            drop(pass);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Явная синхронизация: один poll после пачки (batch dispatch паттерн —
    /// убирает launch-overhead из hybrid_step/hybrid_step2).
    pub fn sync(&self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Патчевый гибридный шаг: err2 = t − W_patch·x2 → W2-delta.
    pub fn hybrid_step2(&self, x: &[f32], t: &[f32], lr: f32) {
        self.queue.write_buffer(&self.x2_buf, 0, bytes(x));
        self.queue.write_buffer(&self.t2_buf, 0, bytes(t));
        self.queue.write_buffer(&self.lr_buf, 0, bytes(&[lr]));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hybrid2-step-enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("residual2-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.residual_pipeline);
            pass.set_bind_group(0, &self.residual_bind_group2, &[]);
            pass.dispatch_workgroups(512 / 64 + 1, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("delta2-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.delta_pipeline);
            pass.set_bind_group(0, &self.bind_group2, &[]);
            pass.dispatch_workgroups(DIM * DIM / 256, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
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
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
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

    // --- KAN: сплайн-обучение на GPU (c[o,i,k] += lr·err[o]·hat_k(x[i])) ---

    /// Upload сплайн-коэффициенты c (DIM²×6 f32).
    pub fn upload_kan(&self, c: &[f32]) {
        self.queue.write_buffer(&self.kan_c_buf, 0, bytes(c));
    }

    /// Batched KAN-обновления: тот же x/err, lr — в lr_kan_buf (kan_pipeline).
    pub fn kan_batch_delta(&self, xs: &[Vec<f32>], errs: &[Vec<f32>], lr: f32) {
        for (x, err) in xs.iter().zip(errs.iter()) {
            self.queue.write_buffer(&self.x_buf, 0, bytes(x));
            self.queue.write_buffer(&self.err_buf, 0, bytes(err));
            self.queue.write_buffer(&self.lr_kan_buf, 0, bytes(&[lr]));
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("kan-batch-enc"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("kan-batch-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.kan_pipeline);
                pass.set_bind_group(0, &self.kan_bind_group, &[]);
                pass.dispatch_workgroups(DIM * DIM / 256, 1, 1);
            }
            self.queue.submit(Some(enc.finish()));
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Dump сплайн-коэффициенты обратно (blocking, DIM²×6 f32).
    pub fn download_kan(&self, out: &mut [f32]) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kan-download"),
            });
        enc.copy_buffer_to_buffer(&self.kan_c_buf, 0, &self.kan_staging, 0, self.kan_staging.size());
        self.queue.submit(Some(enc.finish()));
        loop {
            match self.device.poll(wgpu::PollType::Poll) {
                Ok(_) => break,
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(1)),
            }
        }
        let slice = self.kan_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
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
        let n = (DIM as usize) * (DIM as usize) * 6;
        for i in 0..n {
            out[i] = bytemuck::from_bytes::<f32>(&data[i * 4..i * 4 + 4]);
        }
        drop(data);
        self.kan_staging.unmap();
    }

    /// Мягкий KAN-кап НА GPU (как kan.rs после калибровки: scale=sqrt(CAP/(CAP+sq))).
    /// Убирает download→CPU-cap→upload цикл (6MB×2 per cap) — узкое место.
    pub fn kan_cap_w(&self, cap: f32) {
        self.queue.write_buffer(&self.cap_buf, 0, bytes(&[cap]));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kan-cap-enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kan-cap-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kan_cap_pipeline);
            pass.set_bind_group(0, &self.kan_cap_bind_group, &[]);
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
