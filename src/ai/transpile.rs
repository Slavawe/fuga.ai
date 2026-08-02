// Streaming chunk-wise transpilation: DeepSeek-V4-Flash / Kimi-K3 safetensors
// shards -> deterministic VSA crystal (D = 8192).
//
// Pipeline per shard:
//   1. fetch 8-byte header size, then the JSON header (Range read)
//   2. enumerate tensors (name, dtype, shape, data_offsets)
//   3. Range-fetch each selected tensor, binarize into a WeightSketch
//   4. majority-vote accumulate into L0/L1/L2 + bounded L1 route matrix
//   5. drop the chunk from memory (never retained)
//
// Budget: dim=8192 -> 1KB per vector. L1 route matrix capped at ROUTE_CAP,
// so a full 30GB model streaming finishes at ~1.1MB on disk.

use crate::ai::crystal::{
    CrystalEntry, DEFAULT_DIM, DIM_L2, KIND_L0, KIND_L1, KIND_L2, PhaseCrystal, fnv1a,
};
use crate::ai::sdr::SDR_DENSITY;
use crate::core::hypervector::Hypervector;

pub const ROUTE_CAP: usize = 1024;
pub const CONCEPT_L0: &str = "concept:l0_syntax";
pub const CONCEPT_L1: &str = "concept:l1_phase";
pub const CONCEPT_L2: &str = "concept:l2_memory";

// --- dtype decoding --------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    F64,
    I64,
    I32,
    I16,
    I8,
    U32,
    U16,
    U8,
    F8E4M3,
    F8E5M2,
    F8E8M0,
    F4E2M1,
    Unsupported(String),
}

impl Dtype {
    pub fn from_str(s: &str) -> Self {
        match s {
            "F32" | "FLOAT32" => Dtype::F32,
            "F16" | "FLOAT16" => Dtype::F16,
            "BF16" | "BFLOAT16" => Dtype::BF16,
            "F64" | "FLOAT64" => Dtype::F64,
            "I64" | "INT64" => Dtype::I64,
            "I32" | "INT32" => Dtype::I32,
            "I16" | "INT16" => Dtype::I16,
            "I8" | "INT8" => Dtype::I8,
            "U32" | "UINT32" => Dtype::U32,
            "U16" | "UINT16" => Dtype::U16,
            "U8" | "UINT8" => Dtype::U8,
            "F8_E4M3" | "F8_E4M3FN" => Dtype::F8E4M3,
            "F8_E5M2" => Dtype::F8E5M2,
            "F8_E8M0" | "F8_E8M0FN" => Dtype::F8E8M0,
            "F4_E2M1" => Dtype::F4E2M1,
            other => Dtype::Unsupported(other.to_string()),
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::F64 => "F64",
            Dtype::I64 => "I64",
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
            Dtype::U32 => "U32",
            Dtype::U16 => "U16",
            Dtype::U8 => "U8",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::F8E5M2 => "F8_E5M2",
            Dtype::F8E8M0 => "F8_E8M0",
            Dtype::F4E2M1 => "F4_E2M1",
            Dtype::Unsupported(s) => s,
        }
    }

    pub fn bytes_per_elem(&self) -> usize {
        match self {
            Dtype::F4E2M1 => 0,
            Dtype::I8 | Dtype::U8 | Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::F8E8M0 => 1,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::Unsupported(_) => 0,
        }
    }

    pub fn supported(&self) -> bool {
        !matches!(self, Dtype::Unsupported(_))
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let frac = h & 0x3ff;
    let val = if exp == 0 {
        if frac == 0 {
            0.0
        } else {
            let f = frac as f32 / 1024.0;
            f * 2f32.powi(-14)
        }
    } else if exp == 31 {
        if frac == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        let f = 1.0 + frac as f32 / 1024.0;
        f * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -val } else { val }
}

fn f8e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 3) & 0x0f;
    let man = b & 0x07;
    let val = if exp == 0 {
        man as f32 * 2f32.powi(-6)
    } else if exp == 15 {
        if man == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + man as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    if sign == 1 { -val } else { val }
}

fn f8e5m2_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 2) & 0x1f;
    let man = b & 0x03;
    let val = if exp == 0 {
        man as f32 * 2f32.powi(-14)
    } else if exp == 31 {
        if man == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + man as f32 / 4.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -val } else { val }
}

/// MX micro-scaling E8M0: 1 sign + 7 exponent bits, bias 127, no mantissa.
/// value = sign * 2^(e - 127), e=0 -> 0. Used for per-block fp8 dequant scales.
fn f8e8m0_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let e = b & 0x7F;
    let val = if e == 0 {
        0.0
    } else {
        2f32.powi(e as i32 - 127)
    };
    if sign == 1 { -val } else { val }
}

fn f4e2m1_to_f32(n: u8) -> f32 {
    let sign = (n >> 3) & 1;
    let exp = (n >> 1) & 0x03;
    let man = n & 0x01;
    let val = if exp == 0 {
        man as f32 * 2f32.powi(-1)
    } else {
        (1.0 + man as f32 / 2.0) * 2f32.powi(exp as i32 - 1)
    };
    if sign == 1 { -val } else { val }
}

fn read_elem(dtype: &Dtype, data: &[u8], elem: usize) -> Option<f64> {
    match dtype {
        Dtype::F32 => {
            let p = elem * 4;
            if p + 4 > data.len() {
                return None;
            }
            Some(f32::from_le_bytes(data[p..p + 4].try_into().ok()?) as f64)
        }
        Dtype::F16 => {
            let p = elem * 2;
            if p + 2 > data.len() {
                return None;
            }
            Some(f16_to_f32(u16::from_le_bytes(data[p..p + 2].try_into().ok()?)) as f64)
        }
        Dtype::BF16 => {
            let p = elem * 2;
            if p + 2 > data.len() {
                return None;
            }
            let bits = (u16::from_le_bytes(data[p..p + 2].try_into().ok()?) as u32) << 16;
            Some(f32::from_bits(bits) as f64)
        }
        Dtype::F64 => {
            let p = elem * 8;
            if p + 8 > data.len() {
                return None;
            }
            Some(f64::from_le_bytes(data[p..p + 8].try_into().ok()?))
        }
        Dtype::I64 => {
            let p = elem * 8;
            if p + 8 > data.len() {
                return None;
            }
            Some(i64::from_le_bytes(data[p..p + 8].try_into().ok()?) as f64)
        }
        Dtype::I32 | Dtype::U32 => {
            let p = elem * 4;
            if p + 4 > data.len() {
                return None;
            }
            let raw = i32::from_le_bytes(data[p..p + 4].try_into().ok()?);
            Some(raw as f64)
        }
        Dtype::I16 | Dtype::U16 => {
            let p = elem * 2;
            if p + 2 > data.len() {
                return None;
            }
            Some(i16::from_le_bytes(data[p..p + 2].try_into().ok()?) as f64)
        }
        Dtype::I8 | Dtype::U8 => {
            if elem >= data.len() {
                return None;
            }
            Some(data[elem] as i8 as f64)
        }
        Dtype::F8E4M3 => {
            if elem >= data.len() {
                return None;
            }
            Some(f8e4m3_to_f32(data[elem]) as f64)
        }
        Dtype::F8E5M2 => {
            if elem >= data.len() {
                return None;
            }
            Some(f8e5m2_to_f32(data[elem]) as f64)
        }
        Dtype::F8E8M0 => {
            if elem >= data.len() {
                return None;
            }
            Some(f8e8m0_to_f32(data[elem]) as f64)
        }
        Dtype::F4E2M1 => {
            let byte = elem / 2;
            if byte >= data.len() {
                return None;
            }
            let b = data[byte];
            let n = if elem % 2 == 0 { b & 0x0f } else { b >> 4 };
            Some(f4e2m1_to_f32(n) as f64)
        }
        Dtype::Unsupported(_) => None,
    }
}

// --- safetensors header ----------------------------------------------------

#[derive(Clone, Debug)]
pub struct StTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub offset: usize,
    pub len: usize,
}

impl StTensor {
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// Parse the safetensors header block (must include the 8-byte length prefix).
pub fn parse_safetensors_header(bytes: &[u8]) -> Result<(String, Vec<StTensor>), String> {
    if bytes.len() < 8 {
        return Err("header too short".into());
    }
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + n {
        return Err(format!(
            "incomplete header: need {} bytes, got {}",
            8 + n,
            bytes.len()
        ));
    }
    let json: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + n]).map_err(|e| format!("bad header JSON: {}", e))?;
    let obj = json.as_object().ok_or("header is not an object")?;
    let mut metadata = String::new();
    let mut tensors = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            metadata = v.to_string();
            continue;
        }
        let dtype = v
            .get("dtype")
            .and_then(|d| d.as_str())
            .map(Dtype::from_str)
            .ok_or_else(|| format!("missing dtype for {}", name))?;
        let shape = v
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect::<Vec<u64>>())
            .unwrap_or_default();
        let offs = v
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .filter(|o| o.len() >= 2)
            .map(|o| {
                (
                    o[0].as_u64().unwrap_or(0) as usize,
                    o[1].as_u64().unwrap_or(0) as usize,
                )
            })
            .ok_or_else(|| format!("missing data_offsets for {}", name))?;
        tensors.push(StTensor {
            name: name.clone(),
            dtype,
            shape,
            offset: offs.0,
            len: offs.1 - offs.0,
        });
    }
    tensors.sort_by(|a, b| a.offset.cmp(&b.offset));
    Ok((metadata, tensors))
}

// --- sketch binarizer ------------------------------------------------------

/// splitmix64 finalizer: fast deterministic integer mixing.
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

/// Position-sensitive value-aware hashing into a dim-sized accumulator.
/// Two projections per element: position hash (sign/magnitude) and a
/// value-bucket hash (sign only) — captures both where a weight sits and how
/// large it is. NaN/Inf clamp to 0.
fn sketch_add(acc: &mut [f64], name_seed: u64, idx: u64, v: f64) {
    let w = if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let h = mix64(name_seed ^ idx.wrapping_mul(0x9E3779B97F4A7C15));
    let k = (h as usize) % acc.len();
    acc[k] += w;
    let bucket = (w * 8.0).round() as i8;
    let h2 = mix64(
        name_seed ^ idx.rotate_left(17) ^ ((bucket as u8) as u64).wrapping_mul(0x100000001b3),
    );
    let k2 = (h2 as usize) % acc.len();
    // NaN/Inf must contribute nothing: signum() of NaN is NaN, and NaN added
    // to a bucket permanently poisons it (NaN + x == NaN for any x), which
    // wiped out every bucket of large tensors with even one NaN element.
    acc[k2] += if v.is_finite() { v.signum() } else { 0.0 };
}

#[derive(Clone, Debug)]
pub struct WeightSketch {
    pub dim: usize,
    pub acc: Vec<f64>,
    pub n: u64,
    pub nz: u64,
}

impl WeightSketch {
    /// Final_Bits[i] = 1 if Acc[i] > 0 else 0, kept as a sparse top-margin SDR
    /// (top SDR_DENSITY by |Acc|). Sparse sketches: (a) keep the strongest,
    /// quantization-robust weight structure, (b) restore the "no resonance ->
    /// silence" property that dense ~50% sign vectors would destroy.
    pub fn to_hypervector(&self) -> Hypervector {
        sparse_from_acc(&self.acc, self.dim)
    }

    pub fn density(&self) -> f64 {
        if self.dim == 0 {
            return 0.0;
        }
        self.acc.iter().filter(|&&v| v > 0.0).count() as f64 / self.dim as f64
    }
}

/// Binarize a raw tensor payload into a WeightSketch over the flat index.
/// Uses the GPU tensor_to_sdr kernel when available (with a CPU fallback),
/// producing a bit-identical accumulator because the top-k sparsification
/// runs on the CPU in both paths.
pub fn binarize_tensor(name: &str, dtype: &Dtype, data: &[u8]) -> WeightSketch {
    let dim = DEFAULT_DIM;
    let name_seed = fnv1a(name.as_bytes());
    let count = match dtype {
        Dtype::F4E2M1 => data.len() * 2,
        Dtype::Unsupported(_) => 0,
        _ => data.len() / dtype.bytes_per_elem(),
    };

    let (acc, nz): (Vec<f64>, u64) = match count {
        0 => (vec![0.0f64; dim], 0),
        _ => {
            if let Some(code) = crate::gpu::dtype_code(dtype) {
                if let Some((gacc, gnz)) =
                    crate::gpu::gpu_tensor_to_acc(data, dim, code, name_seed, count)
                {
                    (gacc, gnz)
                } else {
                    cpu_accumulate(dim, name_seed, dtype, data, count)
                }
            } else {
                cpu_accumulate(dim, name_seed, dtype, data, count)
            }
        }
    };

    let n = count as u64;
    WeightSketch { dim, acc, n, nz }
}

fn cpu_accumulate(
    dim: usize,
    name_seed: u64,
    dtype: &Dtype,
    data: &[u8],
    count: usize,
) -> (Vec<f64>, u64) {
    let mut acc = vec![0.0f64; dim];
    let mut nz: u64 = 0;
    for i in 0..count {
        if let Some(v) = read_elem(dtype, data, i) {
            if v.is_finite() && v != 0.0 {
                nz += 1;
            }
            sketch_add(&mut acc, name_seed, i as u64, v);
        }
    }
    (acc, nz)
}

// --- level assignment (L0 syntax / L1 phase / L2 concept) -------------------

pub fn kind_for_name(name: &str) -> u8 {
    let n = name.to_lowercase();
    if n.contains("embed")
        || n.contains("lm_head")
        || n.contains("tok_emb")
        || n.contains("word_embed")
        || n.contains("token_embed")
        || n.contains("output.weight")
    {
        KIND_L0
    } else if n.contains("router")
        || n.contains("gate.weight")
        || n.contains("norm")
        || n.contains("attn")
        || n.contains("kda")
        || n.contains("q_proj")
        || n.contains("k_proj")
        || n.contains("v_proj")
        || n.contains("o_proj")
        || n.contains("wq")
        || n.contains("wkv")
        || n.contains("wo")
        || n.contains("mlp")
        || n.contains("up_proj")
        || n.contains("down_proj")
        || n.contains("experts")
        || n.contains("f_a")
        || n.contains("f_b")
        || n.contains("q_a")
        || n.contains("q_b")
        || n.contains("kv_a")
        || n.contains("kv_b")
        || n.contains("weight")
    {
        KIND_L1
    } else {
        KIND_L2
    }
}

/// MoE router heuristic: name contains "router", or the DeepSeek/Kimi-style
/// `layers.N.ffn.gate.weight` / `mlp.gate.weight` with a 2-D shape (n_experts x hidden).
fn is_router_tensor(name: &str, shape: &[u64]) -> bool {
    let n = name.to_lowercase();
    n.contains("router")
        || ((n.contains("ffn.gate") || n.contains("mlp.gate"))
            && n.ends_with(".weight")
            && shape.len() == 2)
}

/// MoE exclusion for raw-dump mode: routers, gates, and expert FFN tensors.
/// These are already captured as per-expert routes (see add_tensor), so the
/// full raw dump keeps only the dense/shared weights.
fn is_moe_tensor(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("router")
        || n.contains("gate")
        || n.contains("experts")
        || n.contains("expert")
        || n.contains("moe")
        || n.contains("shared_expert")
        || n.contains("e_dense")
        || n.contains("e_h")
        || n.contains("e_b")
}

// --- source abstraction (local file or HTTP Range) --------------------------

pub enum ShardSource {
    Local { path: String },
    Remote { base_url: String },
}

/// A shared HTTP agent reused across all remote shard fetches (keep-alive
/// connection pool — avoids a fresh TCP+TLS handshake per range request).
/// `thread_local` keeps one agent per worker thread, each with its own pool.
pub fn shared_agent() -> ureq::Agent {
    thread_local! {
        static AGENT: ureq::Agent = ureq::AgentBuilder::new()
            .timeout_read(std::time::Duration::from_secs(300))
            .timeout_connect(std::time::Duration::from_secs(60))
            .build();
    }
    AGENT.with(|a| a.clone())
}

impl ShardSource {
    /// Read the full header block: 8-byte length, then JSON.
    pub fn fetch_header(&self) -> Result<Vec<u8>, String> {
        match self {
            ShardSource::Local { path } => {
                let f = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
                let mut r = std::io::BufReader::new(f);
                use std::io::{Read, Seek, SeekFrom};
                let mut head = [0u8; 8];
                r.read_exact(&mut head)
                    .map_err(|e| format!("read {}: {}", path, e))?;
                let n = u64::from_le_bytes(head) as usize;
                let mut buf = Vec::with_capacity(8 + n);
                buf.extend_from_slice(&head);
                buf.resize(8 + n, 0);
                r.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
                r.read_exact(&mut buf)
                    .map_err(|e| format!("read header {}: {}", path, e))?;
                Ok(buf)
            }
            ShardSource::Remote { base_url } => {
                let agent = shared_agent();
                let first = fetch_range(&agent, base_url, 0, 8)?;
                let n = u64::from_le_bytes(first.as_slice().try_into().unwrap()) as usize;
                let mut buf = fetch_range(&agent, base_url, 0, 8 + n)?;
                if buf.len() < 8 + n {
                    return Err("incomplete remote header".into());
                }
                buf.truncate(8 + n);
                Ok(buf)
            }
        }
    }

    pub fn fetch_tensor(&self, data_start: u64, t: &StTensor) -> Result<Vec<u8>, String> {
        let begin = data_start + t.offset as u64;
        let end = begin + t.len as u64;
        match self {
            ShardSource::Local { path } => {
                let f = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
                use std::io::{Read, Seek, SeekFrom};
                let mut r = std::io::BufReader::new(f);
                r.seek(SeekFrom::Start(begin)).map_err(|e| e.to_string())?;
                let mut buf = vec![0u8; t.len];
                r.read_exact(&mut buf)
                    .map_err(|e| format!("read tensor {}: {}", t.name, e))?;
                Ok(buf)
            }
            ShardSource::Remote { base_url } => {
                let agent = shared_agent();
                fetch_range_retry(&agent, base_url, begin, end)
            }
        }
    }

    /// Fetch `len` bytes of the data section starting at data section offset `off`.
    fn fetch_slice(&self, data_start: u64, off: u64, len: u64) -> Result<Vec<u8>, String> {
        match self {
            ShardSource::Local { path } => {
                let f = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
                use std::io::{Read, Seek, SeekFrom};
                let mut r = std::io::BufReader::new(f);
                r.seek(SeekFrom::Start(data_start + off))
                    .map_err(|e| e.to_string())?;
                let mut buf = vec![0u8; len as usize];
                r.read_exact(&mut buf)
                    .map_err(|e| format!("read slice: {}", e))?;
                Ok(buf)
            }
            ShardSource::Remote { base_url } => {
                let agent = shared_agent();
                fetch_range_retry(&agent, base_url, data_start + off, data_start + off + len)
            }
        }
    }

    /// Total bytes of the data section of a shard, given its full header bytes.
    pub fn shard_data_len(&self, header: &[u8]) -> Result<u64, String> {
        let (_meta, tensors) = parse_safetensors_header(header)?;
        let max_end = tensors
            .iter()
            .map(|t| t.offset as u64 + t.len as u64)
            .max()
            .unwrap_or(0);
        Ok(max_end)
    }
}

/// HTTP GET with a `Range: bytes=A-B` header (1-based inclusive end).
fn fetch_range(
    agent: &ureq::Agent,
    url: &str,
    begin: usize,
    end: usize,
) -> Result<Vec<u8>, String> {
    if end <= begin {
        return Ok(Vec::new());
    }
    let range = format!("bytes={}-{}", begin, end - 1);
    let resp = agent
        .get(url)
        .set("Range", &range)
        .set("User-Agent", "fuga-transpile/0.1")
        .call()
        .map_err(|e| format!("range {} -> {}: {}", url, range, e))?;
    let mut buf = Vec::with_capacity(end - begin);
    use std::io::Read;
    resp.into_reader()
        .take((end - begin) as u64 + 1)
        .read_to_end(&mut buf)
        .map_err(|e| format!("read body: {}", e))?;
    if buf.len() < end - begin {
        return Err(format!(
            "short range read: got {} of {}",
            buf.len(),
            end - begin
        ));
    }
    buf.truncate(end - begin);
    Ok(buf)
}

const FETCH_CHUNK: u64 = 16 * 1024 * 1024;
const FETCH_RETRIES: usize = 4;

/// Fetch a range with retry + backoff on transient network errors.
fn fetch_range_retry(
    agent: &ureq::Agent,
    url: &str,
    begin: u64,
    end: u64,
) -> Result<Vec<u8>, String> {
    let mut attempt = 0usize;
    loop {
        match fetch_range(agent, url, begin as usize, end as usize) {
            Ok(buf) => return Ok(buf),
            Err(e) => {
                attempt += 1;
                if attempt > FETCH_RETRIES {
                    return Err(e);
                }
                let backoff = std::time::Duration::from_secs(2u64.pow(attempt as u32).min(30));
                eprintln!(
                    "  … retry {} ({:.1} MB range) in {:?}: {}",
                    attempt,
                    (end - begin) as f64 / 1_048_576.0,
                    backoff,
                    e
                );
                std::thread::sleep(backoff);
            }
        }
    }
}

/// A bounded, in-order chunk stream for a remote shard's data section.
///
/// Spawns `concurrency` worker threads, each pulling the next 256 MB range
/// via its own HTTP connection (like aria2c `-xN -sN`) and pushing results
/// into a channel. The consumer pulls chunks in index order, so tensors are
/// decoded in stream order while downloads for later chunks already run in
/// the background.
struct ChunkStream {
    rx: std::sync::mpsc::Receiver<(usize, Result<Vec<u8>, String>)>,
    next_idx: usize,
    pending: Vec<(usize, Result<Vec<u8>, String>)>,
}

impl ChunkStream {
    fn start(source: &ShardSource, data_start: u64, data_len: u64, concurrency: usize) -> Self {
        let nchunks = (data_len as usize + FETCH_CHUNK as usize - 1) / FETCH_CHUNK as usize;
        let url = match source {
            ShardSource::Remote { base_url } => base_url.clone(),
            ShardSource::Local { path } => path.clone(),
        };
        // Bounded channel: workers block when the consumer falls behind, so
        // at most `concurrency` chunks sit in flight regardless of shard size.
        let workers = concurrency.max(1).min(24);
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Result<Vec<u8>, String>)>(workers);
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // One shared Agent: clones share the connection pool, so workers reuse
        // keep-alive TCP+TLS instead of re-handshaking per chunk.
        let agent = shared_agent();
        for _ in 0..workers {
            let tx = tx.clone();
            let counter = counter.clone();
            let url = url.clone();
            let agent = agent.clone();
            let _ = data_start;
            std::thread::spawn(move || {
                loop {
                    let idx = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if idx >= nchunks {
                        break;
                    }
                    let begin = data_start + (idx as u64) * FETCH_CHUNK;
                    let end = (begin + FETCH_CHUNK).min(data_start + data_len);
                    let t0 = std::time::Instant::now();
                    let res = fetch_range_retry(&agent, &url, begin, end);
                    let dl = (end - begin) as f64 / 1_048_576.0;
                    let mbps = if t0.elapsed().as_secs_f64() > 0.0 {
                        dl / t0.elapsed().as_secs_f64()
                    } else {
                        0.0
                    };
                    eprintln!(
                        "  [chunk {}] {:.1} MB in {:.1}s = {:.1} MB/s",
                        idx,
                        dl,
                        t0.elapsed().as_secs_f64(),
                        mbps
                    );
                    if tx.send((idx, res)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        ChunkStream {
            rx,
            next_idx: 0,
            pending: Vec::new(),
        }
    }

    /// Return the next chunk in data-section order, blocking until it lands.
    fn next_chunk(&mut self) -> Result<Vec<u8>, String> {
        loop {
            if let Some(pos) = self.pending.iter().position(|(i, _)| *i == self.next_idx) {
                let (_, res) = self.pending.remove(pos);
                self.next_idx += 1;
                return res;
            }
            match self.rx.recv() {
                Ok((idx, res)) => {
                    if idx == self.next_idx {
                        self.next_idx += 1;
                        return res;
                    }
                    self.pending.push((idx, res));
                }
                Err(_) => {
                    if let Some(pos) = self.pending.iter().position(|(i, _)| *i == self.next_idx) {
                        let (_, res) = self.pending.remove(pos);
                        self.next_idx += 1;
                        return res;
                    }
                    return Err(format!(
                        "chunk stream ended early at chunk {}",
                        self.next_idx
                    ));
                }
            }
        }
    }
}

/// List `.safetensors` shards of an HF repo via the tree API.
pub fn list_hf_shards(repo: &str, revision: &str) -> Result<Vec<(String, u64)>, String> {
    let rev = if revision.is_empty() {
        "main"
    } else {
        revision
    };
    let host = std::env::var("FUGA_HF_HOST").unwrap_or_else(|_| "huggingface.co".to_string());
    let url = format!(
        "https://{}/api/models/{}/tree/{}?recursive=true",
        host, repo, rev
    );
    let agent = shared_agent();
    let resp = agent
        .get(&url)
        .set("User-Agent", "fuga-transpile/0.1")
        .call()
        .map_err(|e| format!("list {}: {}", url, e))?;
    let body = resp
        .into_string()
        .map_err(|e| format!("read list: {}", e))?;
    let arr: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("list JSON: {}", e))?;
    let mut out = Vec::new();
    if let Some(items) = arr.as_array() {
        for it in items {
            let path = it.get("path").and_then(|p| p.as_str()).unwrap_or("");
            let size = it.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
            if path.ends_with(".safetensors") {
                out.push((path.to_string(), size));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Resolve a shard path to its raw download URL.
/// The host can be overridden via the `FUGA_HF_HOST` env var (e.g. `hf-mirror.com`).
pub fn hf_resolve_url(repo: &str, path: &str, revision: &str) -> String {
    let rev = if revision.is_empty() {
        "main"
    } else {
        revision
    };
    let host = std::env::var("FUGA_HF_HOST").unwrap_or_else(|_| "huggingface.co".to_string());
    format!("https://{}/{}/resolve/{}/{}", host, repo, rev, path)
}

/// Extract a stable, mirror-independent shard label (file name like
/// `model-00030-of-00048.safetensors`) from any shard URL/path. Used so the
/// `done` resume list survives switching between ModelScope / HF mirrors.
pub fn shard_label(url_or_path: &str) -> String {
    if let Some(fp) = url_or_path.split("FilePath=").nth(1) {
        let fp = fp.split('&').next().unwrap_or(fp);
        let fp = fp.split('?').next().unwrap_or(fp);
        if !fp.is_empty() {
            return fp.to_string();
        }
    }
    url_or_path
        .rsplit('/')
        .next()
        .unwrap_or(url_or_path)
        .to_string()
}

/// List `.safetensors` shards of a ModelScope repo via the files API.
/// Falls back to "master" revision when the revision is empty. ModelScope
/// uses subdirectories (tree entries) which are walked recursively.
pub fn list_ms_shards(repo: &str, revision: &str) -> Result<Vec<(String, u64)>, String> {
    fn walk(
        agent: &ureq::Agent,
        repo: &str,
        rev: &str,
        root: &str,
        out: &mut Vec<(String, u64)>,
    ) -> Result<(), String> {
        let url = format!(
            "https://modelscope.cn/api/v1/models/{}/repo/files?Revision={}&Root={}",
            repo, rev, root
        );
        let resp = agent
            .get(&url)
            .set("User-Agent", "fuga-transpile/0.1")
            .call()
            .map_err(|e| format!("list {}: {}", url, e))?;
        let body = resp
            .into_string()
            .map_err(|e| format!("read list: {}", e))?;
        let v: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| format!("list JSON: {}", e))?;
        let files = v.get("Data").and_then(|d| d.get("Files"));
        let mut dirs = Vec::new();
        if let Some(arr) = files.and_then(|f| f.as_array()) {
            for it in arr {
                let path = it.get("Path").and_then(|p| p.as_str()).unwrap_or("");
                let size = it.get("Size").and_then(|s| s.as_u64()).unwrap_or(0);
                let typ = it.get("Type").and_then(|t| t.as_str()).unwrap_or("blob");
                if typ == "tree" {
                    dirs.push(path.to_string());
                } else if path.ends_with(".safetensors") {
                    out.push((path.to_string(), size));
                }
            }
        }
        for d in dirs {
            let sub = format!("{}/", d.trim_end_matches('/'));
            walk(agent, repo, rev, &sub, out)?;
        }
        Ok(())
    }
    let rev = if revision.is_empty() {
        "master"
    } else {
        revision
    };
    let agent = shared_agent();
    let mut out = Vec::new();
    walk(&agent, repo, &rev, "", &mut out)?;
    out.sort();
    Ok(out)
}

/// Resolve a ModelScope shard path to its raw download URL. The API responds
/// with a 302 redirect to a signed CDN URL with a fresh auth_key, so the URL
/// itself stays valid; each request gets a new signature.
pub fn ms_resolve_url(repo: &str, path: &str, revision: &str) -> String {
    let rev = if revision.is_empty() {
        "master"
    } else {
        revision
    };
    format!(
        "https://modelscope.cn/api/v1/models/{}/repo?Revision={}&FilePath={}",
        repo, rev, path
    )
}

// --- accumulator ------------------------------------------------------------

#[derive(Clone)]
pub struct TranspileConfig {
    pub select: Vec<String>,
    pub keep: Vec<String>,
    pub max_tensors: Option<usize>,
    pub max_shards: Option<usize>,
    pub route_cap: usize,
    pub dry_run: bool,
    pub whole: bool,
    pub concurrency: usize,
    /// Full raw-dump mode: every tensor becomes its own phase entry (not just
    /// the `keep` substrings). MoE tensors (routers/gates/experts) are still
    /// excluded — they are already in the crystal as per-expert routes.
    pub raw: bool,
}

impl Default for TranspileConfig {
    fn default() -> Self {
        TranspileConfig {
            select: Vec::new(),
            keep: vec![
                "embed".into(),
                "lm_head".into(),
                "gate".into(),
                "router".into(),
            ],
            max_tensors: None,
            max_shards: None,
            route_cap: ROUTE_CAP,
            dry_run: false,
            whole: false,
            concurrency: 8,
            raw: false,
        }
    }
}

pub struct TranspileAccumulator {
    pub dim: usize,
    pub l0: Vec<f64>,
    pub l1: Vec<f64>,
    pub l2: Vec<f64>,
    pub entries: Vec<CrystalEntry>,
    pub l0_index: std::collections::HashMap<u64, usize>,
    pub processed: usize,
    pub bytes_in: u64,
    pub skipped: Vec<String>,
    pub route_cap: usize,
    pub done: Vec<String>,
}

impl TranspileAccumulator {
    pub fn new(dim: usize) -> Self {
        TranspileAccumulator {
            dim,
            l0: vec![0.0; dim],
            l1: vec![0.0; dim],
            l2: vec![0.0; dim],
            entries: Vec::new(),
            l0_index: std::collections::HashMap::new(),
            processed: 0,
            bytes_in: 0,
            skipped: Vec::new(),
            route_cap: ROUTE_CAP,
            done: Vec::new(),
        }
    }

    fn state_push_str(buf: &mut Vec<u8>, s: &str) {
        let b = s.as_bytes();
        buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
        buf.extend_from_slice(b);
    }

    fn state_take_str(data: &[u8], pos: &mut usize) -> Result<String, String> {
        if *pos + 4 > data.len() {
            return Err("state: short string length".into());
        }
        let n = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap()) as usize;
        *pos += 4;
        if *pos + n > data.len() {
            return Err("state: short string body".into());
        }
        let s = String::from_utf8(data[*pos..*pos + n].to_vec()).map_err(|e| e.to_string())?;
        *pos += n;
        Ok(s)
    }

    /// Persist accumulator state between shards so a long streaming run can
    /// resume after a dropped Range request. Checkpointed after each shard.
    pub fn save_state(&self, path: &str) -> Result<(), String> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"FUGA_ST1");
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.route_cap as u32).to_le_bytes());
        buf.extend_from_slice(&(self.processed as u32).to_le_bytes());
        buf.extend_from_slice(&self.bytes_in.to_le_bytes());
        for v in [&self.l0, &self.l1, &self.l2] {
            for x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        buf.extend_from_slice(&(self.skipped.len() as u32).to_le_bytes());
        for s in &self.skipped {
            Self::state_push_str(&mut buf, s);
        }
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        let wc = (self.dim + 63) / 64;
        for e in &self.entries {
            for w in &e.hv.words {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            buf.extend_from_slice(&e.key.to_le_bytes());
            Self::state_push_str(&mut buf, &e.key_text);
            buf.extend_from_slice(&e.resonance.to_le_bytes());
            buf.extend_from_slice(&e.route.to_le_bytes());
            buf.push(e.kind);
            Self::state_push_str(&mut buf, &e.text);
        }
        buf.extend_from_slice(&(self.done.len() as u32).to_le_bytes());
        for s in &self.done {
            Self::state_push_str(&mut buf, s);
        }
        std::fs::write(path, &buf).map_err(|e| format!("save state {}: {}", path, e))
    }

    pub fn load_state(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read state {}: {}", path, e))?;
        if data.len() < 8 || &data[..8] != b"FUGA_ST1" {
            return Err("state: bad magic".into());
        }
        let mut pos = 8usize;
        let dim = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let route_cap = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let processed = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let bytes_in = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let mut l0 = vec![0.0; dim];
        let mut l1 = vec![0.0; dim];
        let mut l2 = vec![0.0; dim];
        for v in [&mut l0, &mut l1, &mut l2] {
            for x in v.iter_mut() {
                *x = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                pos += 8;
            }
        }
        let mut skipped = Vec::new();
        let nskip = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        for _ in 0..nskip {
            skipped.push(Self::state_take_str(&data, &mut pos)?);
        }
        let wc = (dim + 63) / 64;
        let mut entries = Vec::new();
        let nent = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        for _ in 0..nent {
            if pos + wc * 8 > data.len() {
                return Err("state: short hv".into());
            }
            let mut words = vec![0u64; wc];
            for j in 0..wc {
                words[j] =
                    u64::from_le_bytes(data[pos + j * 8..pos + (j + 1) * 8].try_into().unwrap());
            }
            pos += wc * 8;
            let key = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let key_text = Self::state_take_str(&data, &mut pos)?;
            let resonance = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let route = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
            pos += 2;
            let kind = data[pos];
            pos += 1;
            let text = Self::state_take_str(&data, &mut pos)?;
            entries.push(CrystalEntry {
                hv: Hypervector::from_raw(dim, words),
                key,
                key_text,
                resonance,
                route,
                kind,
                text,
            });
        }
        let mut done = Vec::new();
        let ndone = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        for _ in 0..ndone {
            let raw = Self::state_take_str(&data, &mut pos)?;
            done.push(shard_label(&raw));
        }
        let mut l0_index = std::collections::HashMap::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            l0_index.insert(e.key, i);
        }
        Ok(TranspileAccumulator {
            dim,
            l0,
            l1,
            l2,
            entries,
            l0_index,
            processed,
            bytes_in,
            skipped,
            route_cap,
            done,
        })
    }

    fn merge(&mut self, kind: u8, sketch: &WeightSketch) {
        let target: &mut Vec<f64> = match kind {
            KIND_L0 => &mut self.l0,
            KIND_L1 => &mut self.l1,
            _ => &mut self.l2,
        };
        for (t, s) in target.iter_mut().zip(sketch.acc.iter()) {
            *t += s;
        }
        // majority-vote memory accumulator: every tensor contributes to L2.
        if kind != KIND_L2 {
            for (t, s) in self.l2.iter_mut().zip(sketch.acc.iter()) {
                *t += s;
            }
        }
    }

    fn push_entry(
        &mut self,
        hv: Hypervector,
        key_text: String,
        resonance: f32,
        kind: u8,
        text: String,
    ) {
        let key = fnv1a(key_text.as_bytes());
        let route = ((key >> 8) & 0xFF) as u16;
        let entry = CrystalEntry {
            hv,
            key,
            key_text,
            resonance,
            route,
            kind,
            text,
        };
        let idx = self.entries.len();
        self.l0_index.insert(entry.key, idx);
        self.entries.push(entry);
    }

    /// Process one tensor payload into the accumulator.
    pub fn add_tensor(
        &mut self,
        name: &str,
        dtype: &Dtype,
        shape: &[u64],
        data: &[u8],
        keep: bool,
    ) {
        if !dtype.supported() {
            self.skipped
                .push(format!("{} [{} unsupported]", name, dtype.label()));
            return;
        }
        let sketch = binarize_tensor(name, dtype, data);
        let kind = kind_for_name(name);
        self.merge(kind, &sketch);

        // L1 phase matrix: extract per-expert route HVs from MoE router tensors.
        if is_router_tensor(name, shape) && shape.len() == 2 && self.route_cap > 0 {
            let rows = shape[0] as usize;
            let row_bytes = if dtype.bytes_per_elem() == 0 {
                (shape[1] as usize + 1) / 2
            } else {
                shape[1] as usize * dtype.bytes_per_elem()
            };
            for e in 0..rows {
                if self.route_cap == 0 {
                    break;
                }
                let begin = e * row_bytes;
                let end = (begin + row_bytes).min(data.len());
                if begin >= end {
                    continue;
                }
                let row = &data[begin..end];
                let rs = binarize_tensor(&format!("{}:expert_{}", name, e), dtype, row);
                let hv = rs.to_hypervector();
                let key_text = format!("{}:expert_{}", name, e);
                let resonance = (rs.nz as f64 / rs.n.max(1) as f64).clamp(0.0, 1.0) as f32;
                let text = format!(
                    "{} shape=[{}, {}] dtype={} n={} nz={}",
                    key_text,
                    shape[0],
                    shape[1],
                    dtype.label(),
                    rs.n,
                    rs.nz
                );
                self.push_entry(hv, key_text, resonance, KIND_L1, text);
                self.route_cap -= 1;
            }
        }

        if keep {
            let hv = sketch.to_hypervector();
            let resonance = (sketch.nz as f64 / sketch.n.max(1) as f64).clamp(0.0, 1.0) as f32;
            let text = format!(
                "{} shape={:?} dtype={} n={} nz={}",
                name,
                shape,
                dtype.label(),
                sketch.n,
                sketch.nz
            );
            self.push_entry(hv, name.to_string(), resonance, kind, text);
        }

        self.processed += 1;
        self.bytes_in += data.len() as u64;
    }

    /// Threshold-binarize the majority-vote accumulators into L2 concept vectors
    /// and assemble the PhaseCrystal (L1 matrix + L0/L1/L2 concepts + L0 index).
    ///
    /// Note: the crystal dump format serializes L1 + L2 entries only, so any
    /// kept L0 entry is re-labeled KIND_L1 (a phase profile); the L0 syntax
    /// level itself is carried by the merged `concept:l0_syntax` vector.
    pub fn finalize(&self, threshold: f64) -> PhaseCrystal {
        let mut crystal = PhaseCrystal::new(self.dim, threshold);
        for e in &self.entries {
            let kind = if e.kind == KIND_L0 { KIND_L1 } else { e.kind };
            let mut ce = e.clone();
            ce.kind = kind;
            crystal.entries.push(ce);
            crystal.l0_index.insert(e.key, crystal.entries.len() - 1);
        }
        for (acc, tag) in [
            (&self.l0, CONCEPT_L0),
            (&self.l1, CONCEPT_L1),
            (&self.l2, CONCEPT_L2),
        ] {
            let hv = sparse_from_acc(acc, DIM_L2);
            let key_text = tag.to_string();
            let key = fnv1a(tag.as_bytes());
            let route = ((key >> 8) & 0xFF) as u16;
            let entry = CrystalEntry {
                hv,
                key,
                key_text: key_text.clone(),
                resonance: 0.0,
                route,
                kind: KIND_L2,
                text: key_text,
            };
            let idx = crystal.entries.len();
            crystal.l0_index.insert(key, idx);
            crystal.entries.push(entry);
        }
        crystal
    }

    pub fn stats(&self) -> String {
        format!(
            "accumulator: {} tensors, {} bytes, {} entries, {} skipped, dim={}",
            self.processed,
            self.bytes_in,
            self.entries.len(),
            self.skipped.len(),
            self.dim,
        )
    }
}

/// Sparse SDR from a majority-vote accumulator: keep only the top SDR_DENSITY
/// fraction of bits by |acc| margin. Dense sign-binarization would make every
/// concept vector ~50% dense and false-positive on noise; margin-top selection
/// restores the "no resonance -> silence" property.
fn sparse_from_acc(acc: &[f64], dim: usize) -> Hypervector {
    let k = ((dim as f64 * SDR_DENSITY).ceil() as usize).max(1);
    let mut scored: Vec<(usize, f64)> = acc
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > 0.0)
        .map(|(i, v)| (i, v.abs()))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let keep = scored.len().min(k);
    let mut words = vec![0u64; (dim + 63) / 64];
    for (i, _) in scored.into_iter().take(keep) {
        words[i / 64] |= 1u64 << (i % 64);
    }
    Hypervector::from_raw(dim, words)
}

// --- top-level shard transpilation ------------------------------------------

pub struct TranspileStats {
    pub shard: String,
    pub tensors: usize,
    pub bytes: u64,
    pub entries_added: usize,
    pub elapsed: std::time::Duration,
    pub mbps: f64,
}

pub fn transpile_shard(
    source: &ShardSource,
    acc: &mut TranspileAccumulator,
    cfg: &TranspileConfig,
) -> Result<TranspileStats, String> {
    let label = match source {
        ShardSource::Local { path } => path.clone(),
        ShardSource::Remote { base_url } => base_url.clone(),
    };
    let start = std::time::Instant::now();
    let header = source.fetch_header()?;
    let (_meta, tensors) = parse_safetensors_header(&header)?;
    let data_start = header.len() as u64;

    if cfg.dry_run {
        println!("  {} tensors in {}:", tensors.len(), label);
        for t in tensors.iter().take(64) {
            println!(
                "    {:<60} {:?} {:?} ({} B)",
                t.name,
                t.shape,
                t.dtype.label(),
                t.len
            );
        }
        if tensors.len() > 64 {
            println!("    … {} more", tensors.len() - 64);
        }
        return Ok(TranspileStats {
            shard: label,
            tensors: tensors.len(),
            bytes: 0,
            entries_added: 0,
            elapsed: start.elapsed(),
            mbps: 0.0,
        });
    }

    let total = tensors.len();
    let mut processed = 0usize;
    let mut bytes = 0u64;
    let entries_before = acc.entries.len();
    let max_tensors = cfg.max_tensors.unwrap_or(usize::MAX);

    // Sliding-window whole-shard mode: stream the data section in bounded
    // chunks, keeping only a small window in memory. This avoids both the
    // HF multi-GB socket resets and the OOM kill from buffering an entire
    // 3.4 GB shard on a low-RAM host. Remote shards are downloaded through
    // a parallel chunk stream (multiple connections, like aria2c -xN -sN).
    let data_len = if cfg.whole || (cfg.raw && matches!(source, ShardSource::Remote { .. })) {
        source.shard_data_len(&header)?
    } else {
        0
    };
    let mut chunk_stream = match source {
        ShardSource::Remote { .. } if (cfg.whole || cfg.raw) && data_len > 0 => Some(
            ChunkStream::start(source, data_start, data_len, cfg.concurrency),
        ),
        _ => None,
    };

    let mut window: Vec<u8> = Vec::new();
    let mut win_start: u64 = 0;
    let mut win_pos: usize = 0;
    let mut tensors_done = 0usize;

    for t in tensors {
        if processed >= max_tensors {
            break;
        }
        let selected =
            cfg.select.is_empty() || cfg.select.iter().any(|s| t.name.contains(s.as_str()));
        if !selected {
            continue;
        }
        let keep = if cfg.raw {
            !is_moe_tensor(&t.name)
        } else {
            cfg.keep.iter().any(|s| t.name.contains(s.as_str()))
        };
        if !t.dtype.supported() {
            acc.skipped
                .push(format!("{} [{} unsupported]", t.name, t.dtype.label()));
            continue;
        }
        let off = t.offset as u64;
        if cfg.whole || cfg.raw {
            if off < win_start + win_pos as u64 {
                // Out-of-order tensor behind the read cursor: fetch separately.
                let data = source.fetch_tensor(data_start, &t)?;
                acc.add_tensor(&t.name, &t.dtype, &t.shape, &data, keep);
            } else {
                let rel = (off - win_start) as usize;
                // Extend window first so it covers [rel, rel + t.len).
                while rel + t.len > window.len() {
                    let part = match &mut chunk_stream {
                        Some(stream) => stream.next_chunk()?,
                        None => {
                            let fb = win_start + window.len() as u64;
                            let fl = FETCH_CHUNK.min(data_len - fb);
                            if fl == 0 {
                                return Err(format!(
                                    "window past end of shard data (tensor {})",
                                    t.name
                                ));
                            }
                            source.fetch_slice(data_start, fb, fl)?
                        }
                    };
                    window.extend_from_slice(&part);
                    println!(
                        "  … whole-shard {:.1}% ({:.1} GB fetched)",
                        (win_start + window.len() as u64) as f64 * 100.0 / data_len as f64,
                        (win_start + window.len() as u64) as f64 / 1_073_741_824.0
                    );
                }
                // Now drop the consumed prefix, keeping the window bounded.
                if rel > FETCH_CHUNK as usize / 2 {
                    window.drain(..rel);
                    win_start += rel as u64;
                    win_pos = 0;
                } else {
                    win_pos = rel;
                }
                let data = &window[win_pos..win_pos + t.len];
                acc.add_tensor(&t.name, &t.dtype, &t.shape, data, keep);
                win_pos += t.len;
            }
            tensors_done += 1;
            processed += 1;
            if tensors_done % 100 == 0 {
                println!(
                    "  … {}/{} tensors ({:.1} MB, {})",
                    tensors_done,
                    total,
                    (win_start + win_pos as u64) as f64 / 1_048_576.0,
                    label
                );
            }
        } else {
            let data = source.fetch_tensor(data_start, &t)?;
            acc.add_tensor(&t.name, &t.dtype, &t.shape, &data, keep);
            processed += 1;
            if processed % 100 == 0 {
                println!(
                    "  … {}/{} tensors ({:.1} MB, {})",
                    processed,
                    total,
                    bytes as f64 / 1_048_576.0,
                    label
                );
            }
        }
        bytes += t.len as u64;
    }
    let elapsed = start.elapsed();
    let mbps = if elapsed.as_secs_f64() > 0.0 {
        bytes as f64 / 1_048_576.0 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    Ok(TranspileStats {
        shard: label,
        tensors: processed,
        bytes,
        entries_added: acc.entries.len() - entries_before,
        elapsed,
        mbps,
    })
}

/// Guess whether a CLI source is an HF repo id, a URL, a directory, or a file.
pub fn is_repo_id(s: &str) -> bool {
    !s.starts_with("http") && s.contains('/') && !std::path::Path::new(s).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn build_synth(header: &[u8], tensors: &[(&str, &[u8])]) -> Vec<u8> {
        let mut offsets = Vec::new();
        let mut pos = 0usize;
        for (_, data) in tensors {
            offsets.push((pos, pos + data.len()));
            pos += data.len();
        }
        let mut body = String::from("{");
        for (i, (name, _)) in tensors.iter().enumerate() {
            let n = name.replace('/', "::");
            if i > 0 {
                body.push(',');
            }
            body.push_str(&format!(
                "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                n, 0, offsets[i].0, offsets[i].1
            ));
        }
        body.push('}');
        let n = body.len();
        let mut out = Vec::new();
        out.extend_from_slice(&(n as u64).to_le_bytes());
        out.extend_from_slice(body.as_bytes());
        for (_, data) in tensors {
            out.extend_from_slice(data);
        }
        out
    }

    #[test]
    fn parse_header_and_binarize() {
        let data = f32_bytes(&[0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.0, 3.0]);
        let raw = build_synth(b"", &[("tensor/0", data.as_slice())]);
        let (_m, tensors) = parse_safetensors_header(&raw).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].dtype, Dtype::F32);
        assert_eq!(tensors[0].len, data.len());
        let sk = binarize_tensor(&tensors[0].name, &tensors[0].dtype, &data);
        assert_eq!(sk.n, 8);
        assert_eq!(sk.nz, 7);
        let hv = sk.to_hypervector();
        assert_eq!(hv.words.len(), DEFAULT_DIM / 64);
    }

    fn overlap_fraction(a: &Hypervector, b: &Hypervector) -> f64 {
        let mut inter = 0u64;
        let mut q = 0u64;
        for (wa, wb) in a.words.iter().zip(b.words.iter()) {
            inter += (wa & wb).count_ones() as u64;
            q += wa.count_ones() as u64;
        }
        if q == 0 { 0.0 } else { inter as f64 / q as f64 }
    }

    #[test]
    fn sketch_is_deterministic_and_quantization_robust() {
        // same tensor -> identical sketch
        let a = f32_bytes(&[0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.0, 3.0, 1.5, -2.0]);
        let sk1 = binarize_tensor("w", &Dtype::F32, &a);
        let sk2 = binarize_tensor("w", &Dtype::F32, &a);
        assert_eq!(sk1.acc, sk2.acc);
        assert_eq!(
            overlap_fraction(&sk1.to_hypervector(), &sk2.to_hypervector()),
            1.0
        );

        // quantized (8-bit-rounded) version of the same tensor stays close
        let q: Vec<f32> = a
            .chunks_exact(4)
            .map(|c| {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                (v * 8.0).round() / 8.0
            })
            .collect();
        let skq = binarize_tensor("w", &Dtype::F32, &f32_bytes(&q));
        let same = overlap_fraction(&sk1.to_hypervector(), &skq.to_hypervector());
        assert!(same > 0.35, "quantized variant too far: {:.3}", same);

        // unrelated tensor (different name -> scattered positions) is far
        let other = f32_bytes(&[9.0, -9.0, 8.0, -8.0, 7.0, -7.0, 6.0, -6.0, 5.0, -5.0]);
        let sko = binarize_tensor("other", &Dtype::F32, &other);
        let far = overlap_fraction(&sk1.to_hypervector(), &sko.to_hypervector());
        assert!(far < 0.15, "unrelated tensor too close: {:.3}", far);
    }
}
