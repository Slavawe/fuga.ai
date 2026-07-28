use crate::core::hypervector::Hypervector;
use std::ffi::CString;
use std::ptr;
use std::sync::Mutex;

unsafe extern "C" {
    fn cuInit(flags: u32) -> u32;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> u32;
    fn cuCtxCreate(pctx: *mut u64, flags: u32, device: i32) -> u32;
    fn cuModuleLoadData(module: *mut u64, image: *const u8) -> u32;
    fn cuModuleGetFunction(hfunc: *mut u64, hmod: u64, name: *const i8) -> u32;
    fn cuMemAlloc(dptr: *mut u64, bytesize: u64) -> u32;
    fn cuMemcpyHtoD(dst: u64, src: *const u8, byte_count: u64) -> u32;
    fn cuMemcpyDtoH(dst: *mut u8, src: u64, byte_count: u64) -> u32;
    fn cuMemFree(dptr: u64) -> u32;
    fn cuLaunchKernel(
        f: u64, gx: u32, gy: u32, gz: u32,
        bx: u32, by: u32, bz: u32,
        shared_mem_bytes: u32, hstream: u64,
        kernel_params: *mut *const u8, extra: *mut *const u8,
    ) -> u32;
    fn cuCtxSynchronize() -> u32;
}

const CUDA_SUCCESS: u32 = 0;

fn check(err: u32, msg: &str) {
    if err != CUDA_SUCCESS {
        eprintln!("[cuda] {}: error={}", msg, err);
    }
}

static PTX_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/fuga_kernel.ptx"));

pub struct GpuContext {
    pub available: bool,
    module: u64,
    function_resonance: u64,
    function_entropy: u64,
}

fn init() -> Option<GpuContext> {
    unsafe {
        let err = cuInit(0);
        if err != CUDA_SUCCESS {
            eprintln!("[cuda] cuInit failed: {}", err);
            return None;
        }

        let mut device: i32 = 0;
        if cuDeviceGet(&mut device, 0) != CUDA_SUCCESS {
            eprintln!("[cuda] no device");
            return None;
        }

        let mut ctx: u64 = 0;
        if cuCtxCreate(&mut ctx, 0, device) != CUDA_SUCCESS {
            eprintln!("[cuda] ctx create failed");
            return None;
        }

        let ptx_cstr = CString::new(PTX_SOURCE).ok()?;
        let mut module: u64 = 0;
        if cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const u8) != CUDA_SUCCESS {
            eprintln!("[cuda] module load failed");
            return None;
        }

        let mut func_res: u64 = 0;
        let name_res = CString::new("resonance_scan").unwrap();
        if cuModuleGetFunction(&mut func_res, module, name_res.as_ptr()) != CUDA_SUCCESS {
            eprintln!("[cuda] no resonance_scan kernel");
            return None;
        }

        let mut func_ent: u64 = 0;
        let name_ent = CString::new("batch_entropy").unwrap();
        if cuModuleGetFunction(&mut func_ent, module, name_ent.as_ptr()) != CUDA_SUCCESS {
            eprintln!("[cuda] no batch_entropy kernel");
            return None;
        }

        eprintln!("[cuda] GTX 1660 Ti initialized — sm_75, 6GB VRAM");
        Some(GpuContext {
            available: true,
            module,
            function_resonance: func_res,
            function_entropy: func_ent,
        })
    }
}

pub fn init_gpu() {
    let mut gpu = GPU.lock().unwrap();
    *gpu = init();
}

pub fn is_gpu_available() -> bool {
    GPU.lock().unwrap().as_ref().map(|g| g.available).unwrap_or(false)
}

pub fn gpu_resonance_scan(query: &Hypervector, cells: &[Hypervector]) -> Option<Vec<f32>> {
    let gpu = GPU.lock().unwrap();
    let ctx = gpu.as_ref().filter(|g| g.available)?;

    let n = cells.len() as u32;
    let word_count = query.words.len() as u32;
    let cell_bytes = (word_count as u64) * 8 * (n as u64);
    let scores_bytes = (n as u64) * 4;
    let query_bytes = (word_count as u64) * 8;

    let mut scores = vec![0.0f32; n as usize];

    unsafe {
        let mut d_query: u64 = 0;
        let mut d_cells: u64 = 0;
        let mut d_scores: u64 = 0;

        check(cuMemAlloc(&mut d_query, query_bytes), "alloc query");
        check(cuMemAlloc(&mut d_cells, cell_bytes), "alloc cells");
        check(cuMemAlloc(&mut d_scores, scores_bytes), "alloc scores");

        let query_ptr = query.words.as_ptr() as *const u8;
        check(cuMemcpyHtoD(d_query, query_ptr, query_bytes), "cpy query");

        let mut flat: Vec<u64> = Vec::with_capacity(n as usize * word_count as usize);
        for cell in cells {
            flat.extend_from_slice(&cell.words);
        }
        check(cuMemcpyHtoD(d_cells, flat.as_ptr() as *const u8, cell_bytes), "cpy cells");

        let params: [*const u8; 5] = [
            &d_query as *const u64 as *const u8,
            &d_cells as *const u64 as *const u8,
            &d_scores as *const u64 as *const u8,
            &n as *const u32 as *const u8,
            &word_count as *const u32 as *const u8,
        ];
        let mut param_ptrs: Vec<*const u8> = params.iter().map(|p| *p).collect();
        param_ptrs.push(ptr::null());

        let block_size = 256u32;
        let grid_size = ((n + block_size - 1) / block_size).max(1);

        check(cuLaunchKernel(
            ctx.function_resonance,
            grid_size, 1, 1,
            block_size, 1, 1,
            0, 0,
            param_ptrs.as_mut_ptr(),
            ptr::null_mut(),
        ), "launch resonance_scan");

        check(cuCtxSynchronize(), "sync");

        check(cuMemcpyDtoH(scores.as_mut_ptr() as *mut u8, d_scores, scores_bytes), "cpy scores back");

        check(cuMemFree(d_query), "free query");
        check(cuMemFree(d_cells), "free cells");
        check(cuMemFree(d_scores), "free scores");
    }

    Some(scores)
}

pub fn gpu_memory_search(
    query: &Hypervector,
    all_vectors: &[&Hypervector],
    top_k: usize,
    threshold: f64,
) -> Option<Vec<(usize, f64)>> {
    let gpu = GPU.lock().unwrap();
    let ctx = gpu.as_ref().filter(|g| g.available)?;

    let total = all_vectors.len();
    if total == 0 {
        return Some(Vec::new());
    }
    let word_count = query.words.len() as u32;
    let max_batch = 65536usize;
    let query_bytes = (word_count as u64) * 8;

    let mut all_scores = Vec::with_capacity(total);

    unsafe {
        let mut d_query: u64 = 0;
        let mut d_db: u64 = 0;
        let mut d_scores: u64 = 0;

        check(cuMemAlloc(&mut d_query, query_bytes), "alloc query");
        check(cuMemAlloc(&mut d_db, (word_count as u64) * 8 * (max_batch as u64)), "alloc db");
        check(cuMemAlloc(&mut d_scores, (max_batch as u64) * 4), "alloc scores");

        let query_ptr = query.words.as_ptr() as *const u8;
        check(cuMemcpyHtoD(d_query, query_ptr, query_bytes), "cpy query");

        for chunk_start in (0..total).step_by(max_batch) {
            let chunk_end = (chunk_start + max_batch).min(total);
            let n = (chunk_end - chunk_start) as u32;

            let mut flat: Vec<u64> = Vec::with_capacity(n as usize * word_count as usize);
            for hv in &all_vectors[chunk_start..chunk_end] {
                flat.extend_from_slice(&hv.words);
            }
            check(cuMemcpyHtoD(d_db, flat.as_ptr() as *const u8, (word_count as u64) * 8 * (n as u64)), "cpy db");

            let params: [*const u8; 5] = [
                &d_query as *const u64 as *const u8,
                &d_db as *const u64 as *const u8,
                &d_scores as *const u64 as *const u8,
                &n as *const u32 as *const u8,
                &word_count as *const u32 as *const u8,
            ];
            let mut param_ptrs: Vec<*const u8> = params.iter().map(|p| *p).collect();
            param_ptrs.push(ptr::null());

            let block_size = 256u32;
            let grid_size = ((n + block_size - 1) / block_size).max(1);

            check(cuLaunchKernel(
                ctx.function_resonance,
                grid_size, 1, 1,
                block_size, 1, 1,
                0, 0,
                param_ptrs.as_mut_ptr(),
                ptr::null_mut(),
            ), "launch memory search");

            check(cuCtxSynchronize(), "sync");

            let mut chunk_scores = vec![0.0f32; n as usize];
            check(cuMemcpyDtoH(chunk_scores.as_mut_ptr() as *mut u8, d_scores, (n as u64) * 4), "cpy scores");
            all_scores.extend(chunk_scores);
        }

        check(cuMemFree(d_query), "free query");
        check(cuMemFree(d_db), "free db");
        check(cuMemFree(d_scores), "free scores");
    }

    let mut results: Vec<(usize, f64)> = all_scores.into_iter()
        .enumerate()
        .filter(|(_, s)| *s as f64 > threshold)
        .map(|(i, s)| (i, s as f64))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(top_k);
    Some(results)
}

static GPU: Mutex<Option<GpuContext>> = Mutex::new(None);