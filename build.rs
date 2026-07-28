use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/fuga_kernel.cu");

    let cuda_path = PathBuf::from("/opt/cuda");
    let cuda_bin = cuda_path.join("bin");
    let cuda_lib = cuda_path.join("lib64");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_src = std::fs::canonicalize("native/fuga_kernel.cu").unwrap();

    let nvcc = cuda_bin.join("nvcc");
    let ptx_path = out_dir.join("fuga_kernel.ptx");

    let status = Command::new(&nvcc)
        .arg("-ptx")
        .arg(&format!("-arch=sm_75"))
        .arg("-O3")
        .arg(&kernel_src)
        .arg("-o")
        .arg(&ptx_path)
        .status()
        .expect("Failed to run nvcc. Is CUDA toolkit installed?");

    assert!(status.success(), "nvcc compilation failed");

    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=cuda");
}