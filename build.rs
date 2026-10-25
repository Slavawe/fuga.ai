// build.rs — CPU-совместимая сборка.
// Оригинальный build.rs компилировал native/fuga_kernel.cu через nvcc.
// Сейчас CUDA-путь (src/gpu.rs) не используется для обучения: на Linux/macOS
// линкуется системный cuda.lib; на Windows CUDA toolkit нет, поэтому
// символы driver API дают заглушки cuda_stubs.c (все функции возвращают
// CUDA_ERROR_NO_DEVICE=100 → init() = None → штатный CPU-фолбэк).
fn main() {
    println!("cargo:rerun-if-changed=native/fuga_kernel.cu");
    println!("cargo:rerun-if-changed=cuda_stubs.c");

    #[cfg(windows)]
    {
        cc::Build::new()
            .file("cuda_stubs.c")
            .compile("cuda_stubs");
    }
}
