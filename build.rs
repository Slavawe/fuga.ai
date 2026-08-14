// build.rs — заглушка для CPU-тестирования (CUDA недоступна)
// Оригинальный build.rs компилирует native/fuga_kernel.cu через nvcc.
// Для тестов HybridCore на CPU CUDA не нужна.
fn main() {
    println!("cargo:rerun-if-changed=native/fuga_kernel.cu");
    // CUDA-компиляция пропущена (нет nvcc/CUDA toolkit)
}
