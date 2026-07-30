#[derive(Debug, Clone)]
pub struct CppChunk {
    pub symbol_name: String,
    pub content: String,
    pub is_simd_block: bool,
}

pub fn parse_cpp_file(source: &str) -> Vec<CppChunk> {
    let mut chunks = Vec::new();
    for block in source.split("\n\n") {
        let trimmed = block.trim();
        if trimmed.is_empty() { continue; }
        let is_simd = trimmed.contains("AVX")
            || trimmed.contains("NEON")
            || trimmed.contains("_mm")
            || trimmed.contains("__m256")
            || trimmed.contains("__m128")
            || trimmed.contains("vld1q")
            || trimmed.contains("vmlal");
        let first_line = trimmed.lines().next().unwrap_or("");
        let symbol_name = first_line
            .chars()
            .take(40)
            .collect::<String>()
            .trim()
            .to_string();
        chunks.push(CppChunk {
            symbol_name,
            content: trimmed.to_string(),
            is_simd_block: is_simd,
        });
    }
    chunks
}

pub fn scan_bitnet_kernel(source: &str) -> Vec<CppChunk> {
    let all = parse_cpp_file(source);
    all.into_iter().filter(|c| c.is_simd_block).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty() {
        let chunks = parse_cpp_file("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_parse_simple() {
        let src = "int add(int a, int b) {\n  return a + b;\n}\n\nint mul(int a, int b) {\n  return a * b;\n}";
        let chunks = parse_cpp_file(src);
        assert_eq!(chunks.len(), 2);
        assert!(!chunks[0].is_simd_block);
    }

    #[test]
    fn test_detect_simd_avx() {
        let src = "void bitnet_gemm_avx2() {\n  __m256i a = _mm256_loadu_si256(ptr);\n}";
        let chunks = parse_cpp_file(src);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_simd_block);
    }

    #[test]
    fn test_detect_simd_neon() {
        let src = "void bitnet_gemm_neon() {\n  int8x16_t a = vld1q_s8(ptr);\n}";
        let chunks = parse_cpp_file(src);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_simd_block);
    }

    #[test]
    fn test_scan_bitnet() {
        let src = "void normal() { int x = 1; }\n\nvoid bitnet_gemm_avx2() { __m256i a; }\n\nvoid bitnet_gemm_neon() { int8x16_t a; vmlal_s8(x, y, z); }";
        let simd = scan_bitnet_kernel(src);
        assert_eq!(simd.len(), 2);
        assert!(simd[0].symbol_name.contains("bitnet_gemm_avx2"));
        assert!(simd[1].symbol_name.contains("bitnet_gemm_neon"));
    }
}
