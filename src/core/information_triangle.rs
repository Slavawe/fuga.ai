use super::hypervector::Hypervector;

pub struct InformationTriangle {
    pub dim: usize,
    pub vertex_syntax: Hypervector,
    pub vertex_semantics: Hypervector,
    pub vertex_chaos: Hypervector,
    pub accumulator: Hypervector,
    pub accumulator_count: u64,
}

impl InformationTriangle {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vertex_syntax: Hypervector::random(dim),
            vertex_semantics: Hypervector::random(dim),
            vertex_chaos: Hypervector::random(dim),
            accumulator: Hypervector::random(dim),
            accumulator_count: 1,
        }
    }

    pub fn ingest_pattern(&mut self, pattern: &Hypervector) {
        let syn_sim = self.vertex_syntax.similarity(pattern);
        let sem_sim = self.vertex_semantics.similarity(pattern);
        let chaos_sim = self.vertex_chaos.similarity(pattern);

        if syn_sim >= sem_sim && syn_sim >= chaos_sim {
            self.vertex_syntax = self.vertex_syntax.bundle(&[pattern]);
        } else if sem_sim >= chaos_sim {
            self.vertex_semantics = self.vertex_semantics.bundle(&[pattern]);
        } else {
            self.vertex_chaos = self.vertex_chaos.bundle(&[pattern]);
        }

        let total = self.accumulator_count + 1;
        let weight_old = self.accumulator_count as f64 / total as f64;
        let weight_new = 1.0 / total as f64;

        let wc = (self.dim + 63) / 64;
        let mut sum = vec![0.0f64; self.dim];
        for i in 0..self.dim {
            let acc_bit = (self.accumulator.words[i / 64] >> (i % 64)) & 1;
            let pat_bit = (pattern.words[i / 64] >> (i % 64)) & 1;
            sum[i] = weight_old * if acc_bit == 1 { 1.0 } else { -1.0 }
                + weight_new * if pat_bit == 1 { 1.0 } else { -1.0 };
        }
        let mut words = vec![0u64; wc];
        for (i, &v) in sum.iter().enumerate() {
            if v >= 0.0 {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        self.accumulator = Hypervector {
            dim: self.dim,
            words,
        };
        self.accumulator_count = total;
    }

    pub fn emit_to_cube(&self) -> (&Hypervector, &Hypervector, &Hypervector) {
        (
            &self.vertex_syntax,
            &self.vertex_semantics,
            &self.vertex_chaos,
        )
    }

    pub fn triangle_coherence(&self) -> f64 {
        let ab = self.vertex_syntax.similarity(&self.vertex_semantics);
        let ac = self.vertex_syntax.similarity(&self.vertex_chaos);
        let bc = self.vertex_semantics.similarity(&self.vertex_chaos);
        (ab + ac + bc) / 3.0
    }

    pub fn counterpoint_intensity(&self) -> f64 {
        1.0 - self.triangle_coherence()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_ingestion_classifies() {
        let dim = 10000;
        let mut triangle = InformationTriangle::new(dim);
        let pattern = Hypervector::random(dim);
        let before = triangle.triangle_coherence();
        triangle.ingest_pattern(&pattern);
        let after = triangle.triangle_coherence();
        assert!(before != after);
    }

    #[test]
    fn test_emit_to_cube_returns_valid_vectors() {
        let dim = 1024;
        let triangle = InformationTriangle::new(dim);
        let (syn, sem, chaos) = triangle.emit_to_cube();
        assert_eq!(syn.dim, dim);
        assert_eq!(sem.dim, dim);
        assert_eq!(chaos.dim, dim);
    }
}
