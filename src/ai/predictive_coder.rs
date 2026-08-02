use super::sdr::SdrVector;

#[derive(Clone, Debug)]
pub struct PredictiveCoder {
    pub expected_sdr: Option<SdrVector>,
    pub baseline_error: f32,
}

impl PredictiveCoder {
    pub fn new() -> Self {
        Self {
            expected_sdr: None,
            baseline_error: 0.0,
        }
    }

    pub fn compute_error(&self, predicted: &SdrVector, expected: &SdrVector) -> f32 {
        let denom = predicted.popcount().max(expected.popcount());
        if denom == 0 {
            return 0.0;
        }
        1.0 - predicted.overlap(expected) as f32 / denom as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_sparse_vectors_have_zero_error() {
        let v = super::super::sdr::encode_text("main");
        assert_eq!(PredictiveCoder::new().compute_error(&v, &v), 0.0);
    }

    #[test]
    fn disjoint_sparse_vectors_have_maximum_error() {
        let a = super::super::sdr::encode_text("main");
        let b = super::super::sdr::encode_text("completely_unrelated_token");
        assert!(PredictiveCoder::new().compute_error(&a, &b) > 0.9);
    }
}
