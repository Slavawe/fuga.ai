pub mod topology;

use crate::core::hypervector::Hypervector;

pub trait VsaSpace {
    fn bind(&self, rhs: &Self) -> Self;
    fn unbind(&self, rhs: &Self) -> Self;
    fn similarity(&self, rhs: &Self) -> f64;
}

impl VsaSpace for Hypervector {
    fn bind(&self, rhs: &Self) -> Self {
        self.bind(rhs)
    }
    fn unbind(&self, rhs: &Self) -> Self {
            topology::ls_unbind(self, rhs, 32)
    }
    fn similarity(&self, rhs: &Self) -> f64 {
        let dim = self.dim as f64;
        self.words.iter().zip(rhs.words.iter()).flat_map(|(&wa, &wb)| {
            (0..64).map(move |bi| {
                let ai = if (wa >> bi) & 1 == 1 { 1.0 } else { -1.0 };
                let bi = if (wb >> bi) & 1 == 1 { 1.0 } else { -1.0 };
                ai * bi
            })
        }).sum::<f64>() / dim
    }
}
