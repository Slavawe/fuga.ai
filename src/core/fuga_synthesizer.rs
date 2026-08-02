use super::hypervector::Hypervector;
use super::information_triangle::InformationTriangle;
use super::pentagon_storage::PentagonStorage;
use super::wave_cube::WaveCube;

#[derive(Debug, Clone)]
pub struct FugaResult {
    pub bug_detected: bool,
    pub confidence: f64,
    pub counterpoint_description: String,
    pub bug_vector: Option<Hypervector>,
    pub bug_location: Option<BugLocation>,
}

#[derive(Debug, Clone)]
pub struct BugLocation {
    pub file: Option<String>,
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub function: Option<String>,
    pub code_snippet: Option<String>,
}

pub struct FugaSynthesizer {
    pub dim: usize,
}

impl FugaSynthesizer {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn synthesize<const N: usize, const S: usize>(
        &self,
        cube: &mut WaveCube<N, S>,
        triangle: &mut InformationTriangle,
        pentagon: &PentagonStorage,
        chaos_attack: &Hypervector,
    ) -> FugaResult {
        triangle.ingest_pattern(chaos_attack);

        let (syn, sem, chaos) = triangle.emit_to_cube();
        cube.absorb_from_triangle(syn, sem, chaos);

        let entropy = cube.global_entropy();
        let coherence = cube.coherence();

        if entropy < 0.3 || coherence < 0.4 {
            let diagonal_queries: Vec<Hypervector> = (0..S).map(|i| cube.cell(i, i, i)).collect();
            let query_refs: Vec<&Hypervector> = diagonal_queries.iter().collect();
            let results = pentagon.batch_fetch(&query_refs);

            if !results.is_empty() {
                let replenished = Hypervector::random(self.dim);
                cube.write_cell(S / 2, S / 2, S / 2, &replenished);
                cube.wave_flow_x(1);
                cube.wave_flow_y(1);
                cube.wave_flow_z(1);
            }
        }

        let counterpoint = triangle.counterpoint_intensity();

        let syntax_safety = triangle.vertex_syntax.similarity(chaos_attack);
        let semantics_breakdown = 1.0 - triangle.vertex_semantics.similarity(chaos_attack);

        let not_safety = 1.0 - syntax_safety;

        let bug_score = counterpoint * not_safety * semantics_breakdown;

        let bug_detected = bug_score > 0.5 && counterpoint > 0.3;

        let description = if bug_detected {
            format!(
                "BUG: counterpoint={:.3}, not_safety={:.3}, logic_breakdown={:.3}, score={:.3}",
                counterpoint, not_safety, semantics_breakdown, bug_score
            )
        } else {
            format!(
                "CLEAN: counterpoint={:.3}, not_safety={:.3}, logic_breakdown={:.3}, score={:.3}",
                counterpoint, not_safety, semantics_breakdown, bug_score
            )
        };

        FugaResult {
            bug_detected,
            confidence: bug_score,
            counterpoint_description: description,
            bug_vector: if bug_detected {
                Some(chaos_attack.clone())
            } else {
                None
            },
            bug_location: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_env() -> (
        WaveCube<3, 4>,
        InformationTriangle,
        PentagonStorage,
        FugaSynthesizer,
    ) {
        let dim = 16384;
        let cube = WaveCube::<3, 4>::new(dim);
        let triangle = InformationTriangle::new(dim);
        let mut pentagon = PentagonStorage::new(dim, 0.6);
        pentagon.store("overflow", Hypervector::random(dim));
        pentagon.store("race_condition", Hypervector::random(dim));
        let synth = FugaSynthesizer::new(dim);
        (cube, triangle, pentagon, synth)
    }

    #[test]
    fn test_synthesize_random_attack() {
        let (mut cube, mut triangle, pentagon, synth) = make_test_env();
        let attack = Hypervector::random(synth.dim);
        let result = synth.synthesize(&mut cube, &mut triangle, &pentagon, &attack);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_synthesize_known_bug_pattern() {
        let dim = 65536;
        let cube = WaveCube::<3, 4>::new(dim);
        let mut triangle = InformationTriangle::new(dim);
        let mut pentagon = PentagonStorage::new(dim, 0.6);
        pentagon.store("overflow", Hypervector::random(dim));
        pentagon.store("race_condition", Hypervector::random(dim));
        let synth = FugaSynthesizer::new(dim);

        let chaos_inject = Hypervector::random(dim);
        triangle.vertex_chaos = triangle.vertex_chaos.bundle(&[&chaos_inject]);
        let result = synth.synthesize(&mut cube.clone(), &mut triangle, &pentagon, &chaos_inject);
        assert!(
            result.confidence > 0.0,
            "Constructed attack should produce non-zero signal"
        );
    }
}
