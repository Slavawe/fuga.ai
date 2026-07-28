use crate::core::wave_cube::WaveCube;
use crate::core::information_triangle::InformationTriangle;
use crate::core::pentagon_storage::PentagonStorage;
use crate::core::fuga_synthesizer::{FugaSynthesizer, FugaResult};
use crate::multi::{
    LanguageId, MultiSyntaxLayer, MultiSyntaxResult,
    MultiSemanticLayer, MultiSemanticResult,
    MultiChaosLayer, MultiChaosResult,
};
use crate::engine::{SourceStats, FugaError};

#[derive(Debug, Clone)]
pub struct MultiEngineResult {
    pub syntax: MultiSyntaxResult,
    pub semantic: MultiSemanticResult,
    pub chaos: MultiChaosResult,
    pub fuga_result: Option<FugaResult>,
    pub source_stats: SourceStats,
    pub language: LanguageId,
}

pub struct MultiEngine {
    dim: usize,
    syntax_layer: MultiSyntaxLayer,
    semantic_layer: MultiSemanticLayer,
    chaos_layer: MultiChaosLayer,
    synthesizer: FugaSynthesizer,
    cube: WaveCube<3, 4>,
    triangle: InformationTriangle,
    pentagon: PentagonStorage,
}

impl MultiEngine {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            syntax_layer: MultiSyntaxLayer::new(dim),
            semantic_layer: MultiSemanticLayer::new(dim),
            chaos_layer: MultiChaosLayer::new(dim),
            synthesizer: FugaSynthesizer::new(dim),
            cube: WaveCube::<3, 4>::new(dim),
            triangle: InformationTriangle::new(dim),
            pentagon: PentagonStorage::new(dim, 0.6),
        }
    }

    pub fn analyze(&mut self, source: &str, lang: LanguageId, file_name: &str) -> MultiEngineResult {
        let syntax = self.syntax_layer.analyze(source, lang, file_name);
        let semantic = self.semantic_layer.analyze(source, lang);
        let chaos = self.chaos_layer.analyze(source, lang);

        let best_attack = chaos.attack_vectors.first().map(|a| a.vector.clone());
        let fuga_result = best_attack.map(|vector| {
            self.synthesizer.synthesize(&mut self.cube, &mut self.triangle, &self.pentagon, &vector)
        });

        self.update_pentagon(&syntax, &semantic, &chaos);

        MultiEngineResult {
            source_stats: SourceStats {
                lines: source.lines().count(),
                chars: source.len(),
                functions: syntax.functions,
            },
            syntax,
            semantic,
            chaos,
            fuga_result,
            language: lang,
        }
    }

    pub fn analyze_file(&mut self, path: &str) -> Result<MultiEngineResult, FugaError> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| FugaError::IoError(e.to_string()))?;
        let lang = LanguageId::from_path(std::path::Path::new(path))
            .ok_or_else(|| FugaError::ParseError(format!("Unsupported file extension: {}", path)))?;
        Ok(self.analyze(&source, lang, path))
    }

    fn update_pentagon(&mut self, syntax: &MultiSyntaxResult, semantic: &MultiSemanticResult, chaos: &MultiChaosResult) {
        if !syntax.violations.is_empty() {
            self.pentagon.store(&format!("syntax_violations_{}", syntax.violations.len()), syntax.violation_vector.clone());
        }
        if !semantic.anomalies.is_empty() {
            self.pentagon.store(&format!("semantic_anomalies_{}", semantic.anomalies.len()), semantic.semantic_vector.clone());
        }
        for attack in &chaos.attack_vectors {
            self.pentagon.store(&format!("chaos_{:?}", attack.pattern), attack.vector.clone());
        }
    }
}
