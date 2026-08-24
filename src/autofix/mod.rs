pub mod fix_generator;
pub mod fix_validator;
pub mod patch_generator;

use crate::core::fuga_synthesizer::BugLocation;
use crate::core::hypervector::Hypervector;

/// Стратегия исправления бага
#[derive(Debug, Clone)]
pub enum FixStrategy {
    /// unwrap() → unwrap_or_default()
    ReplaceUnwrapWithDefault,
    /// unwrap() → expect("msg")
    ReplaceUnwrapWithExpect(String),
    /// unsafe { ... } → обёртка с проверкой
    WrapUnsafeWithCheck,
    /// a / b → a.checked_div(b).unwrap_or(0)
    ReplaceDivWithChecked,
    /// Мутация через VSA-куб (экспериментальная)
    VsaMutation(Hypervector),
}

/// Предложение исправления
#[derive(Debug, Clone)]
pub struct FixProposal {
    /// Локация бага
    pub location: BugLocation,
    /// Стратегия фикса
    pub strategy: FixStrategy,
    /// Оригинальный код
    pub original_code: String,
    /// Предложенное исправление
    pub proposed_code: String,
    /// Byte range in source (from tree-sitter, for precise patches)
    pub start_byte: Option<usize>,
    pub end_byte: Option<usize>,
    /// Confidence (0..1): насколько фикс надёжен
    pub confidence: f64,
    /// Описание фикса
    pub description: String,
}

/// Unified diff патч
#[derive(Debug, Clone)]
pub struct UnifiedDiff {
    pub file_path: String,
    pub original_lines: Vec<String>,
    pub patched_lines: Vec<String>,
    pub diff_text: String,
}

pub use fix_generator::FixGenerator;
pub use fix_validator::FixValidator;
pub use patch_generator::PatchGenerator;
