use super::FixProposal;
use crate::engine::FugaEngine;

/// Валидатор фиксов — применяет патч и проверяет что баг исчез
pub struct FixValidator;

impl FixValidator {
    pub fn new() -> Self {
        Self
    }

    /// Валидирует предложенный фикс: применяет его и прогоняет через FugaEngine
    pub fn validate_fix(&self, original_source: &str, proposal: &FixProposal, engine: &mut FugaEngine) -> ValidationResult {
        // Применяем патч
        let patched_source = self.apply_proposal(original_source, proposal);

        // Анализируем патченый код
        let result = match engine.analyze(&patched_source) {
            Ok(r) => r,
            Err(e) => {
                return ValidationResult {
                    valid: false,
                    reason: format!("Failed to analyze patched code: {}", e),
                    safety_score_before: 0.0,
                    safety_score_after: 0.0,
                    bug_score_before: 0.0,
                    bug_score_after: 0.0,
                };
            }
        };

        // Получаем метрики до патча (предполагаем что они есть)
        let original_result = engine.analyze(original_source).unwrap();

        let safety_before = original_result.layer_results.syntax.safety_score;
        let safety_after = result.layer_results.syntax.safety_score;

        let bug_before = original_result.fuga_result.as_ref().map(|f| f.confidence).unwrap_or(0.0);
        let bug_after = result.fuga_result.as_ref().map(|f| f.confidence).unwrap_or(0.0);

        // Фикс валиден если:
        // 1. Safety score вырос или остался на том же уровне
        // 2. Bug score упал (меньше багов)
        // 3. Код всё ещё парсится
        let valid = safety_after >= safety_before && bug_after < bug_before;

        ValidationResult {
            valid,
            reason: if valid {
                "Fix improves safety and reduces bugs".into()
            } else {
                format!("Fix did not improve metrics: safety {:.2}→{:.2}, bugs {:.2}→{:.2}",
                    safety_before, safety_after, bug_before, bug_after)
            },
            safety_score_before: safety_before,
            safety_score_after: safety_after,
            bug_score_before: bug_before,
            bug_score_after: bug_after,
        }
    }

    fn apply_proposal(&self, source: &str, proposal: &FixProposal) -> String {
        // Простая замена построчно
        if let Some(line_num) = proposal.location.line {
            let lines: Vec<&str> = source.lines().collect();
            if line_num > 0 && line_num <= lines.len() {
                let mut patched_lines = lines.clone();
                patched_lines[line_num - 1] = &proposal.proposed_code;
                return patched_lines.join("\n");
            }
        }
        // Fallback: глобальная замена
        source.replace(&proposal.original_code, &proposal.proposed_code)
    }
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub reason: String,
    pub safety_score_before: f64,
    pub safety_score_after: f64,
    pub bug_score_before: f64,
    pub bug_score_after: f64,
}