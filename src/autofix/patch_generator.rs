use super::{FixProposal, UnifiedDiff};

/// Генератор unified diff патчей
pub struct PatchGenerator;

impl PatchGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Генерирует unified diff для списка предложений
    pub fn generate_patch(
        &self,
        file_path: &str,
        original_source: &str,
        proposals: &[FixProposal],
    ) -> UnifiedDiff {
        let has_byte_level = proposals.iter().any(|p| p.start_byte.is_some());

        if has_byte_level {
            self.generate_byte_level_patch(file_path, original_source, proposals)
        } else {
            self.generate_line_level_patch(file_path, original_source, proposals)
        }
    }

    fn generate_byte_level_patch(
        &self,
        file_path: &str,
        original_source: &str,
        proposals: &[FixProposal],
    ) -> UnifiedDiff {
        let original_lines: Vec<String> = original_source.lines().map(|l| l.to_string()).collect();

        // Sort proposals by start_byte descending to apply right-to-left (preserve offsets)
        let mut sorted: Vec<&FixProposal> = proposals
            .iter()
            .filter(|p| p.start_byte.is_some() && p.end_byte.is_some())
            .collect();
        sorted.sort_by(|a, b| b.start_byte.unwrap().cmp(&a.start_byte.unwrap()));

        let mut modified = original_source.to_string();
        for p in &sorted {
            let start = p.start_byte.unwrap();
            let end = p.end_byte.unwrap();
            if start <= end && end <= modified.len() {
                modified.replace_range(start..end, &p.proposed_code);
            }
        }

        let patched_lines: Vec<String> = modified.lines().map(|l| l.to_string()).collect();
        let diff_text = self.format_unified_diff(file_path, &original_lines, &patched_lines);

        UnifiedDiff {
            file_path: file_path.to_string(),
            original_lines,
            patched_lines,
            diff_text,
        }
    }

    fn generate_line_level_patch(
        &self,
        file_path: &str,
        original_source: &str,
        proposals: &[FixProposal],
    ) -> UnifiedDiff {
        let original_lines: Vec<String> = original_source.lines().map(|l| l.to_string()).collect();
        let mut patched_lines = original_lines.clone();

        // Применяем все предложения
        for proposal in proposals {
            if let Some(line_num) = proposal.location.line {
                if line_num > 0 && line_num <= patched_lines.len() {
                    patched_lines[line_num - 1] = proposal.proposed_code.clone();
                }
            }
        }

        let diff_text = self.format_unified_diff(file_path, &original_lines, &patched_lines);

        UnifiedDiff {
            file_path: file_path.to_string(),
            original_lines,
            patched_lines,
            diff_text,
        }
    }

    fn format_unified_diff(
        &self,
        file_path: &str,
        original: &[String],
        patched: &[String],
    ) -> String {
        let mut diff = String::new();
        diff.push_str(&format!("--- a/{}\n", file_path));
        diff.push_str(&format!("+++ b/{}\n", file_path));

        // Простейшая имплементация: показываем все изменённые строки
        // ponytail: полноценный алгоритм diff (Myers, histogram)
        let mut hunk_start = 0;
        let mut hunk_changes = Vec::new();

        for (i, (orig, patch)) in original.iter().zip(patched.iter()).enumerate() {
            if orig != patch {
                if hunk_changes.is_empty() {
                    hunk_start = i;
                }
                hunk_changes.push((i, orig.clone(), patch.clone()));
            }
        }

        if !hunk_changes.is_empty() {
            let context_before = hunk_start.saturating_sub(3);
            let context_after = (hunk_changes.last().unwrap().0 + 3).min(original.len() - 1);
            let orig_line_count = context_after - context_before + 1;
            let patch_line_count = orig_line_count;

            diff.push_str(&format!(
                "@@ -{},{} +{},{} @@\n",
                context_before + 1,
                orig_line_count,
                context_before + 1,
                patch_line_count
            ));

            // Контекст до
            for i in context_before..hunk_start {
                diff.push_str(&format!(" {}\n", original[i]));
            }

            // Изменения
            for (_i, orig, patch) in &hunk_changes {
                diff.push_str(&format!("-{}\n", orig));
                diff.push_str(&format!("+{}\n", patch));
            }

            // Контекст после
            let last_change = hunk_changes.last().unwrap().0;
            for i in (last_change + 1)..=context_after.min(original.len() - 1) {
                diff.push_str(&format!(" {}\n", original[i]));
            }
        }

        diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autofix::{FixProposal, FixStrategy};
    use crate::core::fuga_synthesizer::BugLocation;

    #[test]
    fn test_patch_generation() {
        let patch_gen = PatchGenerator::new();
        let source = "fn main() {\n    let x = Some(42);\n    x.unwrap();\n}";
        let proposal = FixProposal {
            location: BugLocation {
                file: None,
                line: Some(3),
                column: None,
                function: Some("main".into()),
                code_snippet: Some("x.unwrap();".into()),
            },
            strategy: FixStrategy::ReplaceUnwrapWithDefault,
            original_code: "    x.unwrap();".into(),
            proposed_code: "    x.unwrap_or_default();".into(),
            start_byte: None,
            end_byte: None,
            confidence: 0.85,
            description: "Replace unwrap with unwrap_or_default".into(),
        };

        let diff = patch_gen.generate_patch("test.rs", source, &[proposal]);
        assert!(diff.diff_text.contains("--- a/test.rs"));
        assert!(diff.diff_text.contains("+++ b/test.rs"));
        assert!(diff.diff_text.contains("-    x.unwrap();"));
        assert!(diff.diff_text.contains("+    x.unwrap_or_default();"));
    }
}
