use super::{FixStrategy, FixProposal};
use crate::core::fuga_synthesizer::{BugLocation, FugaResult};
use crate::core::wave_cube::WaveCube;
use crate::core::hypervector::Hypervector;
use crate::layers::{SyntaxViolation, ViolationKind};

pub struct FixGenerator {
    cube: WaveCube<3, 4>,
    dim: usize,
}

impl FixGenerator {
    pub fn new(dim: usize) -> Self {
        Self {
            cube: WaveCube::<3, 4>::new(dim),
            dim,
        }
    }

    pub fn generate_fixes(
        &mut self,
        source: &str,
        fuga_result: &FugaResult,
        violations: &[SyntaxViolation],
    ) -> Vec<FixProposal> {
        let mut proposals = Vec::new();

        for violation in violations {
            if let Some(proposal) = self.generate_fix_for_violation(source, violation) {
                proposals.push(proposal);
            }
        }

        if fuga_result.bug_detected {
            if let Some(bug_vec) = &fuga_result.bug_vector {
                if let Some(proposal) = self.generate_vsa_mutation(source, bug_vec, fuga_result) {
                    proposals.push(proposal);
                }
            }
        }

        proposals.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        proposals
    }

    fn generate_fix_for_violation(&self, source: &str, violation: &SyntaxViolation) -> Option<FixProposal> {
        match violation.kind {
            ViolationKind::UnwrapExpect => self.fix_unwrap(source, violation),
            ViolationKind::UnsafeBlock => self.fix_unsafe(source, violation),
            ViolationKind::DivisionByZero => self.fix_division(source, violation),
            _ => None,
        }
    }

    fn fix_unwrap(&self, source: &str, violation: &SyntaxViolation) -> Option<FixProposal> {
        let lines: Vec<&str> = source.lines().collect();
        let mut found_line = None;
        for (i, line) in lines.iter().enumerate() {
            if line.contains(".unwrap()") {
                found_line = Some(i);
                break;
            }
        }

        let line_num = found_line?;
        let original_line = lines[line_num];
        let proposed_line = original_line.replace(".unwrap()", ".unwrap_or_default()");

        Some(FixProposal {
            location: BugLocation {
                file: None,
                line: Some(line_num + 1),
                column: None,
                function: Some(violation.location.clone()),
                code_snippet: Some(original_line.trim().to_string()),
            },
            strategy: FixStrategy::ReplaceUnwrapWithDefault,
            original_code: original_line.to_string(),
            proposed_code: proposed_line,
            start_byte: None,
            end_byte: None,
            confidence: 0.85,
            description: "Replace .unwrap() with .unwrap_or_default() to avoid panic".into(),
        })
    }

    fn fix_unsafe(&self, _source: &str, violation: &SyntaxViolation) -> Option<FixProposal> {
        Some(FixProposal {
            location: BugLocation {
                file: None,
                line: None,
                column: None,
                function: Some(violation.location.clone()),
                code_snippet: None,
            },
            strategy: FixStrategy::WrapUnsafeWithCheck,
            original_code: "unsafe { ... }".into(),
            proposed_code: "// Consider safe alternative or add runtime checks".into(),
            start_byte: None,
            end_byte: None,
            confidence: 0.5,
            description: "Unsafe block detected — manual review recommended".into(),
        })
    }

    fn fix_division(&self, source: &str, violation: &SyntaxViolation) -> Option<FixProposal> {
        let lines: Vec<&str> = source.lines().collect();
        let mut found_line = None;
        for (i, line) in lines.iter().enumerate() {
            if line.contains('/') && !line.contains("//") {
                found_line = Some(i);
                break;
            }
        }

        let line_num = found_line?;
        let original_line = lines[line_num];
        let proposed_line = original_line.replace(" / ", ".checked_div(").replace(';', ").unwrap_or(0);");

        Some(FixProposal {
            location: BugLocation {
                file: None,
                line: Some(line_num + 1),
                column: None,
                function: Some(violation.location.clone()),
                code_snippet: Some(original_line.trim().to_string()),
            },
            strategy: FixStrategy::ReplaceDivWithChecked,
            original_code: original_line.to_string(),
            proposed_code: proposed_line,
            start_byte: None,
            end_byte: None,
            confidence: 0.7,
            description: "Replace division with checked_div to avoid panic".into(),
        })
    }

    fn generate_vsa_mutation(&mut self, _source: &str, bug_vec: &Hypervector, fuga_result: &FugaResult) -> Option<FixProposal> {
        let anti_bug = self.invert_bug_vector(bug_vec);
        Some(FixProposal {
            location: fuga_result.bug_location.clone()?,
            strategy: FixStrategy::VsaMutation(anti_bug.clone()),
            original_code: "/* VSA-detected pattern */".into(),
            proposed_code: "/* VSA-proposed mutation */".into(),
            start_byte: None,
            end_byte: None,
            confidence: 0.6,
            description: format!("VSA-based mutation (experimental) — {}", fuga_result.counterpoint_description),
        })
    }

    fn invert_bug_vector(&mut self, bug_vec: &Hypervector) -> Hypervector {
        let mut words = bug_vec.words.clone();
        for w in &mut words {
            *w = !*w;
        }
        let rem = self.dim % 64;
        if rem != 0 {
            let last = words.len() - 1;
            words[last] &= (1u64 << rem) - 1;
        }
        Hypervector { dim: self.dim, words }
    }
}
