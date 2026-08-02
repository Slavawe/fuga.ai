use crate::autofix::{FixProposal, FixStrategy};
use crate::core::fuga_synthesizer::BugLocation;
use crate::multi::language::LanguageId;
use crate::multi::patterns::ViolationPattern;
use crate::multi::syntax_layer::MultiSyntaxViolation;

pub struct MultiFixGenerator;

impl MultiFixGenerator {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_fixes(
        &self,
        source: &str,
        violations: &[MultiSyntaxViolation],
        lang: LanguageId,
    ) -> Vec<FixProposal> {
        let mut proposals: Vec<FixProposal> = violations
            .iter()
            .filter_map(|v| self.generate_fix(source, v, lang))
            .collect();
        proposals.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        proposals
    }

    fn generate_fix(
        &self,
        source: &str,
        v: &MultiSyntaxViolation,
        lang: LanguageId,
    ) -> Option<FixProposal> {
        match (v.pattern, lang) {
            // ===== Rust =====
            (ViolationPattern::UnwrapOrExpect, LanguageId::Rust) => self.replace_in_line(
                source,
                v,
                ".unwrap()",
                ".unwrap_or_default()",
                "Replace .unwrap() with .unwrap_or_default()",
                0.85,
            ),
            (ViolationPattern::DivisionByZero, LanguageId::Rust) => self.wrap_div_checked(
                source,
                v,
                ".checked_div(",
                ").unwrap_or(0)",
                "Replace division with checked_div",
                0.7,
            ),
            (ViolationPattern::HardcodedSecret, LanguageId::Rust) => {
                self.replace_secret_with_env(source, v, "std::env::var(\"")
            }

            // ===== C =====
            (ViolationPattern::BufferOverflow, LanguageId::C)
            | (ViolationPattern::BufferOverflow, LanguageId::Cpp) => {
                self.fix_c_buffer_overflow(source, v)
            }
            (ViolationPattern::FormatStringVulnerability, LanguageId::C)
            | (ViolationPattern::FormatStringVulnerability, LanguageId::Cpp) => {
                self.fix_c_format_string(source, v)
            }
            (ViolationPattern::DivisionByZero, LanguageId::C)
            | (ViolationPattern::DivisionByZero, LanguageId::Cpp) => {
                self.wrap_c_div_check(source, v)
            }
            (ViolationPattern::HardcodedSecret, LanguageId::C)
            | (ViolationPattern::HardcodedSecret, LanguageId::Cpp) => {
                self.replace_secret_with_env(source, v, "getenv(\"")
            }

            // ===== Go =====
            (ViolationPattern::SqlInjection, LanguageId::Go) => self.replace_in_line(
                source,
                v,
                ".Exec(",
                ".Exec(\"-- parameterized --\", ",
                "Use parameterized query instead of string concatenation",
                0.8,
            ),
            (ViolationPattern::HardcodedSecret, LanguageId::Go) => {
                self.replace_secret_with_env(source, v, "os.Getenv(\"")
            }

            // ===== Python =====
            (ViolationPattern::SqlInjection, LanguageId::Python) => {
                self.fix_python_sql_injection(source, v)
            }
            (ViolationPattern::CommandInjection, LanguageId::Python) => self.replace_in_line(
                source,
                v,
                "os.system",
                "subprocess.run",
                "Use subprocess.run instead of os.system",
                0.8,
            ),
            (ViolationPattern::FormatStringVulnerability, LanguageId::Python) => self
                .replace_in_line(
                    source,
                    v,
                    ".format",
                    " % ",
                    "Prefer % formatting or f-strings over .format() with user input",
                    0.6,
                ),
            (ViolationPattern::HardcodedSecret, LanguageId::Python) => {
                self.replace_secret_with_env(source, v, "os.environ.get(\"")
            }

            // ===== TypeScript / JavaScript =====
            (ViolationPattern::UnsafeBlock, LanguageId::TypeScript)
            | (ViolationPattern::UnsafeBlock, LanguageId::JavaScript) => self.replace_in_line(
                source,
                v,
                "eval(",
                "JSON.parse(",
                "Replace eval() with JSON.parse() for JSON parsing",
                0.85,
            ),
            (ViolationPattern::SqlInjection, LanguageId::TypeScript)
            | (ViolationPattern::SqlInjection, LanguageId::JavaScript) => {
                self.fix_ts_sql_injection(source, v)
            }
            (ViolationPattern::CommandInjection, LanguageId::TypeScript)
            | (ViolationPattern::CommandInjection, LanguageId::JavaScript) => self.replace_in_line(
                source,
                v,
                ".exec(",
                ".execFile(",
                "Use execFile instead of exec to prevent shell injection",
                0.75,
            ),
            (ViolationPattern::HardcodedSecret, LanguageId::TypeScript)
            | (ViolationPattern::HardcodedSecret, LanguageId::JavaScript) => {
                self.replace_secret_with_env(source, v, "process.env.")
            }

            _ => None,
        }
    }

    fn replace_in_line(
        &self,
        _source: &str,
        v: &MultiSyntaxViolation,
        from: &str,
        to: &str,
        desc: &str,
        confidence: f64,
    ) -> Option<FixProposal> {
        let snippet = v.code_snippet.as_deref()?;
        if !snippet.contains(from) {
            return None;
        }
        let proposed_snippet = snippet.replace(from, to);
        Some(FixProposal {
            location: BugLocation {
                file: Some(v.file.clone()),
                line: Some(v.line),
                column: Some(v.column),
                function: v.function.clone(),
                code_snippet: v.code_snippet.clone(),
            },
            strategy: FixStrategy::ReplaceUnwrapWithDefault,
            original_code: snippet.to_string(),
            proposed_code: proposed_snippet,
            start_byte: Some(v.start_byte),
            end_byte: Some(v.end_byte),
            confidence,
            description: desc.to_string(),
        })
    }

    fn wrap_div_checked(
        &self,
        _source: &str,
        v: &MultiSyntaxViolation,
        wrap_pre: &str,
        wrap_post: &str,
        desc: &str,
        confidence: f64,
    ) -> Option<FixProposal> {
        let snippet = v.code_snippet.as_deref()?;
        let original = snippet;
        let proposed = if let Some(pos) = original.find(" / ") {
            let left = &original[..pos];
            let right = &original[pos + 3..];
            let end = right
                .find(|c: char| c == ' ' || c == ')' || c == ';')
                .unwrap_or(right.len());
            let right_expr = &right[..end];
            format!("{}{}{}{}", left, wrap_pre, right_expr, wrap_post)
        } else {
            return None;
        };
        Some(FixProposal {
            location: BugLocation {
                file: Some(v.file.clone()),
                line: Some(v.line),
                column: Some(v.column),
                function: v.function.clone(),
                code_snippet: v.code_snippet.clone(),
            },
            strategy: FixStrategy::ReplaceDivWithChecked,
            original_code: original.to_string(),
            proposed_code: proposed,
            start_byte: Some(v.start_byte),
            end_byte: Some(v.end_byte),
            confidence,
            description: desc.to_string(),
        })
    }

    fn fix_c_buffer_overflow(
        &self,
        _source: &str,
        v: &MultiSyntaxViolation,
    ) -> Option<FixProposal> {
        let snippet = v.code_snippet.as_deref()?;
        let proposed: Option<String> = if snippet.contains("gets(") {
            let buf = self.extract_arg(snippet, "gets")?;
            Some(format!("fgets({}, sizeof({}), stdin)", buf, buf))
        } else if snippet.contains("strcpy(") {
            let args = self.extract_two_args(snippet, "strcpy")?;
            Some(format!(
                "strncpy({}, {}, sizeof({}))",
                args.0, args.1, args.0
            ))
        } else if snippet.contains("strcat(") {
            let args = self.extract_two_args(snippet, "strcat")?;
            Some(format!(
                "strncat({}, {}, sizeof({}) - strlen({}) - 1)",
                args.0, args.1, args.0, args.0
            ))
        } else if snippet.contains("sprintf(") {
            let open = snippet.find('(')?;
            let rest = &snippet[open + 1..];
            let close = rest.rfind(')')?;
            let first_arg = rest[..close].split(',').next()?.trim();
            let new_call = snippet.replace(
                "sprintf(",
                &format!("snprintf({}, sizeof({}), ", first_arg, first_arg),
            );
            Some(new_call)
        } else if snippet.contains("vsprintf(") {
            let open = snippet.find('(')?;
            let rest = &snippet[open + 1..];
            let close = rest.rfind(')')?;
            let first_arg = rest[..close].split(',').next()?.trim();
            Some(snippet.replace(
                "vsprintf(",
                &format!("vsnprintf({}, sizeof({}), ", first_arg, first_arg),
            ))
        } else if snippet.contains("scanf(")
            || snippet.contains("fscanf(")
            || snippet.contains("sscanf(")
        {
            None // Too complex to auto-fix safely
        } else {
            None
        };
        let proposed = proposed?;
        if proposed == snippet {
            return None;
        }
        Some(FixProposal {
            location: BugLocation {
                file: Some(v.file.clone()),
                line: Some(v.line),
                column: Some(v.column),
                function: v.function.clone(),
                code_snippet: v.code_snippet.clone(),
            },
            strategy: FixStrategy::ReplaceUnwrapWithDefault,
            original_code: snippet.to_string(),
            proposed_code: proposed,
            start_byte: Some(v.start_byte),
            end_byte: Some(v.end_byte),
            confidence: 0.8,
            description: format!(
                "Replace {} with safe alternative",
                snippet.split('(').next().unwrap_or("unsafe function")
            ),
        })
    }

    fn fix_c_format_string(&self, _source: &str, v: &MultiSyntaxViolation) -> Option<FixProposal> {
        let snippet = v.code_snippet.as_deref().unwrap_or("");

        let proposed = if snippet.starts_with("fprintf(") || snippet.starts_with("vfprintf(") {
            // fprintf(stderr, fmt, ...) is safe — only fix fprintf(var) with 2 args
            let open_paren = snippet.find('(')?;
            let rest = &snippet[open_paren + 1..];
            let close = rest.rfind(')')?;
            let args_str = &rest[..close];
            let arg_count = args_str.split(',').count();
            if arg_count == 2 {
                let user_var = args_str.trim();
                snippet
                    .replace(
                        &format!("fprintf({})", user_var),
                        &format!("fprintf(\"%s\", {})", user_var),
                    )
                    .replace(
                        &format!("vfprintf({})", user_var),
                        &format!("vfprintf(\"%s\", {})", user_var),
                    )
            } else {
                return None;
            }
        } else if snippet.starts_with("printf(") {
            let open_paren = snippet.find('(')?;
            let first_arg = &snippet[open_paren + 1..].trim();
            if first_arg.starts_with('"') || first_arg.starts_with('\'') {
                return None; // literal format string — safe
            }
            let end = first_arg
                .find(|c: char| c == ',' || c == ')')
                .unwrap_or(first_arg.len());
            let var_name = &first_arg[..end];
            snippet.replace(
                &format!("printf({}", var_name),
                &format!("printf(\"%s\", {}", var_name),
            )
        } else if snippet.starts_with("sprintf(") || snippet.starts_with("vsprintf(") {
            let open_paren = snippet.find('(')?;
            let rest = &snippet[open_paren + 1..];
            let close = rest.rfind(')')?;
            let args_str = &rest[..close];
            let buf_name = args_str.split(',').next()?.trim();
            let new_func = if snippet.starts_with('v') {
                "vsnprintf"
            } else {
                "snprintf"
            };
            snippet.replace(
                &format!("{}({},", snippet.split('(').next().unwrap(), buf_name),
                &format!("{}({}, sizeof({}),", new_func, buf_name, buf_name),
            )
        } else if snippet.starts_with("snprintf(") || snippet.starts_with("vsnprintf(") {
            return None;
        } else {
            return None;
        };

        if proposed == snippet {
            return None;
        }
        Some(FixProposal {
            location: BugLocation {
                file: Some(v.file.clone()),
                line: Some(v.line),
                column: Some(v.column),
                function: v.function.clone(),
                code_snippet: v.code_snippet.clone(),
            },
            strategy: FixStrategy::ReplaceUnwrapWithDefault,
            original_code: snippet.to_string(),
            proposed_code: proposed,
            start_byte: Some(v.start_byte),
            end_byte: Some(v.end_byte),
            confidence: 0.7,
            description: "Add format string specifier to prevent string format vulnerability"
                .into(),
        })
    }

    fn wrap_c_div_check(&self, _source: &str, v: &MultiSyntaxViolation) -> Option<FixProposal> {
        let snippet = v.code_snippet.as_deref()?;
        let proposed = format!(
            "if (divisor != 0) {{ {} }} else {{ /* handle error */ }}",
            snippet.trim()
        );
        Some(FixProposal {
            location: BugLocation {
                file: Some(v.file.clone()),
                line: Some(v.line),
                column: Some(v.column),
                function: v.function.clone(),
                code_snippet: v.code_snippet.clone(),
            },
            strategy: FixStrategy::ReplaceDivWithChecked,
            original_code: snippet.to_string(),
            proposed_code: proposed,
            start_byte: Some(v.start_byte),
            end_byte: Some(v.end_byte),
            confidence: 0.6,
            description: "Wrap division in zero-check".into(),
        })
    }

    fn fix_python_sql_injection(
        &self,
        _source: &str,
        v: &MultiSyntaxViolation,
    ) -> Option<FixProposal> {
        let snippet = v.code_snippet.as_deref()?;
        let proposed = snippet
            .replace("f\"", "\"")
            .replace(".execute(", ".execute(\"-- parameterized --\", ");
        if proposed == snippet {
            return None;
        }
        Some(FixProposal {
            location: BugLocation {
                file: Some(v.file.clone()),
                line: Some(v.line),
                column: Some(v.column),
                function: v.function.clone(),
                code_snippet: v.code_snippet.clone(),
            },
            strategy: FixStrategy::ReplaceUnwrapWithDefault,
            original_code: snippet.to_string(),
            proposed_code: proposed,
            start_byte: Some(v.start_byte),
            end_byte: Some(v.end_byte),
            confidence: 0.75,
            description: "Use parameterized query instead of f-string interpolation".into(),
        })
    }

    fn fix_ts_sql_injection(&self, _source: &str, v: &MultiSyntaxViolation) -> Option<FixProposal> {
        let snippet = v.code_snippet.as_deref()?;
        let proposed = snippet.replace(".query(", ".query(\"-- parameterized --\", ");
        if proposed == snippet {
            return None;
        }
        Some(FixProposal {
            location: BugLocation {
                file: Some(v.file.clone()),
                line: Some(v.line),
                column: Some(v.column),
                function: v.function.clone(),
                code_snippet: v.code_snippet.clone(),
            },
            strategy: FixStrategy::ReplaceUnwrapWithDefault,
            original_code: snippet.to_string(),
            proposed_code: proposed,
            start_byte: Some(v.start_byte),
            end_byte: Some(v.end_byte),
            confidence: 0.75,
            description: "Use parameterized query instead of string concatenation".into(),
        })
    }

    fn replace_secret_with_env(
        &self,
        _source: &str,
        v: &MultiSyntaxViolation,
        env_func: &str,
    ) -> Option<FixProposal> {
        let snippet = v.code_snippet.as_deref()?;
        let var_name = snippet
            .split('=')
            .next()
            .map(|s| s.trim())
            .unwrap_or("SECRET");
        let proposed = format!("{} = {}{})", var_name, env_func, var_name.to_uppercase());
        Some(FixProposal {
            location: BugLocation {
                file: Some(v.file.clone()),
                line: Some(v.line),
                column: Some(v.column),
                function: v.function.clone(),
                code_snippet: v.code_snippet.clone(),
            },
            strategy: FixStrategy::ReplaceUnwrapWithDefault,
            original_code: snippet.to_string(),
            proposed_code: proposed,
            start_byte: Some(v.start_byte),
            end_byte: Some(v.end_byte),
            confidence: 0.7,
            description: format!(
                "Move hardcoded secret to environment variable {}",
                var_name.to_uppercase()
            ),
        })
    }

    fn extract_arg(&self, s: &str, func: &str) -> Option<String> {
        let prefix = format!("{}(", func);
        let start = s.find(&prefix)?;
        let rest = &s[start + prefix.len()..];
        let end = rest.find(')')?;
        Some(rest[..end].trim().to_string())
    }

    fn extract_two_args(&self, s: &str, func: &str) -> Option<(String, String)> {
        let args = self.extract_arg(s, func)?;
        let comma = args.find(',')?;
        Some((
            args[..comma].trim().to_string(),
            args[comma + 1..].trim().to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi::Severity;

    fn make_violation(
        pattern: ViolationPattern,
        line: usize,
        code: &str,
        file: &str,
    ) -> MultiSyntaxViolation {
        MultiSyntaxViolation {
            pattern,
            severity: Severity::High,
            file: file.to_string(),
            line,
            column: 1,
            start_byte: 0,
            end_byte: code.len(),
            function: None,
            code_snippet: Some(code.to_string()),
            message: format!("{:?} detected", pattern),
        }
    }

    #[test]
    fn test_fix_c_gets() {
        let fixer = MultiFixGenerator::new();
        let source = "int main() { char buf[10]; gets(buf); }";
        let v = make_violation(ViolationPattern::BufferOverflow, 1, "gets(buf)", "test.c");
        let fixes = fixer.generate_fixes(source, &[v], LanguageId::C);
        assert!(!fixes.is_empty());
        assert!(
            fixes[0]
                .proposed_code
                .contains("fgets(buf, sizeof(buf), stdin)")
        );
    }

    #[test]
    fn test_fix_rust_unwrap() {
        let fixer = MultiFixGenerator::new();
        let source = "fn main() { let x = Some(42); x.unwrap(); }";
        let v = make_violation(ViolationPattern::UnwrapOrExpect, 1, "x.unwrap()", "test.rs");
        let fixes = fixer.generate_fixes(source, &[v], LanguageId::Rust);
        assert!(!fixes.is_empty());
        assert!(fixes[0].proposed_code.contains("unwrap_or_default"));
    }

    #[test]
    fn test_fix_ts_eval() {
        let fixer = MultiFixGenerator::new();
        let source = "const x = eval(str);";
        let v = make_violation(ViolationPattern::UnsafeBlock, 1, "eval(str)", "test.ts");
        let fixes = fixer.generate_fixes(source, &[v], LanguageId::TypeScript);
        assert!(!fixes.is_empty());
        assert!(fixes[0].proposed_code.contains("JSON.parse"));
    }

    #[test]
    fn test_fix_python_command() {
        let fixer = MultiFixGenerator::new();
        let source = "os.system('ls -la')";
        let v = make_violation(
            ViolationPattern::CommandInjection,
            1,
            "os.system('ls -la')",
            "test.py",
        );
        let fixes = fixer.generate_fixes(source, &[v], LanguageId::Python);
        assert!(!fixes.is_empty());
        assert!(fixes[0].proposed_code.contains("subprocess.run"));
    }
}
