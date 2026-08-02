use super::{Action, Horseman, identity};
use crate::ai::htm_temporal::TemporalMemory;
use crate::ai::sdr::{SdrVector, encode_text};
use tree_sitter::Parser;

pub struct Vlad {
    identity: SdrVector,
}

impl Vlad {
    pub fn new() -> Self {
        Self {
            identity: identity("Влад строгий синтаксис ремесло"),
        }
    }

    /// Select and materialize one grammar-approved body slot using TM scores.
    pub fn fill_body_slot(tm: &TemporalMemory, context_sdr: &SdrVector) -> String {
        let predicted = tm.predict_next(context_sdr);
        let mut best = "}";
        let mut best_score = 0u32;
        for token in ["let", "println!", "}"] {
            let score = predicted.overlap(&encode_text(token));
            if score > best_score {
                best_score = score;
                best = token;
            }
        }
        match best {
            "let" => "let _ = 0; ".to_string(),
            "println!" => "println!(); ".to_string(),
            _ => String::new(),
        }
    }

    /// Build a complete function by letting TM fill one grammar-approved body slot.
    pub fn generate_ast_with_tm(&self, tm: &TemporalMemory, prompt: &str) -> Action {
        let name = prompt
            .split_whitespace()
            .nth(1)
            .filter(|name| name.chars().all(|c| c == '_' || c.is_ascii_alphanumeric()))
            .unwrap_or("main");
        let context = format!("fn {name}() {{");
        let body = Self::fill_body_slot(tm, &encode_text(&context));
        let code = if body.is_empty() {
            format!("fn {name}() {{}}")
        } else {
            format!("fn {name}() {{ {body}}}")
        };
        if Self::validate_rust(&code) {
            Action::GenerateCode(code)
        } else {
            Action::GenerateCode(format!("fn {name}() {{}}"))
        }
    }

    fn validate_rust(code: &str) -> bool {
        let mut parser = Parser::new();
        if parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .is_err()
        {
            return false;
        }
        parser
            .parse(code, None)
            .map(|tree| !tree.root_node().has_error())
            .unwrap_or(false)
    }

    /// Build a minimal syntactically complete Rust function. This is the
    /// structural L1 stage; TM slot filling can be applied to the body.
    pub fn generate_ast_skeleton(&self, prompt: &str) -> Option<String> {
        let name = prompt
            .split_whitespace()
            .skip(2)
            .next()
            .filter(|name| name.chars().all(|c| c == '_' || c.is_ascii_alphanumeric()))
            .unwrap_or("main");
        let candidate = format!("fn {name}() {{}} ");
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .ok()?;
        let tree = parser.parse(&candidate, None)?;
        if tree.root_node().has_error() {
            None
        } else {
            Some(candidate)
        }
    }
}

impl Horseman for Vlad {
    fn name(&self) -> &str {
        "Влад"
    }
    fn identity_sdr(&self) -> &SdrVector {
        &self.identity
    }
    fn process_prompt(&mut self, prompt: &str) -> Action {
        Action::GenerateCode(self.generate_ast_skeleton(prompt).unwrap_or_default())
    }
    fn speak(&self, message: &str) -> String {
        format!("[{}]: {}", self.name(), message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vlad_builds_valid_main_skeleton() {
        let code = Vlad::new().generate_ast_skeleton("fn main").unwrap();
        assert_eq!(code, "fn main() {} ");
    }

    #[test]
    fn vlad_rejects_invalid_function_name() {
        let code = Vlad::new().generate_ast_skeleton("fn bad-name").unwrap();
        assert_eq!(code, "fn main() {} ");
    }

    #[test]
    fn empty_tm_falls_back_to_closing_body_slot() {
        let tm = TemporalMemory::new(32, 4);
        let body = Vlad::fill_body_slot(&tm, &encode_text("fn main {"));
        assert!(body.is_empty());
    }

    #[test]
    fn generate_ast_with_empty_tm_returns_valid_main_action() {
        let tm = TemporalMemory::new(32, 4);
        let action = Vlad::new().generate_ast_with_tm(&tm, "fn main");
        assert_eq!(action, Action::GenerateCode("fn main() {}".into()));
    }
}
