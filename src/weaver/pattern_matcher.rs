use super::super_token::TokenRole;

#[derive(Debug, Clone)]
pub struct WindowBoundary {
    pub start: usize,
    pub end: usize,
    pub role: TokenRole,
}

pub struct PatternMatcher {
    window_size: usize,
}

impl PatternMatcher {
    pub fn new(window_size: usize) -> Self {
        Self { window_size: window_size.max(2).min(16) }
    }

    pub fn window_size(&self) -> usize { self.window_size }

    pub fn find_boundaries(&self, stream: &[TokenInfo]) -> Vec<WindowBoundary> {
        let mut boundaries = Vec::new();
        if stream.is_empty() { return boundaries; }

        let mut i = 0;
        while i < stream.len() {
            let role = self.detect_role(&stream[i..]);
            let window_end = if i + self.window_size <= stream.len() {
                let natural_break = self.find_natural_break(&stream[i..], self.window_size);
                i + natural_break
            } else {
                stream.len()
            };
            boundaries.push(WindowBoundary {
                start: i,
                end: window_end,
                role,
            });
            i = window_end;
        }
        boundaries
    }

    fn find_natural_break(&self, window: &[TokenInfo], max_size: usize) -> usize {
        let mut depth: i32 = 0;
        for (j, tok) in window.iter().enumerate() {
            if j >= max_size { return j; }
            match tok.text.as_str() {
                "{" | "[" | "(" => depth += 1,
                "}" | "]" | ")" => {
                    depth -= 1;
                    if depth < 0 { return j + 1; }
                }
                _ => {}
            }
            if depth == 0 && j >= 2 {
                if tok.text == " " || tok.text == "\n" || tok.text == "\t" {
                    return j + 1;
                }
            }
        }
        window.len().min(max_size)
    }

    fn detect_role(&self, window: &[TokenInfo]) -> TokenRole {
        let text: String = window.iter().take(4).map(|t| t.text.as_str()).collect();
        let trimmed = text.trim();
        if trimmed.starts_with('{') || trimmed.starts_with('[') || text.contains("\"") {
            return TokenRole::JSON;
        }
        if text.contains("fn ") || text.contains("let ") || text.contains("=>")
            || text.contains("def ") || text.contains("class ")
            || text.contains("int ") || text.contains("void ")
        {
            return TokenRole::CODE;
        }
        if text.contains(['+', '-', '*', '/', '=', '>', '<'].as_slice()) {
            return TokenRole::MATH;
        }
        if window.iter().any(|t| t.text.len() > 1 && t.text.chars().all(|c| c.is_alphanumeric() || c == '_')) {
            return TokenRole::NATURAL_LANGUAGE;
        }
        TokenRole::empty()
    }
}

#[derive(Debug, Clone)]
pub struct TokenInfo {
    pub id: u32,
    pub text: String,
}
