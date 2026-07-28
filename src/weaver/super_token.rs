use crate::core::hypervector::Hypervector;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenRole(u8);

impl TokenRole {
    pub const NATURAL_LANGUAGE: TokenRole = TokenRole(0b0000_0001);
    pub const CODE:             TokenRole = TokenRole(0b0000_0010);
    pub const JSON:             TokenRole = TokenRole(0b0000_0100);
    pub const MATH:             TokenRole = TokenRole(0b0000_1000);
    pub const STRUCTURAL:       TokenRole = TokenRole(0b0001_0000);
    pub const SPECIAL:          TokenRole = TokenRole(0b0010_0000);
    pub const CODE_CHUNK:       TokenRole = TokenRole(0b0100_0000);
    pub const TOOL_CALL:        TokenRole = TokenRole(0b1000_0000);
    pub const MATH_EXPR:        TokenRole = TokenRole(0b0000_1000);
    pub const MEMORY_UPDATE:    TokenRole = TokenRole(0b0001_0000);

    pub fn empty() -> Self { TokenRole(0) }

    pub fn contains(&self, other: TokenRole) -> bool {
        (self.0 & other.0) == other.0
    }

    pub fn insert(&mut self, other: TokenRole) {
        self.0 |= other.0;
    }

    pub fn bits(&self) -> u8 { self.0 }

    pub fn from_bits(bits: u8) -> Self { TokenRole(bits) }
}

#[derive(Debug, Clone)]
pub struct SuperToken {
    pub vector: Hypervector,
    pub token_count: usize,
    pub role_flags: TokenRole,
    pub start_pos: usize,
    pub raw_tokens: Vec<u32>,
}

impl SuperToken {
    pub fn new(vector: Hypervector, start_pos: usize) -> Self {
        Self {
            vector,
            token_count: 0,
            role_flags: TokenRole::empty(),
            start_pos,
            raw_tokens: Vec::new(),
        }
    }

    pub fn compression_ratio(&self) -> f64 {
        if self.token_count == 0 { 1.0 }
        else { self.raw_tokens.len() as f64 / self.token_count as f64 }
    }
}
