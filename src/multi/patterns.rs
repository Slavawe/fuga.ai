use crate::multi::language::LanguageId;
use std::collections::HashMap;

/// Список всех паттернов (для итерации)
static ALL_PATTERNS: &[ViolationPattern] = &[
    ViolationPattern::UnsafeBlock,
    ViolationPattern::UnwrapOrExpect,
    ViolationPattern::NullPointerDeref,
    ViolationPattern::DivisionByZero,
    ViolationPattern::ArrayIndexOutOfBounds,
    ViolationPattern::IntegerOverflow,
    ViolationPattern::InfiniteLoop,
    ViolationPattern::MissingBreak,
    ViolationPattern::DeepRecursion,
    ViolationPattern::UseAfterFree,
    ViolationPattern::DoubleFree,
    ViolationPattern::UninitializedMemory,
    ViolationPattern::BufferOverflow,
    ViolationPattern::RaceCondition,
    ViolationPattern::DeadlockRisk,
    ViolationPattern::SqlInjection,
    ViolationPattern::CommandInjection,
    ViolationPattern::FormatStringVulnerability,
    ViolationPattern::PathTraversal,
    ViolationPattern::HardcodedSecret,
    ViolationPattern::WeakRandomness,
    ViolationPattern::PanicInDrop,
    ViolationPattern::UnusedMustUseResult,
    ViolationPattern::NonExhaustiveMatch,
    ViolationPattern::UnreachableCode,
];

/// Универсальные паттерны нарушений (language-agnostic)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ViolationPattern {
    // Safety
    UnsafeBlock,
    UnwrapOrExpect,
    NullPointerDeref,
    DivisionByZero,
    ArrayIndexOutOfBounds,
    IntegerOverflow,

    // Control flow
    InfiniteLoop,
    MissingBreak,
    DeepRecursion,

    // Memory
    UseAfterFree,
    DoubleFree,
    UninitializedMemory,
    BufferOverflow,

    // Concurrency
    RaceCondition,
    DeadlockRisk,

    // Security
    SqlInjection,
    CommandInjection,
    FormatStringVulnerability,
    PathTraversal,
    HardcodedSecret,
    WeakRandomness,

    // Quality
    PanicInDrop,
    UnusedMustUseResult,
    NonExhaustiveMatch,
    UnreachableCode,
}

impl ViolationPattern {
    /// Все паттерны для итерации
    pub fn iter_all() -> &'static [ViolationPattern] {
        ALL_PATTERNS
    }

    /// Возвращает tree-sitter query для конкретного языка
    pub fn query(&self, lang: LanguageId) -> Option<&'static str> {
        QUERIES.get(&(*self, lang)).copied()
    }

    /// Severity по умолчанию
    pub fn default_severity(&self) -> Severity {
        match self {
            ViolationPattern::UnsafeBlock => Severity::Critical,
            ViolationPattern::UnwrapOrExpect => Severity::High,
            ViolationPattern::NullPointerDeref => Severity::Critical,
            ViolationPattern::DivisionByZero => Severity::Critical,
            ViolationPattern::ArrayIndexOutOfBounds => Severity::High,
            ViolationPattern::IntegerOverflow => Severity::High,
            ViolationPattern::InfiniteLoop => Severity::Medium,
            ViolationPattern::MissingBreak => Severity::Low,
            ViolationPattern::DeepRecursion => Severity::Medium,
            ViolationPattern::UseAfterFree => Severity::Critical,
            ViolationPattern::DoubleFree => Severity::Critical,
            ViolationPattern::UninitializedMemory => Severity::Critical,
            ViolationPattern::BufferOverflow => Severity::Critical,
            ViolationPattern::RaceCondition => Severity::High,
            ViolationPattern::DeadlockRisk => Severity::High,
            ViolationPattern::SqlInjection => Severity::Critical,
            ViolationPattern::CommandInjection => Severity::Critical,
            ViolationPattern::FormatStringVulnerability => Severity::High,
            ViolationPattern::PathTraversal => Severity::High,
            ViolationPattern::HardcodedSecret => Severity::Critical,
            ViolationPattern::WeakRandomness => Severity::Medium,
            ViolationPattern::PanicInDrop => Severity::High,
            ViolationPattern::UnusedMustUseResult => Severity::Medium,
            ViolationPattern::NonExhaustiveMatch => Severity::Low,
            ViolationPattern::UnreachableCode => Severity::Low,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Tree-sitter queries для каждого (pattern, language)
/// Формат: (pattern, language) -> query string
lazy_static::lazy_static! {
    static ref QUERIES: HashMap<(ViolationPattern, LanguageId), &'static str> = {
        let mut m = HashMap::new();

        // ===== RUST =====
        m.insert((ViolationPattern::UnsafeBlock, LanguageId::Rust),
            "(unsafe_block) @violation");
        m.insert((ViolationPattern::UnwrapOrExpect, LanguageId::Rust),
            "(call_expression
                function: (scoped_identifier
                    path: (path
                        segments: (path_segment name: (identifier) @method
                            (#any-of? @method \"unwrap\" \"expect\")
                        )
                    )
                )
            ) @violation");
        m.insert((ViolationPattern::DivisionByZero, LanguageId::Rust),
            "(binary_expression
                operator: (div | rem)
                right: (literal_expression value: (integer_literal) @zero
                    (#eq? @zero \"0\")
                )
            ) @violation");
        m.insert((ViolationPattern::InfiniteLoop, LanguageId::Rust),
            "(loop_expression
                body: (block) @violation
                (#not-has-break? @violation)
            )");
        m.insert((ViolationPattern::SqlInjection, LanguageId::Rust),
            "(macro_invocation
                macro: (identifier) @macro
                (#any-of? @macro \"sqlx\" \"diesel\" \"query\" \"execute\")
                (token_tree) @violation
            )");
        m.insert((ViolationPattern::CommandInjection, LanguageId::Rust),
            "(call_expression
                function: (scoped_identifier
                    path: (path
                        segments: (path_segment name: (identifier) @func
                            (#any-of? @func \"Command\" \"spawn\" \"exec\" \"output\")
                        )
                    )
                )
            ) @violation");
        m.insert((ViolationPattern::HardcodedSecret, LanguageId::Rust),
            "(assignment_expression
                left: (identifier) @var
                (#match? @var \"(?i)(password|secret|token|key|api)\")
                right: (string_literal) @violation
            )");

        // ===== C / C++ =====
        m.insert((ViolationPattern::UnsafeBlock, LanguageId::C),
            "// C has no explicit unsafe, but we check for risky patterns
            (call_expression
                function: (identifier) @func
                (#any-of? @func \"gets\" \"strcpy\" \"strcat\" \"sprintf\" \"vsprintf\" \"scanf\")
            ) @violation");
        m.insert((ViolationPattern::NullPointerDeref, LanguageId::C),
            "(unary_expression
                operator: \"*\"
                argument: (identifier) @ptr
            ) @violation");
        m.insert((ViolationPattern::DivisionByZero, LanguageId::C),
            "(binary_expression
                operator: \"/\" | \"%\"
                right: (number_literal) @zero
                (#eq? @zero \"0\")
            ) @violation");
        m.insert((ViolationPattern::BufferOverflow, LanguageId::C),
            "(call_expression
                function: (identifier) @func
                (#any-of? @func \"gets\" \"strcpy\" \"strcat\" \"sprintf\" \"vsprintf\" \"scanf\" \"fscanf\" \"sscanf\")
            ) @violation");
        m.insert((ViolationPattern::FormatStringVulnerability, LanguageId::C),
            "(call_expression
                function: (identifier) @func
                (#any-of? @func \"printf\" \"fprintf\" \"sprintf\" \"snprintf\" \"vprintf\" \"vfprintf\" \"vsprintf\" \"vsnprintf\")
                arguments: (argument_list
                    (string_literal) @fmt
                )
            ) @violation");
        m.insert((ViolationPattern::SqlInjection, LanguageId::C),
            "// C: look for string concatenation in SQL queries
            (binary_expression
                operator: \"+\" | \"%\" 
                left: (string_literal) @sql
                (#match? @sql \"(?i)(select|insert|update|delete|union)\")
            ) @violation");
        m.insert((ViolationPattern::HardcodedSecret, LanguageId::C),
            "(init_declarator
                declarator: (identifier) @var
                (#match? @var \"(?i)(password|secret|token|key|api)\")
                value: (string_literal) @violation
            )");
        // C++ same as C for most
        let c_unsafe = m.get(&(ViolationPattern::UnsafeBlock, LanguageId::C)).copied().unwrap_or("");
        m.insert((ViolationPattern::UnsafeBlock, LanguageId::Cpp), c_unsafe);
        let c_null = m.get(&(ViolationPattern::NullPointerDeref, LanguageId::C)).copied().unwrap_or("");
        m.insert((ViolationPattern::NullPointerDeref, LanguageId::Cpp), c_null);
        let c_div = m.get(&(ViolationPattern::DivisionByZero, LanguageId::C)).copied().unwrap_or("");
        m.insert((ViolationPattern::DivisionByZero, LanguageId::Cpp), c_div);
        let c_buf = m.get(&(ViolationPattern::BufferOverflow, LanguageId::C)).copied().unwrap_or("");
        m.insert((ViolationPattern::BufferOverflow, LanguageId::Cpp), c_buf);
        let c_fmt = m.get(&(ViolationPattern::FormatStringVulnerability, LanguageId::C)).copied().unwrap_or("");
        m.insert((ViolationPattern::FormatStringVulnerability, LanguageId::Cpp), c_fmt);
        let c_sql = m.get(&(ViolationPattern::SqlInjection, LanguageId::C)).copied().unwrap_or("");
        m.insert((ViolationPattern::SqlInjection, LanguageId::Cpp), c_sql);
        let c_secret = m.get(&(ViolationPattern::HardcodedSecret, LanguageId::C)).copied().unwrap_or("");
        m.insert((ViolationPattern::HardcodedSecret, LanguageId::Cpp), c_secret);

        // ===== GO =====
        m.insert((ViolationPattern::NullPointerDeref, LanguageId::Go),
            "(selector_expression
                operand: (nil)
            ) @violation");
        m.insert((ViolationPattern::SqlInjection, LanguageId::Go),
            "(call_expression
                function: (selector_expression
                    field: (field_identifier) @method
                    (#any-of? @method \"Exec\" \"Query\" \"QueryRow\")
                )
                arguments: (argument_list (string_literal) @sql)
            ) @violation");
        m.insert((ViolationPattern::CommandInjection, LanguageId::Go),
            "(call_expression
                function: (selector_expression
                    field: (field_identifier) @cmd
                    (#any-of? @cmd \"Command\" \"Run\" \"Start\" \"Output\" \"CombinedOutput\")
                )
            ) @violation");
        m.insert((ViolationPattern::HardcodedSecret, LanguageId::Go),
            "(var_declaration
                (var_spec
                    name: (identifier) @var
                    (#match? @var \"(?i)(password|secret|token|key|api)\")
                    value: (string_literal) @violation
                )
            )");

        // ===== PYTHON =====
        // Note: Python's `call` and `attribute` nodes use positional children, not named fields
        // We match broadly and filter in Rust code (syntax_layer.rs)
        m.insert((ViolationPattern::UnwrapOrExpect, LanguageId::Python),
            "(call (attribute) @violation)");
        m.insert((ViolationPattern::DivisionByZero, LanguageId::Python),
            "(binary_operator
                operator: \"/\" | \"%\"
                right: (integer) @zero
                (#eq? @zero \"0\")
            ) @violation");
        m.insert((ViolationPattern::SqlInjection, LanguageId::Python),
            "(call
                (attribute)
                (argument_list
                    (string (interpolation) @interp) @violation
                )
            )");
        m.insert((ViolationPattern::CommandInjection, LanguageId::Python),
            "(call (attribute) @violation)");
        m.insert((ViolationPattern::FormatStringVulnerability, LanguageId::Python),
            "(call (attribute) @violation)");
        m.insert((ViolationPattern::HardcodedSecret, LanguageId::Python),
            "(assignment
                left: (identifier) @var
                (#match? @var \"(?i)(password|secret|token|key|api)\")
                right: (string) @violation
            )");

        // ===== TYPESCRIPT / JAVASCRIPT =====
        m.insert((ViolationPattern::NullPointerDeref, LanguageId::TypeScript),
            "(member_expression
                object: (null)
            ) @violation");
        m.insert((ViolationPattern::DivisionByZero, LanguageId::TypeScript),
            "(binary_expression
                operator: \"/\" | \"%\"
                right: (number) @zero
                (#eq? @zero \"0\")
            ) @violation");
        m.insert((ViolationPattern::SqlInjection, LanguageId::TypeScript),
            "(call_expression
                function: (member_expression
                    property: (property_identifier) @method
                    (#any-of? @method \"query\" \"execute\" \"run\")
                )
                arguments: (arguments (string) @sql)
            ) @violation");
        m.insert((ViolationPattern::CommandInjection, LanguageId::TypeScript),
            "(call_expression
                function: (member_expression
                    property: (property_identifier) @cmd
                    (#any-of? @cmd \"exec\" \"spawn\" \"execSync\" \"spawnSync\")
                )
            ) @violation");
        m.insert((ViolationPattern::FormatStringVulnerability, LanguageId::TypeScript),
            "(call_expression
                function: (member_expression
                    property: (property_identifier) @fmt
                    (#any-of? @fmt \"format\" \"template\")
                )
            ) @violation");
        m.insert((ViolationPattern::HardcodedSecret, LanguageId::TypeScript),
            "(lexical_declaration
                (variable_declarator
                    name: (identifier) @var
                    (#match? @var \"(?i)(password|secret|token|key|api)\")
                    value: (string) @violation
                )
            )");
        m.insert((ViolationPattern::UnsafeBlock, LanguageId::TypeScript),
            "// TypeScript: look for eval, Function constructor
            (call_expression
                function: (identifier) @func
                (#any-of? @func \"eval\" \"Function\" \"setTimeout\" \"setInterval\")
            ) @violation");

        // JavaScript shares most with TypeScript
        let ts_null = m.get(&(ViolationPattern::NullPointerDeref, LanguageId::TypeScript)).copied().unwrap_or("");
        m.insert((ViolationPattern::NullPointerDeref, LanguageId::JavaScript), ts_null);
        let ts_div = m.get(&(ViolationPattern::DivisionByZero, LanguageId::TypeScript)).copied().unwrap_or("");
        m.insert((ViolationPattern::DivisionByZero, LanguageId::JavaScript), ts_div);
        let ts_sql = m.get(&(ViolationPattern::SqlInjection, LanguageId::TypeScript)).copied().unwrap_or("");
        m.insert((ViolationPattern::SqlInjection, LanguageId::JavaScript), ts_sql);
        let ts_cmd = m.get(&(ViolationPattern::CommandInjection, LanguageId::TypeScript)).copied().unwrap_or("");
        m.insert((ViolationPattern::CommandInjection, LanguageId::JavaScript), ts_cmd);
        let ts_fmt = m.get(&(ViolationPattern::FormatStringVulnerability, LanguageId::TypeScript)).copied().unwrap_or("");
        m.insert((ViolationPattern::FormatStringVulnerability, LanguageId::JavaScript), ts_fmt);
        let ts_secret = m.get(&(ViolationPattern::HardcodedSecret, LanguageId::TypeScript)).copied().unwrap_or("");
        m.insert((ViolationPattern::HardcodedSecret, LanguageId::JavaScript), ts_secret);
        let ts_unsafe = m.get(&(ViolationPattern::UnsafeBlock, LanguageId::TypeScript)).copied().unwrap_or("");
        m.insert((ViolationPattern::UnsafeBlock, LanguageId::JavaScript), ts_unsafe);

        m
    };
}
