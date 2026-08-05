// Fuga Mentalese: Concept-Driven Generation
// Transition from token stream to thought stream.
// L1: interpret_thought()  — TM prediction -> abstract Thought
// L2: render_thought()     — Thought -> syntactically valid Rust code
// L2+: PainAvoidance veto  — concept-level veto

use crate::ai::sdr::{encode_text, SdrVector};

fn overlap(a: &SdrVector, b: &SdrVector) -> u32 { a.overlap(b) }
fn bind(a: &SdrVector, b: &SdrVector) -> SdrVector { a.bind(b) }
fn bundle(a: &SdrVector, others: &[&SdrVector]) -> SdrVector { a.bundle(others) }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Thought {
    BinaryOp,      // a + b, a - b, a * b
    ReturnVar,     // return x
    Print,         // println!("{}", x)
    DeclareVar,    // let x = ...
    IfNone,        // if x.is_none() { ... }
    MethodCall,    // x.method()
    FieldAccess,   // x.field
    Loop,          // for/while
    Match,         // match x { ... }
    Assign,        // x = y
    Unknown,
}

impl Thought {
    /// SDR representation for resonance matching
    pub fn to_sdr(&self) -> SdrVector {
        match self {
            Thought::BinaryOp   => encode_text("THOUGHT_BINARY_OP"),
            Thought::ReturnVar  => encode_text("THOUGHT_RETURN_VAR"),
            Thought::Print      => encode_text("THOUGHT_PRINT"),
            Thought::DeclareVar => encode_text("THOUGHT_DECLARE_VAR"),
            Thought::IfNone     => encode_text("THOUGHT_IF_NONE"),
            Thought::MethodCall => encode_text("THOUGHT_METHOD_CALL"),
            Thought::FieldAccess=> encode_text("THOUGHT_FIELD_ACCESS"),
            Thought::Loop       => encode_text("THOUGHT_LOOP"),
            Thought::Match      => encode_text("THOUGHT_MATCH"),
            Thought::Assign     => encode_text("THOUGHT_ASSIGN"),
            Thought::Unknown    => encode_text("THOUGHT_UNKNOWN"),
        }
    }

    /// All known thoughts for classification
    pub fn all() -> &'static [Thought] {
        &[
            Thought::BinaryOp,
            Thought::ReturnVar,
            Thought::Print,
            Thought::DeclareVar,
            Thought::IfNone,
            Thought::MethodCall,
            Thought::FieldAccess,
            Thought::Loop,
            Thought::Match,
            Thought::Assign,
        ]
    }
}

/// L1: Project TM prediction (token-level SDR) onto the space of abstract Thoughts.
/// Returns the best-matching Thought with its confidence score (0.0-1.0).
pub fn interpret_thought(tm_prediction: &SdrVector) -> (Thought, f64) {
    let mut best = Thought::Unknown;
    let mut max_overlap = 0u32;

    for thought in Thought::all() {
        let ov = overlap(tm_prediction, &thought.to_sdr());
        if ov > max_overlap {
            max_overlap = ov;
            best = *thought;
        }
    }

    // Normalize by prediction popcount to get ~confidence
    let denom = tm_prediction.popcount().max(1) as f64;
    let confidence = max_overlap as f64 / denom;
    (best, confidence)
}

/// L2: Render a Thought into syntactically valid Rust code.
/// args are pulled from the function signature (param names).
pub fn render_thought(thought: Thought, args: &[&str], has_return: bool) -> String {
    match thought {
        Thought::BinaryOp => {
            // Need at least 2 args: a, b
            if args.len() >= 2 {
                format!("{} + {}", args[0], args[1])
            } else if args.len() == 1 {
                format!("{} + 0", args[0])
            } else {
                "0 + 0".to_string()
            }
        }
        Thought::ReturnVar => {
            if !args.is_empty() {
                args[0].to_string()
            } else if has_return {
                "Default::default()".to_string()
            } else {
                "".to_string()
            }
        }
        Thought::Print => {
            if !args.is_empty() {
                format!("println!(\"{{}}\", {});", args[0])
            } else {
                "println!();".to_string()
            }
        }
        Thought::DeclareVar => {
            if args.len() >= 2 {
                format!("let {} = {};", args[0], args[1])
            } else if args.len() == 1 {
                format!("let {} = 0;", args[0])
            } else {
                "let x = 0;".to_string()
            }
        }
        Thought::IfNone => {
            if args.len() >= 2 {
                format!("if {}.is_none() {{ {} }}", args[0], args[1])
            } else if args.len() == 1 {
                format!("if {}.is_none() {{ }}", args[0])
            } else {
                "if true { }".to_string()
            }
        }
        Thought::MethodCall => {
            if args.len() >= 2 {
                format!("{}.{}()", args[0], args[1])
            } else if args.len() == 1 {
                format!("{}.method()", args[0])
            } else {
                "x.method()".to_string()
            }
        }
        Thought::FieldAccess => {
            if args.len() >= 2 {
                format!("{}.{}", args[0], args[1])
            } else if args.len() == 1 {
                format!("{}.field", args[0])
            } else {
                "x.field".to_string()
            }
        }
        Thought::Loop => {
            if args.len() >= 1 {
                format!("for {} in iter {{ }}", args[0])
            } else {
                "for x in iter { }".to_string()
            }
        }
        Thought::Match => {
            if args.len() >= 1 {
                format!("match {} {{ _ => {{ }} }}", args[0])
            } else {
                "match x { _ => { } }".to_string()
            }
        }
        Thought::Assign => {
            if args.len() >= 2 {
                format!("{} = {};", args[0], args[1])
            } else if args.len() == 1 {
                format!("{} = 0;", args[0])
            } else {
                "x = 0;".to_string()
            }
        }
        Thought::Unknown | _ => "Default::default()".to_string(),
    }
}

/// Extract parameter names from a Rust function signature.
/// e.g., "fn sum(a: i32, b: i32) -> i32" -> ["a", "b"]
pub fn extract_params(signature: &str) -> Vec<String> {
    // Find content between ( and )
    let start = signature.find('(').unwrap_or(0);
    let end = signature.rfind(')').unwrap_or(signature.len());
    if start >= end { return vec![]; }
    
    let params_str = &signature[start+1..end];
    params_str
        .split(',')
        .filter_map(|p| {
            let p = p.trim();
            if p.is_empty() { return None; }
            // Split by ':' to get name before type
            p.split(':').next().map(|s| s.trim().to_string())
        })
        .collect()
}

/// Build the full function body by combining signature + rendered thought.
pub fn generate_body(signature: &str, thought: Thought) -> String {
    let params = extract_params(signature);
    let param_refs: Vec<&str> = params.iter().map(|s| s.as_str()).collect();
    let has_return = signature.contains("->");
    let body = render_thought(thought, &param_refs, has_return);
    
    // If body is just an expression, wrap in block
    let body = if body.trim().ends_with(';') || body.trim().starts_with("if ") || body.trim().starts_with("for ") || body.trim().starts_with("match ") {
        format!(" {}", body)
    } else if !body.is_empty() {
        format!(" {{ {} }}", body)
    } else if has_return {
        " { Default::default() }".to_string()
    } else {
        " { }".to_string()
    };
    
    format!("{}{}", signature, body)
}