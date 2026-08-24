use crate::multi::language::{LanguageId, collect_function_names, parse_source};

pub struct CodeTranslator;

impl CodeTranslator {
    pub fn new() -> Self {
        Self
    }

    pub fn translate(
        &self,
        source: &str,
        from: LanguageId,
        to: LanguageId,
    ) -> Result<String, String> {
        match (from, to) {
            (LanguageId::C, LanguageId::Rust) => self.c_to_rust(source),
            (LanguageId::Cpp, LanguageId::Rust) => self.c_to_rust(source),
            (LanguageId::Python, LanguageId::Rust) => self.python_to_rust(source),
            (LanguageId::Go, LanguageId::Rust) => self.go_to_rust(source),
            (LanguageId::Rust, LanguageId::C) => self.rust_to_c(source),
            (LanguageId::Rust, LanguageId::Python) => self.rust_to_python(source),
            (from_lang, to_lang) => Err(format!(
                "Translation from {:?} to {:?} is not yet supported",
                from_lang, to_lang
            )),
        }
    }

    fn c_to_rust(&self, source: &str) -> Result<String, String> {
        let tree = parse_source(source, LanguageId::C).ok_or("Failed to parse C source")?;

        let functions = collect_function_names(&tree, source, LanguageId::C);
        if functions.is_empty() {
            return Err("No functions found in C source".to_string());
        }

        let mut output = String::new();
        output.push_str("// Auto-translated from C to Rust\n");
        output.push_str("// Review and verify before use\n\n");

        // Extract includes as comments
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("#include") {
                output.push_str(&format!("// {}\n", trimmed));
            }
        }
        output.push('\n');

        // Translate function signatures
        let root = tree.root_node();
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if child.kind() == "function_definition" {
                if let Some(rust_fn) = self.translate_c_function(child, source) {
                    output.push_str(&rust_fn);
                    output.push_str("\n\n");
                }
            }
        }

        Ok(output)
    }

    fn translate_c_function(&self, node: tree_sitter::Node, source: &str) -> Option<String> {
        let mut cursor = node.walk();
        let mut return_type = "()";
        let mut name = String::new();
        let mut params = Vec::new();
        let mut body_start = 0;
        let mut body_end = 0;

        for child in node.children(&mut cursor) {
            match child.kind() {
                "primitive_type" | "type_identifier" | "sized_type_specifier" => {
                    return_type = source[child.byte_range()].trim();
                }
                "function_declarator" => {
                    let mut dec_cursor = child.walk();
                    for dec_child in child.children(&mut dec_cursor) {
                        match dec_child.kind() {
                            "identifier" => {
                                name = source[dec_child.byte_range()].to_string();
                            }
                            "qualified_identifier" => {
                                let mut qc = dec_child.walk();
                                for qchild in dec_child.children(&mut qc) {
                                    if qchild.kind() == "identifier" {
                                        name = source[qchild.byte_range()].to_string();
                                    }
                                }
                            }
                            "parameter_list" => {
                                let mut p_cursor = dec_child.walk();
                                for param in dec_child.children(&mut p_cursor) {
                                    if param.kind() == "parameter_declaration" {
                                        if let Some(p) = self.parse_c_param(param, source) {
                                            params.push(p);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "compound_statement" => {
                    body_start = child.start_byte();
                    body_end = child.end_byte();
                }
                _ => {}
            }
        }

        let rust_return = self.c_type_to_rust(return_type);
        let rust_params: Vec<String> = params
            .iter()
            .map(|(n, t)| format!("{}: {}", n, t))
            .collect();
        let rust_sig = format!(
            "fn {}({}) -> {} {{",
            name,
            rust_params.join(", "),
            rust_return
        );

        let body = if rust_return == "()" {
            &source[body_start..body_end]
        } else {
            let raw_body = &source[body_start..body_end];
            raw_body
        };

        let body = self.translate_c_body(body, &rust_return);

        Some(format!("{}\n{}\n}}", rust_sig, body))
    }

    fn parse_c_param(&self, node: tree_sitter::Node, source: &str) -> Option<(String, String)> {
        let mut cursor = node.walk();
        let mut c_type = String::new();
        let mut name = String::new();
        let mut pointer_depth = 0usize;
        for child in node.children(&mut cursor) {
            match child.kind() {
                "primitive_type" | "type_identifier" => {
                    c_type = source[child.byte_range()].trim().to_string();
                }
                "struct_specifier" | "class_specifier" | "union_specifier" => {
                    let mut sc = child.walk();
                    for schild in child.children(&mut sc) {
                        if schild.kind() == "type_identifier" {
                            c_type = source[schild.byte_range()].to_string();
                            break;
                        }
                    }
                    if c_type.is_empty() {
                        c_type = source[child.byte_range()].trim().to_string();
                    }
                }
                "pointer_declarator" => {
                    pointer_depth += 1;
                    let mut dc = child.walk();
                    for dchild in child.children(&mut dc) {
                        if dchild.kind() == "identifier" {
                            name = source[dchild.byte_range()].to_string();
                        } else if dchild.kind() == "pointer_declarator" {
                            pointer_depth += 1;
                            let mut dd = dchild.walk();
                            for ddchild in dchild.children(&mut dd) {
                                if ddchild.kind() == "identifier" {
                                    name = source[ddchild.byte_range()].to_string();
                                }
                            }
                        }
                    }
                }
                "identifier" => {
                    if name.is_empty() {
                        name = source[child.byte_range()].to_string();
                    }
                }
                _ => {}
            }
        }
        if !name.is_empty() {
            let rust_type = if c_type.is_empty() {
                if pointer_depth > 0 {
                    let mut t = String::from("*mut i8");
                    for _ in 1..pointer_depth {
                        t = format!("*mut {}", t);
                    }
                    t
                } else {
                    "()".to_string()
                }
            } else {
                let mut t = self.c_type_to_rust(&c_type);
                for _ in 0..pointer_depth {
                    t = format!("*mut {}", t);
                }
                t
            };
            Some((name, rust_type))
        } else {
            None
        }
    }

    fn c_type_to_rust(&self, c_type: &str) -> String {
        let trimmed = c_type.trim();
        match trimmed {
            "int" => "i32".to_string(),
            "long" => "i64".to_string(),
            "float" => "f32".to_string(),
            "double" => "f64".to_string(),
            "char" => "i8".to_string(),
            "void" => "()".to_string(),
            "size_t" => "usize".to_string(),
            "bool" | "_Bool" => "bool".to_string(),
            "uint8_t" => "u8".to_string(),
            "uint16_t" => "u16".to_string(),
            "uint32_t" => "u32".to_string(),
            "uint64_t" => "u64".to_string(),
            "int8_t" => "i8".to_string(),
            "int16_t" => "i16".to_string(),
            "int32_t" => "i32".to_string(),
            "int64_t" => "i64".to_string(),
            other => other.to_string(),
        }
    }

    fn translate_c_body(&self, body: &str, _ret_type: &str) -> String {
        let mut out = String::new();
        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed == "{" || trimmed == "}" {
                out.push_str(line);
                out.push('\n');
                continue;
            }
            let translated = if trimmed.starts_with("return ") {
                format!("    {}", trimmed)
            } else if trimmed.starts_with("fprintf(stderr,") {
                let args = trimmed
                    .trim_start_matches("fprintf(stderr, ")
                    .trim_end_matches(';')
                    .trim_end_matches(')');
                if let Some(comma) = args.find(',') {
                    let fmt = &args[..comma];
                    let rest = &args[comma + 1..];
                    format!("    eprintln!({}, {});", fmt, rest.trim())
                } else {
                    format!("    eprintln!({});", args)
                }
            } else if trimmed.starts_with("fprintf(") {
                let inner = trimmed.trim_start_matches("fprintf(").trim_end_matches(';');
                let close = inner.rfind(')').unwrap_or(inner.len());
                let inner = &inner[..close];
                if let Some(comma) = inner.find(',') {
                    let file = &inner[..comma].trim();
                    let rest = &inner[comma + 1..];
                    if let Some(comma2) = rest.find(',') {
                        let fmt = &rest[..comma2].trim();
                        let args = &rest[comma2 + 1..];
                        format!("    // fprintf({}, {}, {});", file, fmt, args.trim())
                    } else {
                        format!("    // fprintf({}, {});", file, rest.trim())
                    }
                } else {
                    format!("    // fprintf({});", inner)
                }
            } else if trimmed.starts_with("printf(") {
                let args = self.extract_parens(trimmed, "printf");
                format!("    print!(\"{{}}\", {});", args)
            } else if trimmed.contains("nullptr") || trimmed.contains("NULL") {
                let l = trimmed.replace("nullptr", "None").replace("NULL", "None");
                let l = l.trim_end_matches(';');
                format!("    {};", l)
            } else if trimmed.starts_with("int ")
                || trimmed.starts_with("float ")
                || trimmed.starts_with("double ")
                || trimmed.starts_with("char ")
                || trimmed.starts_with("size_t")
                || trimmed.starts_with("uint")
                || trimmed.starts_with("bool ")
                || trimmed.starts_with("auto ")
            {
                let without_type = trimmed.splitn(2, ' ').nth(1).unwrap_or("");
                format!("    let mut {}", without_type)
            } else if trimmed.starts_with("const ") {
                let without_const = trimmed.trim_start_matches("const ");
                let rest = without_const.splitn(2, ' ').nth(1).unwrap_or(without_const);
                format!("    let {}", rest)
            } else if trimmed.starts_with("if (") {
                let cond = self.extract_parens(trimmed, "if");
                format!("    if {} {{", cond)
            } else if trimmed.starts_with("for (") {
                let _init_end = trimmed.find(';').unwrap_or(trimmed.len());
                format!("    // for loop: {}", trimmed)
            } else if trimmed.starts_with("while (") {
                let cond = self.extract_parens(trimmed, "while");
                format!("    while {} {{", cond)
            } else if trimmed.starts_with("//") || trimmed.starts_with("/*") {
                format!("    {}", trimmed)
            } else if trimmed.starts_with("std::") {
                let l = trimmed
                    .replace("std::vector<", "Vec::<")
                    .replace("std::string", "String")
                    .replace("std::map<", "HashMap::<")
                    .replace("std::unordered_map<", "HashMap::<")
                    .replace("std::to_string", ".to_string")
                    .replace("std::max", "std::cmp::max")
                    .replace("std::min", "std::cmp::min");
                let l = l.trim_end_matches(';');
                format!("    {};", l)
            } else if trimmed.starts_with("GGML_") {
                format!("    // {} (FFI call)", trimmed)
            } else if trimmed.starts_with("static ")
                || trimmed.starts_with("struct ")
                || trimmed.starts_with("enum ")
            {
                format!("    // C++ type: {}", trimmed)
            } else {
                let l = trimmed.trim_end_matches(';');
                format!("    {};", l)
            };
            out.push_str(&translated);
            out.push('\n');
        }
        out
    }

    fn python_to_rust(&self, source: &str) -> Result<String, String> {
        let tree =
            parse_source(source, LanguageId::Python).ok_or("Failed to parse Python source")?;

        let _functions = collect_function_names(&tree, source, LanguageId::Python);
        let mut output = String::new();
        output.push_str("// Auto-translated from Python to Rust\n");
        output.push_str("// Review and verify before use\n\n");

        let root = tree.root_node();
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if child.kind() == "function_definition" {
                if let Some(rust_fn) = self.translate_py_function(child, source) {
                    output.push_str(&rust_fn);
                    output.push_str("\n\n");
                }
            }
        }

        if output.len() < 100 {
            return Err("Translation produced minimal output".to_string());
        }

        Ok(output)
    }

    fn translate_py_function(&self, node: tree_sitter::Node, source: &str) -> Option<String> {
        let mut cursor = node.walk();
        let mut name = String::new();
        let mut params = Vec::new();
        let mut body = String::new();

        for child in node.children(&mut cursor) {
            match child.kind() {
                "identifier" => {
                    name = source[child.byte_range()].to_string();
                }
                "parameters" => {
                    let mut p_cursor = child.walk();
                    for param in child.children(&mut p_cursor) {
                        if param.kind() == "identifier" {
                            params.push((source[param.byte_range()].to_string(), String::new()));
                        } else if param.kind() == "typed_parameter" {
                            // Python 3 type hints - rare in basic code
                            let mut tp_cursor = param.walk();
                            for tp_child in param.children(&mut tp_cursor) {
                                if tp_child.kind() == "identifier" {
                                    params.push((
                                        source[tp_child.byte_range()].to_string(),
                                        String::new(),
                                    ));
                                }
                            }
                        }
                    }
                }
                "block" => {
                    let block_src = &source[child.byte_range()];
                    body = self.translate_py_block(block_src);
                }
                _ => {}
            }
        }

        let rust_params: Vec<String> = params
            .into_iter()
            .map(|(n, _)| format!("{}: /* type */", n))
            .collect();

        Some(format!(
            "fn {}({}) {{\n{}\n}}",
            name,
            rust_params.join(", "),
            body
        ))
    }

    fn translate_py_block(&self, source: &str) -> String {
        let mut out = String::new();
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed == "pass" {
                continue;
            }
            let translated = if trimmed.starts_with("return ") {
                format!("    {}", trimmed)
            } else if trimmed.starts_with("def ") {
                format!("    // inner function: {}", trimmed)
            } else if trimmed.starts_with("if ") {
                let cond = trimmed.trim_start_matches("if ").trim_end_matches(':');
                let cond_rust = cond
                    .replace(" and ", " && ")
                    .replace(" or ", " || ")
                    .replace(" not ", " !");
                format!("    if {} {{", cond_rust)
            } else if trimmed.starts_with("elif ") {
                let cond = trimmed.trim_start_matches("elif ").trim_end_matches(':');
                let cond_rust = cond.replace(" and ", " && ").replace(" or ", " || ");
                format!("    }} else if {} {{", cond_rust)
            } else if trimmed.starts_with("else:") {
                "    } else {".to_string()
            } else if trimmed.starts_with("for ") {
                let rest = trimmed.trim_start_matches("for ").trim_end_matches(':');
                format!("    // for loop: {}", rest)
            } else if trimmed.starts_with("while ") {
                let rest = trimmed.trim_start_matches("while ").trim_end_matches(':');
                let cond_rust = rest.replace(" and ", " && ").replace(" or ", " || ");
                format!("    while {} {{", cond_rust)
            } else if trimmed.starts_with("try:") {
                "    // try-catch not directly supported".to_string()
            } else if trimmed.starts_with("except") {
                "    // exception handling".to_string()
            } else if trimmed.starts_with('#') {
                format!("    //{}", &trimmed[1..])
            } else if trimmed.starts_with("print(") {
                let args = self.extract_parens(trimmed, "print");
                format!("    println!(\"{{:?}}\", {});", args)
            } else {
                format!("    {};", trimmed.trim_end_matches(':'))
            };
            out.push_str(&translated);
            out.push('\n');
        }
        out
    }

    fn go_to_rust(&self, source: &str) -> Result<String, String> {
        let tree = parse_source(source, LanguageId::Go).ok_or("Failed to parse Go source")?;
        let _functions = collect_function_names(&tree, source, LanguageId::Go);
        let mut output = String::new();
        output.push_str("// Auto-translated from Go to Rust\n");
        output.push_str("// Review and verify before use\n\n");

        let root = tree.root_node();
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if child.kind() == "function_declaration" {
                let mut fc = child.walk();
                let mut name = String::new();
                for gc in child.children(&mut fc) {
                    if gc.kind() == "identifier" {
                        name = source[gc.byte_range()].to_string();
                    }
                }
                if !name.is_empty() {
                    output.push_str(&format!(
                        "fn {}() {{\n    // translated from Go\n}}\n\n",
                        name
                    ));
                }
            }
        }
        Ok(output)
    }

    fn rust_to_python(&self, source: &str) -> Result<String, String> {
        let tree = parse_source(source, LanguageId::Rust).ok_or("Failed to parse Rust source")?;
        let mut output = String::new();
        output.push_str("# Auto-translated from Rust to Python\n");
        output.push_str("# Review and verify before use\n\n");

        let root = tree.root_node();
        let mut fns: Vec<tree_sitter::Node> = Vec::new();
        collect_fn_items(root, &mut fns);
        let mut translated = 0usize;
        for child in fns {
            let mut fc = child.walk();
            let mut name = String::from("fn");
            let mut params: Vec<(String, String)> = Vec::new();
            let mut ret_type: Option<String> = None;
            let mut in_ret = false;
            for sc in child.children(&mut fc) {
                match sc.kind() {
                    "identifier" => name = source[sc.byte_range()].to_string(),
                    "parameters" => {
                        params = self.translate_params(sc, source);
                    }
                    "->" => in_ret = true,
                    "primitive_type" | "type_identifier" | "generic_type"
                    | "scoped_type_identifier" | "array_type" | "reference_type" | "pointer_type"
                        if in_ret =>
                    {
                        ret_type = Some(self.map_rust_type(source[sc.byte_range()].trim()));
                        in_ret = false;
                    }
                    _ => {}
                }
            }
            let mut sig = format!("def {}(", name);
            sig.push_str(
                &params
                    .iter()
                    .map(|(n, t)| format!("{}: {}", n, t))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            sig.push(')');
            if let Some(rt) = ret_type {
                sig.push_str(&format!(" -> {}", rt));
            }
            sig.push_str(":\n    ...  # translated from Rust\n\n");
            output.push_str(&sig);
            translated += 1;
        }

        if translated == 0 {
            return Err("No functions found in Rust source".to_string());
        }
        Ok(output)
    }

    fn translate_params(&self, params_node: tree_sitter::Node, source: &str) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let mut c = params_node.walk();
        for p in params_node.children(&mut c) {
            let kind = p.kind();
            // Rust: pattern (identifier/self) + type_identifier / generics / reference_type
            let mut id = String::new();
            let mut ty = String::from("Any");
            let mut pc = p.walk();
            for cc in p.children(&mut pc) {
                match cc.kind() {
                    "identifier" => id = source[cc.byte_range()].to_string(),
                    "self" => id = "self".to_string(),
                    "type_identifier" | "primitive_type" => {
                        ty = self.map_rust_type(source[cc.byte_range()].trim());
                    }
                    "reference_type" | "pointer_type" | "generic_type" | "scoped_type_identifier"
                    | "array_type" => {
                        let txt = source[cc.byte_range()].trim().to_string();
                        let mapped = self.map_rust_type(&txt);
                        if !mapped.starts_with("&") {
                            ty = mapped;
                        }
                    }
                    "mutable_specifier" => {}
                    _ => {}
                }
            }
            if kind == "self_parameter" || id == "self" {
                out.push(("self".to_string(), "Any".to_string()));
            } else if !id.is_empty() {
                out.push((id, ty));
            }
        }
        out
    }

    fn map_rust_type(&self, t: &str) -> String {
        let t = t.trim();
        let t = t.trim_start_matches("->").trim();
        let core = t
            .trim_start_matches("&")
            .trim_start_matches("mut ")
            .trim()
            .trim_start_matches("Option<")
            .trim_end_matches('>');
        let base = match core {
            "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "usize"
            | "isize" => "int",
            "f32" | "f64" => "float",
            "String" | "str" => "str",
            "bool" => "bool",
            "Vec" | "VecDeque" | "LinkedList" | "&[T]" | "[T]" => "list",
            "HashMap" | "BTreeMap" | "map" => "dict",
            "HashSet" | "BTreeSet" | "set" => "set",
            "Result" => "Any",
            other if !other.is_empty() => "Any",
            _ => "Any",
        };
        if t.contains("Option") {
            return format!("{} | None", base);
        }
        base.to_string()
    }

    fn rust_to_c(&self, source: &str) -> Result<String, String> {
        let tree = parse_source(source, LanguageId::Rust).ok_or("Failed to parse Rust source")?;
        let functions = collect_function_names(&tree, source, LanguageId::Rust);
        let mut output = String::new();
        output.push_str("// Auto-translated from Rust to C\n");
        output.push_str("// Review and verify before use\n\n");

        for fn_name in &functions {
            output.push_str(&format!(
                "void {}(/* params */) {{\n    // translated from Rust\n}}\n\n",
                fn_name
            ));
        }

        if output.len() < 80 {
            return Err("Translation produced minimal output".to_string());
        }
        Ok(output)
    }

    fn extract_parens(&self, s: &str, _keyword: &str) -> String {
        let start = s.find('(').unwrap_or(0);
        let rest = &s[start..];
        let mut depth = 0;
        let mut end = 0;
        for (i, c) in rest.char_indices() {
            if c == '(' {
                depth += 1;
            } else if c == ')' {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
        }
        if end > 0 {
            rest[1..end].to_string()
        } else {
            rest.to_string()
        }
    }
}

fn collect_fn_items<'a>(node: tree_sitter::Node<'a>, out: &mut Vec<tree_sitter::Node<'a>>) {
    let mut c = node.walk();
    for child in node.children(&mut c) {
        if child.kind() == "function_item" {
            out.push(child);
        } else {
            collect_fn_items(child, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_c_to_rust() {
        let translator = CodeTranslator::new();
        let source = r#"
int add(int a, int b) {
    return a + b;
}
"#;
        let result = translator.translate(source, LanguageId::C, LanguageId::Rust);
        assert!(result.is_ok(), "Translation failed: {:?}", result.err());
        let output = result.unwrap();
        assert!(output.contains("fn add("));
        assert!(output.contains("a: i32"));
        assert!(output.contains("b: i32"));
        assert!(output.contains("-> i32"));
    }

    #[test]
    fn test_translate_python_to_rust() {
        let translator = CodeTranslator::new();
        let source = r#"
def greet(name):
    print("Hello, " + name)
    return name
"#;
        let result = translator.translate(source, LanguageId::Python, LanguageId::Rust);
        assert!(result.is_ok(), "Translation failed: {:?}", result.err());
        let output = result.unwrap();
        assert!(output.contains("fn greet("));
        assert!(output.contains("println!"));
    }

    #[test]
    fn test_unsupported_pair() {
        let translator = CodeTranslator::new();
        let result = translator.translate("fn x() {}", LanguageId::Rust, LanguageId::Go);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_rust_to_python() {
        let translator = CodeTranslator::new();
        let source = r#"
pub fn add(a: i32, b: u64) -> f64 {
    (a as f64) + (b as f64)
}

impl Counter {
    pub fn new(limit: usize) -> Self {
        Counter { limit }
    }
}
"#;
        let result = translator.translate(source, LanguageId::Rust, LanguageId::Python);
        assert!(result.is_ok(), "Translation failed: {:?}", result.err());
        let output = result.unwrap();
        assert!(output.contains("def add(a: int, b: int) -> float:"));
        assert!(output.contains("def new(limit: int) -> Any:"));
    }
}
