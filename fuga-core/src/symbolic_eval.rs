use pyo3::prelude::*;

#[pyclass]
/// Символьный исполнитель: строгая арифметика над слотами VSA-плана.
/// Латентный плановик выбирает ЧТО считать (оператор и слоты),
/// здесь гарантируется ТОЧНОСТЬ результата (никаких галлюцинаций чисел).
pub struct SymbolicExecutor;

fn tokenize_expr(expr: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = expr.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' => {
                chars.next();
            }
            '0'..='9' | '.' => {
                let mut num = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2.is_ascii_digit() || c2 == '.' {
                        num.push(c2);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let v: f64 = num
                    .parse()
                    .map_err(|_| format!("bad number: {num}"))?;
                tokens.push(Token::Num(v));
            }
            '^' => {
                tokens.push(Token::Op('^'));
                chars.next();
            }
            '+' | '-' | '*' | '/' | '(' | ')' => {
                tokens.push(Token::Op(c));
                chars.next();
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut name = String::new();
                while let Some(&c2) = chars.peek() {
                    if c2.is_alphanumeric() || c2 == '_' {
                        name.push(c2);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let v = match name.as_str() {
                    "pi" => std::f64::consts::PI,
                    "e" => std::f64::consts::E,
                    _ => return Err(format!("unknown identifier: {name}")),
                };
                tokens.push(Token::Num(v));
            }
            other => return Err(format!("unexpected char: {other}")),
        }
    }
    Ok(tokens)
}

#[derive(Clone, Debug)]
enum Token {
    Num(f64),
    Op(char),
}

/// Рекурсивный спуск: expr := term (('+'|'-') term)*
///               term  := factor (('*'|'/') factor)*
///               factor:= Num | '(' expr ')' | '-' factor
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expr(&mut self) -> Result<f64, String> {
        let mut v = self.term()?;
        while matches!(self.peek(), Some(Token::Op('+')) | Some(Token::Op('-'))) {
            let op = match self.next() {
                Some(Token::Op(c)) => c,
                _ => break,
            };
            let r = self.term()?;
            v = if op == '+' { v + r } else { v - r };
        }
        Ok(v)
    }

    fn term(&mut self) -> Result<f64, String> {
        let mut v = self.factor()?;
        while matches!(self.peek(), Some(Token::Op('*')) | Some(Token::Op('/'))) {
            let op = match self.next() {
                Some(Token::Op(c)) => c,
                _ => break,
            };
            let r = self.factor()?;
            if op == '*' {
                v *= r;
            } else {
                if r == 0.0 {
                    return Err("ZeroDivisionError".into());
                }
                v /= r;
            }
        }
        Ok(v)
    }

    fn factor(&mut self) -> Result<f64, String> {
        let v = self.primary()?;
        // правоассоциативная степень: a^b^c = a^(b^c)
        if matches!(self.peek(), Some(Token::Op('^'))) {
            self.next();
            let e = self.factor()?;
            return Ok(v.powf(e));
        }
        Ok(v)
    }

    fn primary(&mut self) -> Result<f64, String> {
        match self.next() {
            Some(Token::Num(v)) => Ok(v),
            Some(Token::Op('(')) => {
                let v = self.expr()?;
                match self.next() {
                    Some(Token::Op(')')) => Ok(v),
                    _ => Err("expected ')'".into()),
                }
            }
            Some(Token::Op('-')) => self.factor().map(|v| -v),
            other => Err(format!("unexpected token: {other:?}")),
        }
    }
}

#[pymethods]
impl SymbolicExecutor {
    #[new]
    pub fn new() -> Self {
        SymbolicExecutor
    }

    /// Одиночная операция над слотами плана.
    pub fn execute(&self, op: &str, a: f64, b: f64) -> PyResult<f64> {
        match op {
            "add" | "+" => Ok(a + b),
            "sub" | "-" => Ok(a - b),
            "mul" | "*" => Ok(a * b),
            "div" | "/" => {
                if b == 0.0 {
                    Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                        "division by zero",
                    ))
                } else {
                    Ok(a / b)
                }
            }
            "pow" => Ok(a.powf(b)),
            "mod" => {
                if b == 0.0 {
                    Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                        "modulo by zero",
                    ))
                } else {
                    Ok(a % b)
                }
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown op: {other}"
            ))),
        }
    }

    /// Полное выражение со скобками и приоритетами: "(48/2)+24*3".
    pub fn eval_expression(&self, _py: Python<'_>, expr: String) -> PyResult<f64> {
        let tokens = tokenize_expr(&expr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        if tokens.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("empty expression"));
        }
        let mut p = Parser { tokens, pos: 0 };
        let v = p.expr().map_err(pyo3::exceptions::PyValueError::new_err)?;
        if p.pos != p.tokens.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("trailing tokens"));
        }
        Ok(v)
    }
}
