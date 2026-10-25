// syntax_guard.rs — Syntax Loss на декоде (Compiler-in-the-Loop, 25.10).
//
// Продолжение v8 (02fd4d8): там tree-sitter C считал ERROR-узлы ДЕКОДА
// как метрику («логирование»); отложенный шаг «градиентный штраф (демоция)
// за errors>0, не только метрика» реализован здесь для Python.
//
// Механика: на каждом шаге генерации кандидаты байтов (топ-N по скору)
// дописываются к текущему код-сегменту, сегмент парсится tree-sitter-python,
// считается число ERROR+missing узлов. Кандидат, увеличивающий число ошибок
// относительно лучшей без него альтернативы, штрафуется (score -= pen ×
// прирост). Цель — пред-регистрированный критерий v1.1: code% ≥ 5% на
// PY-сидах (и cyr ≤ 30%).
//
// Честные допущения:
//  - незавершённый код ВСЕГДА имеет ошибки в хвосте (EOF) → сравниваем
//    ОТНОСИТЕЛЬНЫЙ прирост ошибок кандидата, не абсолют;
//  - парсится только текущий код-сегмент (с последнего маркера [CODE] или
//    начала сида, обрезка до 4KB) — не вся история;
//  - stream может рвать 2-байтовые UTF-8 → перед парсингом хвост обрезается
//    до границы валидного UTF-8 (иначе tree-sitter читает мусор как ERROR).
use tree_sitter::Parser;

/// Число ERROR+missing узлов AST python для байтового среза.
/// Некорректный UTF-8 хвост обрезается до границы символа.
pub fn python_error_count(src: &[u8]) -> usize {
    let clean = trim_to_utf8_boundary(src);
    let mut parser = Parser::new();
    if parser.set_language(&tree_sitter_python::LANGUAGE.into()).is_err() {
        return 0; // грамматика недоступна — гард молча выключается
    }
    match parser.parse(&clean[..], None) {
        Some(tree) => {
            let mut errs = 0usize;
            let mut cursor = tree.walk();
            loop {
                let node = cursor.node();
                if node.is_error() || node.is_missing() {
                    errs += 1;
                }
                if cursor.goto_first_child() {
                    continue;
                }
                while !cursor.goto_next_sibling() {
                    if !cursor.goto_parent() {
                        return errs;
                    }
                }
            }
        }
        None => 0,
    }
}

/// Обрезка хвоста до границы валидного UTF-8 (частичный многобайтовый
/// символ в конце потока не должен считаться синтаксической ошибкой).
pub fn trim_to_utf8_boundary(src: &[u8]) -> &[u8] {
    match std::str::from_utf8(src) {
        Ok(_) => src,
        Err(e) => &src[..e.valid_up_to()],
    }
}

/// SyntaxGuard: инкрементальная демоция кандидатов на шаге декода.
pub struct SyntaxGuard {
    pub enabled: bool,
    pub penalty: f32,
    pub max_segment: usize,
}

impl SyntaxGuard {
    pub fn new(enabled: bool, penalty: f32) -> Self {
        SyntaxGuard {
            enabled,
            penalty,
            max_segment: 4096,
        }
    }

    /// Текущий код-сегмент: хвост state с последней границы «кода»
    /// (маркер [CODE], \n после пустой строки — упрощённо: последние
    /// max_segment байт; сегмент — локальный синтаксис, не вся программа).
    fn segment<'a>(&self, state: &'a [u8]) -> &'a [u8] {
        let lo = state.len().saturating_sub(self.max_segment);
        &state[lo..]
    }

    /// Демоция: для каждого кандидата (топ по скору) — прирост ошибок
    /// segment+cand против минимума среди проверенных. Кандидаты с
    /// положительным приростом получают score -= pen·delta.
    /// cands отсортированы по убыванию скора (вход/выход).
    pub fn demote(&self, state: &[u8], cands: &mut [(u8, f32)], top_n: usize) {
        if !self.enabled || self.penalty <= 0.0 || cands.is_empty() {
            return;
        }
        let seg = self.segment(state);
        let n = top_n.min(cands.len());
        let mut base: Option<usize> = None;
        let mut deltas: Vec<(usize, usize)> = Vec::with_capacity(n); // (idx, err)
        for (i, (b, _)) in cands.iter().enumerate().take(n) {
            let mut trial = Vec::with_capacity(seg.len() + 1);
            trial.extend_from_slice(seg);
            trial.push(*b);
            let e = python_error_count(&trial);
            deltas.push((i, e));
            if base.is_none() || e < base.unwrap() {
                base = Some(e);
            }
        }
        let base = base.unwrap_or(0);
        for (i, e) in deltas {
            if e > base {
                cands[i].1 -= self.penalty * ((e - base) as f32);
            }
        }
        // пересортировка после демоции
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_python_has_zero_errors() {
        assert_eq!(python_error_count(b"def f():\n    return 1\n"), 0);
        assert_eq!(python_error_count(b"import json, os\nx = 4\n"), 0);
        assert_eq!(python_error_count(b"for i in range(4):\n    print(i)\n"), 0);
    }

    #[test]
    fn broken_python_has_errors() {
        assert!(python_error_count(b"def f(:\n    return 1\n") > 0);
        assert!(python_error_count(b"x = = 4\n") > 0);
        assert!(python_error_count(b"def f():\n return 1\n  )}\n") > 0);
    }

    #[test]
    fn partial_utf8_tail_does_not_break_parse() {
        // 'ы' = D1 8B; обрезанный D1 — невалидный UTF-8, границей режется
        let src = b"x = 1\n\xd1";
        assert_eq!(python_error_count(src), python_error_count(b"x = 1\n"));
    }

    #[test]
    fn demote_penalizes_syntax_breaking_candidates() {
        let g = SyntaxGuard::new(true, 0.5);
        // хвост «x = 1» + '(' → валидное продолжение; + '=' → x = 1= ошибка
        let state = b"def f():\n    x = 1".to_vec();
        let mut cands = vec![(b'(', 0.90f32), (b'=', 0.89f32), (b')', 0.88f32)];
        g.demote(&state, &mut cands, 3);
        // '=' должен был потерять первенство (прирост ошибок у '='/')')
        assert_ne!(cands[0].0, b'=', "кандидат, ломающий синтаксис, должен упасть");
    }

    #[test]
    fn disabled_guard_is_noop() {
        let g = SyntaxGuard::new(false, 0.5);
        let state = b"def f(:\n".to_vec();
        let mut cands = vec![(b'a', 0.9f32), (b'b', 0.8f32)];
        g.demote(&state, &mut cands, 2);
        assert_eq!(cands, vec![(b'a', 0.9f32), (b'b', 0.8f32)]);
    }
}
