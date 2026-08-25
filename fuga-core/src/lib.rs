pub mod filter;
pub mod ibm_model;
pub mod symbolic_eval;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;
use tree_sitter::{Node, Parser};

const HYPER_DIM_U64: usize = 32; // 32 * 64 bits = 2048-bit hypervectors
const HYPER_DIM_BITS: usize = 2048;

const LANG_PYTHON: i32 = 0;
const LANG_C: i32 = 1;

fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic hypervector for a node kind: same kind -> same HV in every
/// process and thread (no shared cache needed for rayon parallelism).
fn kind_hv(node_type: &str) -> Vec<u64> {
    let mut rng = fnv1a(node_type.as_bytes());
    (0..HYPER_DIM_U64).map(|_| splitmix64(&mut rng)).collect()
}

fn bind(v1: &[u64], v2: &[u64]) -> Vec<u64> {
    v1.iter().zip(v2.iter()).map(|(&a, &b)| a ^ b).collect()
}

fn permute_bit_rotate(v: &[u64], shift: usize) -> Vec<u64> {
    let mut out = vec![0u64; HYPER_DIM_U64];
    let word_shift = (shift / 64) % HYPER_DIM_U64;
    let bit_shift = shift % 64;
    for (i, &word) in v.iter().enumerate() {
        let target = (i + word_shift) % HYPER_DIM_U64;
        if bit_shift == 0 {
            out[target] ^= word;
        } else {
            out[target] ^= word << bit_shift;
            let next = (target + 1) % HYPER_DIM_U64;
            out[next] ^= word >> (64 - bit_shift);
        }
    }
    out
}

fn bundle_majority(children: &[Vec<u64>]) -> Vec<u64> {
    // Per-bit majority vote; ties break to 1 (matches the Python encoder).
    let n = children.len();
    let mut bundled = vec![0u64; HYPER_DIM_U64];
    for w in 0..HYPER_DIM_U64 {
        let mut cnt = [0u32; 64];
        for c in children {
            let word = c[w];
            for b in 0..64 {
                cnt[b] += ((word >> b) & 1) as u32;
            }
        }
        let mut out = 0u64;
        for b in 0..64 {
            if (cnt[b] as usize) * 2 >= n {
                out |= 1 << b;
            }
        }
        bundled[w] = out;
    }
    bundled
}

fn encode_node(node: Node, _code: &[u8]) -> Vec<u64> {
    let base_hv = kind_hv(node.kind());
    let n_children = node.child_count() as usize;
    if n_children == 0 {
        return base_hv;
    }

    let mut cursor = node.walk();
    let children: Vec<Node> = node.children(&mut cursor).collect();
    let rotated: Vec<Vec<u64>> = children
        .iter()
        .enumerate()
        .map(|(i, child)| {
            let hv = encode_node(*child, _code);
            permute_bit_rotate(&hv, i + 1)
        })
        .collect();

    bind(&base_hv, &bundle_majority(&rotated))
}

fn make_parser(lang_id: i32) -> Result<Parser, PyErr> {
    let mut parser = Parser::new();
    match lang_id {
        LANG_PYTHON => {
            let lang: tree_sitter::Language = tree_sitter_python::LANGUAGE.into();
            parser
                .set_language(&lang)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
        }
        LANG_C => {
            let lang: tree_sitter::Language = tree_sitter_c::LANGUAGE.into();
            parser
                .set_language(&lang)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
        }
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown language id: {other}"
            )))
        }
    }
    Ok(parser)
}

#[pyclass]
pub struct RustVSAEncoder;

#[pymethods]
impl RustVSAEncoder {
    #[new]
    fn new() -> Self {
        RustVSAEncoder
    }

    /// Batch AST->HV encoding on rayon threads.
    /// Returns float32 [B, 2048] with values in {-1, 1}.
    #[pyo3(signature = (code_list, lang=LANG_PYTHON))]
    fn encode_batch_py<'py>(
        &self,
        py: Python<'py>,
        code_list: Vec<String>,
        lang: i32,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        make_parser(lang)?;
        let b = code_list.len();
        let flat: Vec<f32> = py
            .allow_threads(|| {
                let rows: Vec<Vec<f32>> = code_list
                    .into_par_iter()
                    .map(|code| {
                        let mut parser =
                            make_parser(lang).expect("parser validated before threads");
                        let bytes = code.as_bytes();
                        let hv = match parser.parse(bytes, None) {
                            Some(tree) => encode_node(tree.root_node(), bytes),
                            None => vec![0u64; HYPER_DIM_U64],
                        };
                        packed_to_f32(&hv)
                    })
                    .collect();
                let mut flat = Vec::with_capacity(rows.len() * HYPER_DIM_BITS);
                for row in rows {
                    flat.extend(row);
                }
                flat
            });
        ndarray::Array2::from_shape_vec((b, HYPER_DIM_BITS), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))
            .map(|arr| arr.into_pyarray(py))
    }

    /// Compact storage path: returns uint64 words [B, 32] (roadmap: unpack to
    /// floats only right before the Fast-KAN pass).
    #[pyo3(signature = (code_list, lang=LANG_PYTHON))]
    fn encode_batch_packed<'py>(
        &self,
        py: Python<'py>,
        code_list: Vec<String>,
        lang: i32,
    ) -> PyResult<Bound<'py, PyArray2<u64>>> {
        make_parser(lang)?;
        let b = code_list.len();
        let flat: Vec<u64> = py
            .allow_threads(|| {
                let rows: Vec<Vec<u64>> = code_list
                    .into_par_iter()
                    .map(|code| {
                        let mut parser =
                            make_parser(lang).expect("parser validated before threads");
                        let bytes = code.as_bytes();
                        match parser.parse(bytes, None) {
                            Some(tree) => encode_node(tree.root_node(), bytes),
                            None => vec![0u64; HYPER_DIM_U64],
                        }
                    })
                    .collect();
                let mut flat = Vec::with_capacity(rows.len() * HYPER_DIM_U64);
                for row in rows {
                    flat.extend(row);
                }
                flat
            });
        ndarray::Array2::from_shape_vec((b, HYPER_DIM_U64), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))
            .map(|arr| arr.into_pyarray(py))
    }

    fn hyper_dim(&self) -> usize {
        HYPER_DIM_BITS
    }
}

fn packed_to_f32(hv: &[u64]) -> Vec<f32> {
    let mut out = vec![-1.0f32; HYPER_DIM_BITS];
    for (bit_idx, slot) in out.iter_mut().enumerate() {
        if (hv[bit_idx / 64] >> (bit_idx % 64)) & 1 == 1 {
            *slot = 1.0;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// HybridBinder: структурный VSA-уровень для полного rust+python гибрида.
// Всё битовое (bind/rotate/unbind/hamming) живёт здесь на packed uint64;
// Python получает только готовые массивы и скоры.
// ---------------------------------------------------------------------------

fn rotate_words(v: &[u64], words: usize, shift: usize) -> Vec<u64> {
    let word_shift = (shift / 64) % words;
    let bit_shift = shift % 64;
    let mut out = vec![0u64; words];
    for (i, &word) in v.iter().enumerate().take(words) {
        let target = (i + word_shift) % words;
        if bit_shift == 0 {
            out[target] ^= word;
        } else {
            out[target] ^= word << bit_shift;
            let next = (target + 1) % words;
            out[next] ^= word >> (64 - bit_shift);
        }
    }
    out
}

fn bundle_bits(vectors: &[Vec<u64>], words: usize) -> Vec<u64> {
    // Побитовое большинство; ничья → 1. Векторизовано по словам через
    // попарные суммы битов (bitsum trick: считаем единицы на позицию).
    let n = vectors.len();
    if n == 0 {
        return vec![0u64; words];
    }
    // Побитовые счётчики через вертикальную сумму (n мал, O(n*words)).
    let mut bundled = vec![0u64; words];
    for w in 0..words {
        let mut cnt = [0u8; 64];
        for v in vectors {
            let word = v[w];
            for b in 0..64 {
                cnt[b] += ((word >> b) & 1) as u8;
            }
        }
        let mut out = 0u64;
        for b in 0..64 {
            if (cnt[b] as usize) * 2 >= n {
                out |= 1 << b;
            }
        }
        bundled[w] = out;
    }
    bundled
}

#[pyclass]
pub struct HybridBinder {
    words: usize,
    bits: usize,
}

#[pymethods]
impl HybridBinder {
    #[new]
    #[pyo3(signature = (bits=2048))]
    fn new(bits: usize) -> Self {
        let words = (bits / 64).max(1);
        HybridBinder { words, bits: words * 64 }
    }

    /// Детерминированный HV элемента по имени (одинаков во всех процессах).
    fn hv_of<'py>(&self, py: Python<'py>, item: &str)
        -> PyResult<Bound<'py, PyArray1<u64>>>
    {
        let mut rng = fnv1a(item.as_bytes());
        let hv: Vec<u64> = (0..self.words).map(|_| splitmix64(&mut rng)).collect();
        Ok(ndarray::Array1::from(hv).into_pyarray(py))
    }

    /// Структурное предложение: XOR-бандл(word_i ⊗ rotate(pos_i)) по батчу.
    /// items: [B][N_i]; позиции — индекс элемента в последовательности.
    /// Возвращает packed uint64 [B, words].
    fn bind_batch<'py>(
        &self,
        py: Python<'py>,
        items: Vec<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<u64>>> {
        let b = items.len();
        let flat: Vec<u64> = py.allow_threads(|| {
            let rows: Vec<Vec<u64>> = items
                .into_par_iter()
                .map(|sentence| self.bind_sentence(&sentence))
                .collect();
            let mut flat = Vec::with_capacity(b * self.words);
            for row in rows {
                flat.extend(row);
            }
            flat
        });
        ndarray::Array2::from_shape_vec((b, self.words), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))
            .map(|arr| arr.into_pyarray(py))
    }

    /// Развязка: обратная ротация позиции pos у каждого hv батча.
    /// Ротация НЕ самоинверсна (в отличие от XOR с позиционным вектором),
    /// поэтому крутим назад на bits - pos.
    /// hv: [B, words] -> [B, words].
    fn unbind_batch<'py>(
        &self,
        py: Python<'py>,
        hv: &Bound<'py, PyArray2<u64>>,
        pos: usize,
    ) -> PyResult<Bound<'py, PyArray2<u64>>> {
        let view = unsafe { hv.as_array() };
        let b = view.shape()[0];
        let inv = self.bits - (pos % self.bits);
        let mut flat = Vec::with_capacity(b * self.words);
        for row in view.rows() {
            let owned: Vec<u64> = row.iter().copied().collect();
            flat.extend(rotate_words(&owned, self.words, inv));
        }
        ndarray::Array2::from_shape_vec((b, self.words), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))
            .map(|arr| arr.into_pyarray(py))
    }

    /// Хэмминг-скор каждого кандидата против запроса (доля совпавших бит).
    /// query: [B, words]; candidates: список имён. Возвращает f32 [B, C].
    fn score_items<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyArray2<u64>>,
        candidates: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        // SAFETY: GIL удержан, других мутабельных ссылок на буфер нет.
        let view = unsafe { query.as_array() };
        let b = view.shape()[0];
        let cand_hvs: Vec<Vec<u64>> = candidates
            .iter()
            .map(|name| {
                let mut rng = fnv1a(name.as_bytes());
                (0..self.words).map(|_| splitmix64(&mut rng)).collect()
            })
            .collect();
        let mut flat = Vec::with_capacity(b * cand_hvs.len());
        for row in view.rows() {
            let q: Vec<u64> = row.iter().copied().collect();
            for c in &cand_hvs {
                let diff: u32 = q
                    .iter()
                    .zip(c.iter())
                    .map(|(&a, &b)| (a ^ b).count_ones())
                    .sum();
                let sim = (self.bits - diff as usize) as f32 / self.bits as f32;
                flat.push(sim);
            }
        }
        ndarray::Array2::from_shape_vec((b, cand_hvs.len()), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))
            .map(|arr| arr.into_pyarray(py))
    }

    fn bits(&self) -> usize {
        self.bits
    }

    /// VSA Cleanup Memory: денойзинг суперпозиции — ближайший чистый атом
    /// словаря по хэммингу + его канонический HV.
    fn cleanup<'py>(
        &self,
        py: Python<'py>,
        noisy: &Bound<'_, PyArray2<u64>>,
        candidates: Vec<String>,
    ) -> PyResult<(Vec<String>, Bound<'py, numpy::PyArray2<u64>>)> {
        use numpy::PyArrayMethods;
        // SAFETY: GIL удержан, буфер не мутируется.
        let view = unsafe { noisy.as_array() };
        let b = view.shape()[0];
        let cand_hvs: Vec<Vec<u64>> = candidates
            .iter()
            .map(|name| {
                let mut rng = fnv1a(name.as_bytes());
                (0..self.words).map(|_| splitmix64(&mut rng)).collect()
            })
            .collect();
        let mut best_names = Vec::with_capacity(b);
        let mut flat_clean = Vec::with_capacity(b * self.words);
        for row in view.rows() {
            let q: Vec<u64> = row.iter().copied().collect();
            let mut best: Option<(usize, u32)> = None;
            for (ci, c) in cand_hvs.iter().enumerate() {
                let diff: u32 = q
                    .iter()
                    .zip(c.iter())
                    .map(|(&a, &b)| (a ^ b).count_ones())
                    .sum();
                if best.map_or(true, |(_, bd)| diff < bd) {
                    best = Some((ci, diff));
                }
            }
            let ci = best.expect("non-empty candidates").0;
            best_names.push(candidates[ci].clone());
            flat_clean.extend(cand_hvs[ci].iter().copied());
        }
        let arr = ndarray::Array2::from_shape_vec((b, self.words), flat_clean)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
        Ok((best_names, arr.into_pyarray(py)))
    }

    /// Пословные слоты: [B, max_len, words], каждое слово повёрнуто на свою
    /// позицию (i+1). Пустые слоты (паддинг) — нулевые слова.
    #[pyo3(signature = (rows, max_len=12))]
    fn extract_word_hvs_batch<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<Vec<String>>,
        max_len: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray3<u64>>> {
        let b = rows.len();
        // Параллельно строим строки, затем склеиваем (flat не шарится между потоками).
        let rows_out: Vec<Vec<u64>> = py.allow_threads(|| {
            rows.par_iter()
                .map(|row| {
                    let mut line = vec![0u64; max_len * self.words];
                    for (si, tok) in row.iter().take(max_len).enumerate() {
                        let mut rng = fnv1a(tok.as_bytes());
                        let w: Vec<u64> =
                            (0..self.words).map(|_| splitmix64(&mut rng)).collect();
                        let rotated = rotate_words(&w, self.words, si + 1);
                        let off = si * self.words;
                        line[off..off + self.words].copy_from_slice(&rotated);
                    }
                    line
                })
                .collect()
        });
        let mut flat = Vec::with_capacity(b * max_len * self.words);
        for line in rows_out {
            flat.extend(line);
        }
        ndarray::Array3::from_shape_vec((b, max_len, self.words), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))
            .map(|arr| arr.into_pyarray(py))
    }
}

impl HybridBinder {
    fn bind_sentence(&self, tokens: &[String]) -> Vec<u64> {
        let parts: Vec<Vec<u64>> = tokens
            .iter()
            .enumerate()
            .map(|(i, tok)| {
                let mut rng = fnv1a(tok.as_bytes());
                let w: Vec<u64> = (0..self.words).map(|_| splitmix64(&mut rng)).collect();
                rotate_words(&w, self.words, i + 1)
            })
            .collect();
        bundle_bits(&parts, self.words)
    }
}

#[pymodule]
fn fuga_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustVSAEncoder>()?;
    m.add_class::<HybridBinder>()?;
    m.add_class::<filter::RustLinguisticFilter>()?;
    m.add_class::<filter::RustASTGrammarFilter>()?;
    m.add_class::<ibm_model::IbmModel1>()?;
    m.add_class::<symbolic_eval::SymbolicExecutor>()?;
    m.add_class::<FastVSA>()?;
    m.add_function(wrap_pyfunction!(packed_u64_to_f32, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// FastVSA: нативные битовые операции над packed состояниями (горячий путь).
// ---------------------------------------------------------------------------

/// Циклический сдвиг битов массива u64 (LSB-first внутри слов).
fn rotate_words_u64(v: &[u64], shift: usize) -> Vec<u64> {
    let words = v.len();
    let total = words * 64;
    let s = shift % total.max(1);
    let word_shift = s / 64;
    let bit_shift = s % 64;
    let mut out = vec![0u64; words];
    for i in 0..words {
        let src = v[(i + words - word_shift) % words]; // сдвиг вправо по словам
        if bit_shift == 0 {
            out[i] = src;
        } else {
            let lo = src << bit_shift;
            let hi_src = v[(i + words - word_shift - 1 + words) % words];
            let hi = hi_src >> (64 - bit_shift);
            out[i] = lo | hi;
        }
    }
    out
}

#[pyclass]
pub struct FastVSA {
    dim_bits: usize,
    dim_words: usize,
}

#[pymethods]
impl FastVSA {
    #[new]
    pub fn new(dim_bits: usize) -> Self {
        FastVSA { dim_bits, dim_words: dim_bits.div_ceil(64) }
    }

    /// Случайное packed состояние [dim_words] u64.
    pub fn random_state<'py>(
        &self, py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<u64>>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let v: Vec<u64> = (0..self.dim_words).map(|_| rng.gen()).collect();
        Ok(numpy::PyArray1::from_vec(py, v))
    }

    /// XOR-связывание (binding). a, b: [words].
    pub fn bind<'py>(&self, py: Python<'py>, a: &Bound<'_, numpy::PyArray1<u64>>,
                     b: &Bound<'_, numpy::PyArray1<u64>>)
        -> PyResult<Bound<'py, numpy::PyArray1<u64>>>
    {
        // SAFETY: только чтение под GIL.
        let av = unsafe { a.as_array() };
        let bv = unsafe { b.as_array() };
        let out: Vec<u64> = av.iter().zip(bv.iter()).map(|(&x, &y)| x ^ y).collect();
        Ok(numpy::PyArray1::from_vec(py, out))
    }

    /// Битовая ротация.
    pub fn rotate<'py>(&self, py: Python<'py>, x: &Bound<'_, numpy::PyArray1<u64>>,
                       shift: usize)
        -> PyResult<Bound<'py, numpy::PyArray1<u64>>>
    {
        use numpy::PyArrayMethods;
        // SAFETY: только чтение.
        let xv = unsafe { x.as_array() };
        let owned: Vec<u64> = xv.iter().copied().collect();
        Ok(numpy::PyArray1::from_vec(
            py,
            rotate_words_u64(&owned, shift),
        ))
    }

    /// Побитовое большинство (bundling) по списку состояний.
    pub fn bundle<'py>(&self, py: Python<'py>,
                       states: Vec<Vec<u64>>)
        -> PyResult<Bound<'py, numpy::PyArray1<u64>>>
    {
        let n = states.len().max(1);
        let mut bundled = vec![0u64; self.dim_words];
        for w in 0..self.dim_words {
            let mut cnt = [0u32; 64];
            for st in &states {
                let word = st.get(w).copied().unwrap_or(0);
                for b in 0..64 {
                    cnt[b] += ((word >> b) & 1) as u32;
                }
            }
            let mut out = 0u64;
            for b in 0..64 {
                if (cnt[b] as usize) * 2 >= n {
                    out |= 1 << b;
                }
            }
            bundled[w] = out;
        }
        Ok(numpy::PyArray1::from_vec(py, bundled))
    }

    fn dim_bits(&self) -> usize {
        self.dim_words * 64
    }
}

/// Свободная функция модуля: packed u64 -> ±1 f32.
#[pyfunction]
fn packed_u64_to_f32<'py>(py: Python<'py>, hv: &Bound<'_, numpy::PyArray1<u64>>)
    -> PyResult<Bound<'py, numpy::PyArray2<f32>>>
{
    use numpy::PyArrayMethods;
    // SAFETY: только чтение под GIL.
    let view = unsafe { hv.as_array() };
    let words = view.len();
    let mut out = vec![-1.0f32; words * 64];
    for (wi, &word) in view.iter().enumerate() {
        for b in 0..64 {
            if (word >> b) & 1 == 1 {
                out[wi * 64 + b] = 1.0;
            }
        }
    }
    let arr = ndarray::Array2::from_shape_vec((words, 64), out)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
    Ok(arr.into_pyarray(py))
}
