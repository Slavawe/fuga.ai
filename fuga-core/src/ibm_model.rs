use std::collections::HashMap;

use pyo3::prelude::*;

#[pyclass]
/// IBM Model-1: P(en | ru) через EM на параллельных предложениях.
/// Строки интернируются в u32-id один раз при загрузке — внутри EM только
/// целочисленные ключи (наивный вариант с HashMap<(String,String)> тонет
/// в аллокациях клонов).
pub struct IbmModel1 {
    // Интернинг
    ru_ids: HashMap<String, u32>,
    en_ids: HashMap<String, u32>,
    ru_strs: Vec<String>,
    en_strs: Vec<String>,
    // t[(en_id, ru_id)] = P(en | ru); индексация (en, ru) для s_total по en.
    t: HashMap<(u32, u32), f32>,
}

#[pymethods]
impl IbmModel1 {
    #[new]
    pub fn new() -> Self {
        IbmModel1 {
            ru_ids: HashMap::new(),
            en_ids: HashMap::new(),
            ru_strs: Vec::new(),
            en_strs: Vec::new(),
            t: HashMap::new(),
        }
    }

    /// EM-обучение на параллельном корпусе (уже токенизированные пары).
    #[pyo3(signature = (corpus, epochs=3))]
    pub fn train(
        &mut self,
        py: Python<'_>,
        corpus: Vec<(Vec<String>, Vec<String>)>,
        epochs: usize,
    ) -> usize {
        let _ = py;
        // 1) Интернинг
        let mut corpus_ids: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(corpus.len());
        for (ru, en) in &corpus {
            let rid: Vec<u32> = ru
                .iter()
                .map(|w| {
                    let next = self.ru_ids.len() as u32;
                    *self.ru_ids.entry(w.to_lowercase()).or_insert(next)
                })
                .collect();
            let eid: Vec<u32> = en
                .iter()
                .map(|w| {
                    let next = self.en_ids.len() as u32;
                    *self.en_ids.entry(w.to_lowercase()).or_insert(next)
                })
                .collect();
            corpus_ids.push((rid, eid));
        }
        self.ru_strs = vec![String::new(); self.ru_ids.len()];
        for (s, &id) in self.ru_ids.iter() {
            self.ru_strs[id as usize] = s.clone();
        }
        self.en_strs = vec![String::new(); self.en_ids.len()];
        for (s, &id) in self.en_ids.iter() {
            self.en_strs[id as usize] = s.clone();
        }

        // 2) Инициализация: равномерно по длине EN-предложения
        self.t.clear();
        for (rid, eid) in &corpus_ids {
            let init = 1.0 / eid.len().max(1) as f32;
            for &r in rid {
                for &e in eid {
                    self.t.entry((e, r)).or_insert(init);
                }
            }
        }

        // 3) EM-итерации (последовательные; для 10K пар узким местом не является)
        let mut count: HashMap<(u32, u32), f64> = HashMap::new();
        let mut total: HashMap<u32, f64> = HashMap::new();
        for _epoch in 0..epochs {
            count.clear();
            total.clear();
            for (rid, eid) in &corpus_ids {
                for &e in eid {
                    // s_total = сумма t(e|r) по всем r предложения
                    let mut s_total = 0.0f64;
                    for &r in rid {
                        s_total += *self.t.get(&(e, r)).unwrap_or(&1e-9) as f64;
                    }
                    if s_total <= 0.0 {
                        continue;
                    }
                    for &r in rid {
                        let p = *self.t.get(&(e, r)).unwrap_or(&1e-9) as f64;
                        let c = p / s_total;
                        *count.entry((e, r)).or_insert(0.0) += c;
                        *total.entry(r).or_insert(0.0) += c;
                    }
                }
            }
            // M-step
            for ((e, r), c) in count.iter() {
                let norm = total.get(r).copied().unwrap_or(1.0);
                self.t.insert((*e, *r), (*c / norm) as f32);
            }
        }
        self.t.len()
    }

    /// Soft-alignment матрица пары: [len_ru, len_en], строки нормированы
    /// (для каждого RU-слова распределение по EN-словам предложения).
    pub fn align_pair(
        &self,
        ru_sent: Vec<String>,
        en_sent: Vec<String>,
    ) -> Vec<Vec<f32>> {
        let n_r = ru_sent.len();
        let n_e = en_sent.len();
        let mut m = vec![vec![0f32; n_e.max(1)]; n_r];
        for (i, rw) in ru_sent.iter().enumerate() {
            let rid = match self.ru_ids.get(&rw.to_lowercase()) {
                Some(id) => *id,
                None => continue,
            };
            let mut row: Vec<f32> = en_sent
                .iter()
                .map(|ew| {
                    *self
                        .t
                        .get(&(self.en_ids.get(&ew.to_lowercase()).copied().unwrap_or(u32::MAX), rid))
                        .unwrap_or(&0.0)
                })
                .collect();
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
                m[i] = row;
            } else if n_e > 0 {
                let u = 1.0 / n_e as f32;
                m[i] = vec![u; n_e];
            }
        }
        m
    }

    /// Топ-K переводов слова (автономный словарь без GPU).
    pub fn translate_topk(&self, py: Python<'_>, ru_word: String, k: usize) -> Vec<(String, f32)> {
        let rid = match self.ru_ids.get(&ru_word.to_lowercase()) {
            Some(id) => *id,
            None => return Vec::new(),
        };
        let mut cands: Vec<(u32, f32)> = self
            .t
            .iter()
            .filter(|((_, r), _)| *r == rid)
            .map(|((e, _), p)| (*e, *p))
            .collect();
        let _ = py;
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(k);
        cands
            .into_iter()
            .map(|(e, p)| (self.en_strs[e as usize].clone(), p))
            .collect()
    }

    pub fn vocab_sizes(&self) -> (usize, usize) {
        (self.ru_ids.len(), self.en_ids.len())
    }
}
