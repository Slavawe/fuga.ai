use std::collections::HashMap;

use super::hypervector::Hypervector;

/// Пентагон (Пятиугольник) — внешнее гипервекторное хранилище долгосрочной памяти.
/// Вмещает сотни тысяч концептов в пространстве 100,000 измерений.
/// Использует расстояние Хэмминга (popcount) для мгновенного поиска релевантных векторов.
///
/// Активируется когда Волновой Куб испытывает дефицит: энтропия → 0 или когерентность падает.
pub struct PentagonStorage {
    pub dim: usize,
    /// Банк памяти: label → гипервектор
    pub memory_bank: HashMap<String, Hypervector>,
    /// Порог сходства для срабатывания fetch_on_deficit (Hamming distance)
    pub similarity_threshold: f64,
}

impl PentagonStorage {
    pub fn new(dim: usize, similarity_threshold: f64) -> Self {
        Self {
            dim,
            memory_bank: HashMap::new(),
            similarity_threshold,
        }
    }

    /// Запись концепта в Пентагон.
    pub fn store(&mut self, label: &str, hv: Hypervector) {
        self.memory_bank.insert(label.to_string(), hv);
    }

    /// Запрос при дефиците: Куб посылает query_vector,
    /// Пентагон ищет ближайший концепт по расстоянию Хэмминга.
    /// Возвращает (label, hypervector, similarity) если найден концепт выше порога.
    pub fn fetch_on_deficit(&self, query_vector: &Hypervector) -> Option<(&str, &Hypervector, f64)> {
        let mut best: Option<(&str, &Hypervector, f64)> = None;

        for (label, hv) in &self.memory_bank {
            let sim = query_vector.similarity(hv);
            if sim >= self.similarity_threshold {
                match best {
                    None => best = Some((label.as_str(), hv, sim)),
                    Some((_, _, best_sim)) if sim > best_sim => {
                        best = Some((label.as_str(), hv, sim));
                    }
                    _ => {}
                }
            }
        }
        best
    }

    /// Пакетный запрос: Куб посылает все свои диагональные ячейки как запросы,
    /// Пентагон возвращает список совпадений.
    pub fn batch_fetch(&self, queries: &[&Hypervector]) -> Vec<(String, f64)> {
        queries
            .iter()
            .filter_map(|q| self.fetch_on_deficit(q))
            .map(|(label, _, sim)| (label.to_string(), sim))
            .collect()
    }

    /// Размер банка памяти.
    pub fn size(&self) -> usize {
        self.memory_bank.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_fetch() {
        let mut p = PentagonStorage::new(10000, 0.6);
        let concept = Hypervector::random(10000);
        p.store("syntax_overflow_pattern", concept.clone());
        let found = p.fetch_on_deficit(&concept);
        assert!(found.is_some(), "Should find stored concept by its own vector");
        let (label, _, sim) = found.unwrap();
        assert_eq!(label, "syntax_overflow_pattern");
        assert!(sim > 0.99, "Self-similarity should be near 1.0, got {}", sim);
    }

    #[test]
    fn test_fetch_below_threshold() {
        let mut p = PentagonStorage::new(10000, 0.9);
        let a = Hypervector::random(10000);
        let b = Hypervector::random(10000);
        p.store("concept_a", a);
        let found = p.fetch_on_deficit(&b);
        assert!(found.is_none(), "Random vectors should not match above 0.9 threshold");
    }
}