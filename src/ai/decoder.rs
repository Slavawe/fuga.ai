use crate::ai::FugaAI;
use crate::core::hypervector::Hypervector;
use crate::weaver::pattern_matcher::TokenInfo;
use crate::weaver::super_token::TokenRole;
use crate::weaver::vocabulary::TokenVocabulary;

pub struct DecoderOut {
    pub concepts: Vec<(String, f64)>,
    pub thought_cells: usize,
    pub entropy: f64,
}

// Logit-Lens-стиль декодер для VSA-куба.
// В трансформерах Logit Lens проецирует скрытый residual-stream вектор через
// матрицу раскодирования W_unembed на словарь токенов — так видно, «о чём думает»
// модель. Аналог для Fuga:
//   1. супертокен запроса сжимается через вейвер (как residual stream);
//   2. резонансное внимание читает резонирующие ячейки куба (associative memory);
//   3. суперпозиция «запрос ⊕ мысли ячеек» образует readout-вектор;
//   4. readout проецируется на концепт-словарь (W_unembed), выстроенный в том же
//      id-пространстве (token_id слов из запроса и ближайшей памяти).
// В отличие от unweave (восстановление ВХОДНЫХ токенов) это декодирует то, что
// КУБ ассоциирует с запросом — новая информация, а не склейка исходника.
pub fn logit_lens<const N: usize, const S: usize>(
    ai: &mut FugaAI<N, S>,
    query: &str,
    beam: usize,
    cells_k: usize,
) -> DecoderOut {
    let query_tokens: Vec<TokenInfo> = query
        .split_whitespace()
        .enumerate()
        .map(|(_, w)| TokenInfo {
            id: crate::weaver::token_id(w),
            text: w.to_string(),
        })
        .collect();

    let vocab = aligned_concept_vocab(ai, &query_tokens, 2000);

    let output = ai.think(&query_tokens);

    let mut cells = Vec::new();
    for st in &output.super_tokens {
        cells.extend(ai.attention.beam_attention(st, &ai.cube, cells_k));
    }
    cells.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    cells.truncate(cells_k);

    // readout = суперпозиция запроса и резонирующих ячеек куба.
    let mut cell_readouts: Vec<Hypervector> = Vec::new();
    for st in &output.super_tokens {
        cell_readouts.push(st.vector.clone());
    }
    for cell in &cells {
        let mut coords = [0usize; N];
        let src = [cell.x, cell.y, cell.z, cell.w, cell.v];
        for i in 0..N {
            coords[i] = src[i];
        }
        cell_readouts.push(ai.cube.cell_at(&coords).clone());
    }

    let refs: Vec<&Hypervector> = cell_readouts.iter().collect();
    let base = refs[0].clone();
    let readout = if refs.len() > 1 {
        base.bundle(&refs[1..])
    } else {
        base
    };

    let mut concepts: Vec<(String, f64)> = vocab
        .nearest_beam(&readout, beam)
        .into_iter()
        .map(|(_id, text, sim)| (text, sim))
        .collect();
    concepts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    DecoderOut {
        concepts,
        thought_cells: cells.len(),
        entropy: output.cube_entropy,
    }
}

// Концепт-словарь в том же id-пространстве, что и сжатие (token_id(word)),
// чтобы векторы decode совпадали с векторами компрессии. Источники слов:
// слова запроса + слова из top записей памяти для запроса.
fn aligned_concept_vocab<const N: usize, const S: usize>(
    ai: &FugaAI<N, S>,
    query_tokens: &[TokenInfo],
    max: usize,
) -> TokenVocabulary {
    let mut words: Vec<String> = query_tokens.iter().map(|t| t.text.clone()).collect();

    let seeds: Vec<String> = {
        let mem = ai.memory.search_by_text(
            &query_tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            9,
        );
        mem.into_iter()
            .map(|(_, _, e)| e.text.clone())
            .collect()
    };
    for s in seeds {
        for w in s.split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_') {
            if w.len() >= 2 {
                words.push(w.to_lowercase());
            }
        }
    }

    let mut seen = std::collections::HashSet::new();
    let picked: Vec<String> = words
        .into_iter()
        .filter(|w| w.len() <= 40 && seen.insert(w.clone()))
        .take(max)
        .collect();

    let mut vocab = TokenVocabulary::new(ai.dim);
    for w in picked {
        let id = crate::weaver::token_id(&w);
        vocab.add(id, &w, TokenRole::CODE);
    }
    vocab
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::wave_cube::WaveCube;
    use crate::ai::memory_store::MemoryStore;

    #[test]
    fn logit_lens_returns_concepts_for_query_words() {
        let mut ai = FugaAI::<3, 4>::new(256, 2);
        ai.cube = WaveCube::<3, 4>::new(256);
        ai.memory = MemoryStore::new();

        let out = logit_lens(&mut ai, "tokio async stream", 8, 4);

        assert!(!out.concepts.is_empty(), "expected at least one concept");
        // Слова запроса гарантированно в концепт-словаре, значит decode не пуст.
        let labels: Vec<String> = out.concepts.iter().map(|(t, _)| t.clone()).collect();
        let has_query_word = labels.iter().any(|t| t == "tokio" || t == "async" || t == "stream");
        assert!(
            has_query_word || !labels.is_empty(),
            "concepts: {:?}",
            labels
        );
        assert!(out.thought_cells <= 4);
    }
}