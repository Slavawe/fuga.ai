use std::collections::HashMap;

pub struct Canonicalizer;

impl Canonicalizer {
    pub fn dedup(fragments: &[String]) -> Vec<String> {
        let mut seen: HashMap<&str, usize> = HashMap::new();
        let mut result: Vec<Option<&str>> = vec![None; fragments.len()];

        for (i, frag) in fragments.iter().enumerate() {
            let sig = frag.lines().next().unwrap_or("").trim();
            if let Some(&prev) = seen.get(sig) {
                result[prev] = None;
            }
            seen.insert(sig, i);
            result[i] = Some(frag);
        }

        result.into_iter().flatten().map(String::from).collect()
    }
}
