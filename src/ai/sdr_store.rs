use crate::ai::sdr::{SdrIndex, SdrVector, domain_sdr, encode_text, sparsify};
use crate::core::hypervector::Hypervector;

pub struct SdrStore {
    pub index: SdrIndex,
    pub code_domain: SdrVector,
    pub doc_domain: SdrVector,
}

impl SdrStore {
    pub fn new() -> Self {
        SdrStore {
            index: SdrIndex::new(),
            code_domain: domain_sdr("code"),
            doc_domain: domain_sdr("doc"),
        }
    }

    pub fn query(&self, text: &str, top_k: usize) -> Vec<(usize, f64, &str)> {
        let query = encode_text(text);
        self.index.search(&query, top_k)
    }

    pub fn query_cross(
        &self,
        text: &str,
        from_domain: &str,
        top_k: usize,
    ) -> Vec<(usize, f64, &str)> {
        let query = encode_text(text);
        let qdom = domain_sdr(from_domain);
        self.index
            .search_cross(&query, &qdom, &self.code_domain, top_k)
    }

    pub fn build_from_mem(path: &str, max_entries: usize) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
        let mmap =
            unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap {}: {}", path, e))?;
        let data = &mmap[..];
        let mut pos = 0usize;

        // Guard-ридеры (аудит 22.08): файл < 4 байт / total=0 / усечённый
        // хвост больше не паникуют (0/0 и slice out of range) — тихий выход
        // с пустым индексом или break.
        let rd_u32 = |data: &[u8], pos: &mut usize| -> Option<u32> {
            if *pos + 4 > data.len() {
                return None;
            }
            let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Some(v)
        };
        let total = match rd_u32(data, &mut pos) {
            Some(t) => t as usize,
            None => {
                println!("  SDR: empty/truncated {} — empty index", path);
                return Ok(SdrStore::new());
            }
        };
        // max(1): иначе total=0 или max_entries=0 дают деление 0/0 ниже.
        let cap = total.min(max_entries).max(1);
        println!("  SDR: {} total entries, capping at {}", total, cap);

        let stride = (total / cap).max(1);
        let mut store = SdrStore::new();
        store.index.nodes.reserve(cap.min(1 << 22));
        store.index.texts.reserve(cap.min(1 << 22));
        let mut count = 0usize;

        for i in 0..total {
            if i % 100000 == 0 && i > 0 {
                print!(
                    "\r  SDR: {}/{} entries scanned ({} stored)",
                    i, total, count
                );
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            let Some(dim) = rd_u32(data, &mut pos) else { break };
            let dim = dim as usize;
            let wc = (dim + 63) / 64;
            let vec_bytes = wc * 8;
            if pos + vec_bytes > data.len() {
                break;
            }
            let mut words = vec![0u64; wc];
            for j in 0..wc {
                words[j] =
                    u64::from_le_bytes(data[pos + j * 8..pos + (j + 1) * 8].try_into().unwrap());
            }
            pos += vec_bytes;
            let hv = Hypervector { dim, words };

            if i % stride == 0 && count < cap {
                let sdr = sparsify(&hv);
                store.index.nodes.push(sdr);
            }

            let Some(text_len) = rd_u32(data, &mut pos) else { break };
            let text_len = text_len as usize;
            if pos + text_len > data.len() {
                break;
            }
            let text = String::from_utf8(data[pos..pos + text_len].to_vec()).unwrap_or_default();
            pos += text_len;

            if i % stride == 0 && count < cap {
                let snippet: String = text.chars().take(80).collect();
                store.index.texts.push(snippet);
                count += 1;
            }

            let Some(doc_len) = rd_u32(data, &mut pos) else { break };
            let doc_len = doc_len as usize;
            if pos + doc_len + 4 > data.len() {
                break;
            }
            pos += doc_len;
            let Some(role_len) = rd_u32(data, &mut pos) else { break };
            let role_len = role_len as usize;
            if pos + role_len > data.len() {
                break;
            }
            pos += role_len;
        }
        println!(
            "\n  SDR: built index with {} nodes ({} scanned)",
            count, total
        );
        println!(
            "  SDR: avg popcount = {:.1}",
            if !store.index.nodes.is_empty() {
                store.index.nodes.iter().map(|n| n.popcount()).sum::<u32>() as f64
                    / store.index.nodes.len() as f64
            } else {
                0.0
            }
        );
        Ok(store)
    }

    pub fn search(&self, query: &SdrVector, top_k: usize) -> Vec<(usize, f64, &str)> {
        self.index.search(query, top_k)
    }
}
