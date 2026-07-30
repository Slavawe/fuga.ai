use crate::ai::sdr::{SdrVector, SdrIndex, sparsify, encode_text, domain_sdr};
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

    pub fn query_cross(&self, text: &str, from_domain: &str, top_k: usize) -> Vec<(usize, f64, &str)> {
        let query = encode_text(text);
        let qdom = domain_sdr(from_domain);
        self.index.search_cross(&query, &qdom, &self.code_domain, top_k)
    }

    pub fn build_from_mem(path: &str, max_entries: usize) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("open {}: {}", path, e))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("mmap {}: {}", path, e))?;
        let data = &mmap[..];
        let mut pos = 0usize;

        let total = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;
        let cap = total.min(max_entries);
        println!("  SDR: {} total entries, capping at {}", total, cap);

        let stride = (total / cap).max(1);
        let mut store = SdrStore::new();
        store.index.nodes.reserve(cap);
        store.index.texts.reserve(cap);
        let mut count = 0usize;

        for i in 0..total {
            if i % 100000 == 0 && i > 0 {
                print!("\r  SDR: {}/{} entries scanned ({} stored)", i, total, count);
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            if pos + 4 > data.len() { break; }
            let dim = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            let wc = (dim + 63) / 64;
            let vec_bytes = wc * 8;
            if pos + vec_bytes > data.len() { break; }
            let mut words = vec![0u64; wc];
            for j in 0..wc {
                words[j] = u64::from_le_bytes(data[pos+j*8..pos+(j+1)*8].try_into().unwrap());
            }
            pos += vec_bytes;
            let hv = Hypervector { dim, words };

            if i % stride == 0 && count < cap {
                let sdr = sparsify(&hv);
                store.index.nodes.push(sdr);
            }

            if pos + 4 > data.len() { break; }
            let text_len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + text_len > data.len() { break; }
            let text = String::from_utf8(data[pos..pos+text_len].to_vec())
                .unwrap_or_default();
            pos += text_len;

            if i % stride == 0 && count < cap {
                let snippet: String = text.chars().take(80).collect();
                store.index.texts.push(snippet);
                count += 1;
            }

            let doc_len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            pos += doc_len;
            let role_len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            pos += role_len;
        }
        println!("\n  SDR: built index with {} nodes ({} scanned)", count, total);
        println!("  SDR: avg popcount = {:.1}", 
            if !store.index.nodes.is_empty() {
                store.index.nodes.iter().map(|n| n.popcount()).sum::<u32>() as f64 / store.index.nodes.len() as f64
            } else { 0.0 }
        );
        Ok(store)
    }

    pub fn search(&self, query: &SdrVector, top_k: usize) -> Vec<(usize, f64, &str)> {
        self.index.search(query, top_k)
    }
}
