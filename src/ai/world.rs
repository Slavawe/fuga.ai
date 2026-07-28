use crate::weaver::pattern_matcher::TokenInfo;
use crate::weaver::token_id;

pub const WIKI_API: &str = "https://en.wikipedia.org/w/api.php";

pub const SEED_TOPICS: &[&str] = &[
    "Physics", "Chemistry", "Biology", "Mathematics", "Astronomy",
    "Computer science", "Artificial intelligence", "Programming language",
    "Rust (programming language)", "Machine learning", "Neural network",
    "History of science", "Philosophy", "Logic", "Consciousness",
    "Earth", "Solar System", "Universe", "Evolution", "Genetics",
    "Thermodynamics", "Quantum mechanics", "Relativity", "Gravity",
    "Electricity", "Magnetism", "Atom", "Molecule", "Cell (biology)",
    "DNA", "Protein", "Enzyme", "Ecosystem", "Climate",
    "Ocean", "Atmosphere", "Geology", "Plate tectonics", "Volcano",
    "Human", "Brain", "Immune system",
    "Medicine", "Disease", "Vaccine", "Virus",
    "Bacteria", "Plant", "Animal",
    "Algebra", "Geometry", "Calculus", "Statistics",
    "Probability", "Number theory",
    "Computer", "Internet", "World Wide Web", "Data structure",
    "Algorithm", "Operating system", "Database", "Computer network",
    "Programming paradigm", "Object-oriented programming",
    "Functional programming", "Type system", "Compiler",
    "Software engineering", "Version control",
    "History", "Ancient history", "Middle Ages", "Renaissance",
    "Industrial Revolution", "World War I", "World War II",
    "Cold War", "Space Race", "Information Age",
    "Society", "Culture", "Language",
    "Literature", "Art", "Music",
    "Neuroscience", "Psychology",
    "Economics", "Democracy", "Law", "Ethics",
    "Metaphysics", "Epistemology",
    "Python (programming language)", "JavaScript", "Java (programming language)",
    "C (programming language)", "C++", "Go (programming language)",
    "TypeScript", "Haskell", "Lisp (programming language)",
    "SQL", "Assembly language", "HTML", "CSS",
    "Bash (Unix shell)", "Ruby (programming language)", "PHP",
    "Swift (programming language)", "Kotlin (programming language)",
    "Scala (programming language)", "Elixir (programming language)",
    "Clojure", "Erlang (programming language)", "Julia (programming language)",
    "MATLAB", "R (programming language)", "Lua (programming language)",
    "Dart (programming language)", "Zig (programming language)",
    "Biochemistry", "Molecular biology", "Organic chemistry",
    "Particle physics", "Nuclear physics", "Astrophysics", "Cosmology",
    "Paleontology", "Zoology", "Botany", "Microbiology", "Immunology",
    "Pharmacology", "Optics", "Acoustics", "Fluid dynamics",
    "Materials science", "Chaos theory", "Information theory",
    "Game theory", "Graph theory", "Topology",
    "Combinatorics", "Linear algebra", "Differential equation",
    "Statistical mechanics", "Electromagnetism",
    "Photosynthesis", "Natural selection", "Speciation",
    "Biome", "Cellular respiration", "Cell division",
    "Cryptography", "Computer security", "Parallel computing",
    "Distributed computing", "Quantum computing",
    "Cloud computing", "DevOps", "Microservices",
    "Design pattern", "Software architecture",
    "Concurrency (computer science)", "Exception handling",
    "Garbage collection (computer science)", "Memory management",
    "Regular expression", "Unicode",
];
pub fn fetch_wikipedia(title: &str) -> Result<(String, String), String> {
    let url = format!(
        "{}?action=query&titles={}&prop=extracts&explaintext=true&format=json&redirects=1",
        WIKI_API, urlencode(title)
    );

    let mut last_err = String::new();
    for attempt in 0..5 {
        if attempt > 0 {
            let delay = (attempt as u64) * 2;
            std::thread::sleep(std::time::Duration::from_secs(delay));
        }

        let resp = match ureq::get(&url).call() {
            Ok(r) => r,
            Err(ureq::Error::Status(429, _)) => {
                last_err = format!("rate limited (attempt {})", attempt + 1);
                continue;
            }
            Err(e) => {
                return Err(format!("HTTP error for '{}': {}", title, e));
            }
        };

        let json: serde_json::Value = match resp.into_json() {
            Ok(j) => j,
            Err(e) => return Err(format!("JSON error: {}", e)),
        };

        let pages = json["query"]["pages"].as_object()
            .ok_or_else(|| format!("No pages in response for '{}'", title))?;

        for (_id, page) in pages {
            if let Some(extract) = page["extract"].as_str() {
                let page_title = page["title"].as_str().unwrap_or(title);
                return Ok((page_title.to_string(), extract.to_string()));
            }
            if page.get("missing").is_some() {
                return Err(format!("Page '{}' not found on Wikipedia", title));
            }
        }
        return Err(format!("No extract for '{}'", title));
    }

    Err(format!("Failed after retries: {}", last_err))
}

pub fn fetch_random_articles(count: usize) -> Result<Vec<String>, String> {
    let url = format!(
        "{}?action=query&list=random&rnlimit={}&format=json&rnnamespace=0",
        WIKI_API, count.min(50)
    );

    let mut last_err = String::new();
    for attempt in 0..5 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_secs(attempt as u64));
        }

        let resp = match ureq::get(&url).call() {
            Ok(r) => r,
            Err(ureq::Error::Status(429, _)) => {
                last_err = format!("rate limited (attempt {})", attempt + 1);
                continue;
            }
            Err(e) => return Err(format!("HTTP error: {}", e)),
        };

        let json: serde_json::Value = match resp.into_json() {
            Ok(j) => j,
            Err(e) => return Err(format!("JSON error: {}", e)),
        };

        let items = json["query"]["random"].as_array()
            .ok_or("No random items")?;

        return Ok(items.iter()
            .filter_map(|item| item["title"].as_str().map(|s| s.to_string()))
            .filter(|t| !t.starts_with("List of") && !t.starts_with("Wikipedia:"))
            .collect());
    }

    Err(format!("Failed to fetch random articles: {}", last_err))
}

pub fn chunk_text(text: &str, max_tokens: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = (start + max_tokens).min(words.len());
        chunks.push(words[start..end].join(" "));
        start = end;
    }
    chunks
}

pub fn make_tokens(text: &str) -> Vec<TokenInfo> {
    text.split_whitespace().map(|w| TokenInfo {
        id: token_id(&w),
        text: w.to_string(),
    }).collect()
}

fn urlencode(s: &str) -> String {
    s.chars().map(|c| match c {
        'A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '.' | '-' => c.to_string(),
        ' ' => "%20".to_string(),
        c => {
            let bytes = c.to_string().into_bytes();
            bytes.iter().map(|&b| format!("%{:02X}", b)).collect()
        }
    }).collect()
}
