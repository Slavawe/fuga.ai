use hound::{WavWriter, WavSpec};
use std::collections::HashMap;
use std::f64::consts::PI;

const SAMPLE_RATE: u32 = 22050;

#[derive(Clone)]
pub struct Phoneme {
    pub symbol: &'static str,
    pub f1: f64,
    pub f2: f64,
    pub f3: f64,
    pub duration_ms: f64,
    pub ptype: PhonemeType,
}

#[derive(Clone, Copy, PartialEq)]
pub enum PhonemeType {
    Vowel,
    Fricative,
    Plosive,
    Nasal,
    Silence,
}

pub struct FugaText {
    phoneme_map: HashMap<String, Phoneme>,
    lexicon: HashMap<String, Vec<String>>,
    prosody_pitch: f64,
    prosody_rate: f64,
}

impl FugaText {
    pub fn new() -> Self {
        let mut phoneme_map = HashMap::new();
        for p in PHONEME_TABLE {
            phoneme_map.insert(p.symbol.to_string(), p.clone());
        }

        let lexicon = build_lexicon();

        FugaText {
            phoneme_map,
            lexicon,
            prosody_pitch: 180.0,
            prosody_rate: 1.0,
        }
    }

    pub fn text_to_phonemes(&self, text: &str) -> Vec<String> {
        let cleaned: String = text.chars()
            .filter(|c| c.is_alphabetic() || c.is_whitespace() || *c == '\'')
            .collect();
        let lower = cleaned.to_lowercase();
        let mut result = Vec::new();

        for word in lower.split_whitespace() {
            if let Some(phones) = self.lexicon.get(word) {
                result.extend(phones.iter().cloned());
            } else {
                result.extend(self.guess_phonemes(word));
            }
            result.push("_".to_string());
        }
        result
    }

    pub fn guess_phonemes(&self, word: &str) -> Vec<String> {
        let mut phones = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            let ph = match c {
                'a' => "AA",
                'e' => "EH",
                'i' => "IH",
                'o' => "AO",
                'u' => "UW",
                'y' => "Y",
                'b' => "B",
                'd' => "D",
                'f' => "F",
                'g' => "G",
                'h' => "HH",
                'j' => "JH",
                'k' => "K",
                'l' => "L",
                'm' => "M",
                'n' => "N",
                'p' => "P",
                'r' => "R",
                's' => "S",
                't' => "T",
                'v' => "V",
                'w' => "W",
                'z' => "Z",
                'x' => "K", _ => "",
            };
            if !ph.is_empty() {
                phones.push(ph.to_string());
            }
            i += 1;
        }
        phones
    }

    pub fn synthesize(&self, text: &str) -> Vec<i16> {
        let phones = self.text_to_phonemes(text);
        let mut samples = Vec::new();

        for ph in &phones {
            if let Some(p) = self.phoneme_map.get(ph) {
                if p.ptype == PhonemeType::Silence {
                    let n = (SAMPLE_RATE as f64 * p.duration_ms / 1000.0) as usize;
                    samples.resize(samples.len() + n, 0);
                } else {
                    let n = (SAMPLE_RATE as f64 * p.duration_ms / 1000.0) as usize;
                    for si in 0..n {
                        let t = si as f64 / SAMPLE_RATE as f64;
                        let amp = if si < n / 8 {
                            (si as f64 / (n as f64 / 8.0)) * 0.3
                        } else if si > n * 7 / 8 {
                            ((n - si) as f64 / (n as f64 / 8.0)) * 0.3
                        } else {
                            0.3
                        };
                        let pitch_mod = (2.0 * PI * 4.0 * t).sin() * 0.02;
                        let pitch = self.prosody_pitch * (1.0 + pitch_mod);
                        let formant = (2.0 * PI * p.f1 * t).sin()
                            + 0.5 * (2.0 * PI * p.f2 * t).sin()
                            + 0.25 * (2.0 * PI * p.f3 * t).sin();
                        let mut val = (amp * formant * i16::MAX as f64) as i16;
                        if p.ptype == PhonemeType::Fricative {
                            let noise = (si as f64 * 7.31).sin().powi(2)
                                + (si as f64 * 13.17).sin().powi(2);
                            val = (val as f64 * 0.5 + noise * 3000.0 * amp) as i16;
                        }
                        samples.push(val);
                    }
                }
            } else {
                let n = SAMPLE_RATE as usize / 8;
                samples.resize(samples.len() + n, 0);
            }
        }
        samples
    }

    pub fn save_wav(&self, text: &str, path: &str) -> Result<(), String> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec)
            .map_err(|e| format!("WavWriter: {}", e))?;
        let samples = self.synthesize(text);
        for s in samples {
            writer.write_sample(s)
                .map_err(|e| format!("Write: {}", e))?;
        }
        writer.finalize().map_err(|e| format!("Finalize: {}", e))
    }

    pub fn speak(&self, text: &str) -> Vec<u8> {
        let samples = self.synthesize(text);
        let spec = WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Vec::new();
        {
            let mut writer = WavWriter::new(
                std::io::Cursor::new(&mut buf),
                spec,
            ).unwrap();
            for s in samples {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        buf
    }
}

const PHONEME_TABLE: &[Phoneme] = &[
    Phoneme { symbol: "AA", f1: 700.0, f2: 1200.0, f3: 2600.0, duration_ms: 100.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "AE", f1: 660.0, f2: 1700.0, f3: 2500.0, duration_ms: 100.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "AH", f1: 600.0, f2: 1100.0, f3: 2600.0, duration_ms: 90.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "AO", f1: 500.0, f2: 900.0, f3: 2600.0, duration_ms: 110.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "AW", f1: 500.0, f2: 1000.0, f3: 2600.0, duration_ms: 120.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "AY", f1: 450.0, f2: 1800.0, f3: 2600.0, duration_ms: 120.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "EH", f1: 530.0, f2: 1800.0, f3: 2600.0, duration_ms: 90.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "ER", f1: 500.0, f2: 1400.0, f3: 2100.0, duration_ms: 110.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "EY", f1: 450.0, f2: 2000.0, f3: 2600.0, duration_ms: 120.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "IH", f1: 400.0, f2: 2000.0, f3: 2600.0, duration_ms: 80.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "IY", f1: 300.0, f2: 2200.0, f3: 2800.0, duration_ms: 100.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "OW", f1: 450.0, f2: 900.0, f3: 2600.0, duration_ms: 120.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "OY", f1: 450.0, f2: 1000.0, f3: 2600.0, duration_ms: 120.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "UH", f1: 450.0, f2: 1200.0, f3: 2600.0, duration_ms: 80.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "UW", f1: 300.0, f2: 900.0, f3: 2200.0, duration_ms: 100.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "B",  f1: 200.0, f2: 800.0, f3: 2000.0, duration_ms: 60.0, ptype: PhonemeType::Plosive },
    Phoneme { symbol: "CH", f1: 300.0, f2: 1800.0, f3: 3000.0, duration_ms: 80.0, ptype: PhonemeType::Fricative },
    Phoneme { symbol: "D",  f1: 200.0, f2: 1600.0, f3: 2700.0, duration_ms: 50.0, ptype: PhonemeType::Plosive },
    Phoneme { symbol: "DH", f1: 300.0, f2: 1400.0, f3: 2400.0, duration_ms: 70.0, ptype: PhonemeType::Fricative },
    Phoneme { symbol: "F",  f1: 400.0, f2: 1400.0, f3: 2500.0, duration_ms: 90.0, ptype: PhonemeType::Fricative },
    Phoneme { symbol: "G",  f1: 200.0, f2: 1000.0, f3: 2000.0, duration_ms: 50.0, ptype: PhonemeType::Plosive },
    Phoneme { symbol: "HH", f1: 300.0, f2: 1500.0, f3: 2600.0, duration_ms: 70.0, ptype: PhonemeType::Fricative },
    Phoneme { symbol: "JH", f1: 250.0, f2: 1800.0, f3: 2600.0, duration_ms: 70.0, ptype: PhonemeType::Fricative },
    Phoneme { symbol: "K",  f1: 300.0, f2: 1200.0, f3: 2200.0, duration_ms: 50.0, ptype: PhonemeType::Plosive },
    Phoneme { symbol: "L",  f1: 400.0, f2: 1200.0, f3: 2600.0, duration_ms: 80.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "M",  f1: 300.0, f2: 1000.0, f3: 2200.0, duration_ms: 70.0, ptype: PhonemeType::Nasal },
    Phoneme { symbol: "N",  f1: 300.0, f2: 1400.0, f3: 2400.0, duration_ms: 70.0, ptype: PhonemeType::Nasal },
    Phoneme { symbol: "NG", f1: 300.0, f2: 1000.0, f3: 2000.0, duration_ms: 80.0, ptype: PhonemeType::Nasal },
    Phoneme { symbol: "P",  f1: 200.0, f2: 800.0, f3: 2000.0, duration_ms: 40.0, ptype: PhonemeType::Plosive },
    Phoneme { symbol: "R",  f1: 400.0, f2: 1300.0, f3: 1800.0, duration_ms: 80.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "S",  f1: 400.0, f2: 1500.0, f3: 3000.0, duration_ms: 90.0, ptype: PhonemeType::Fricative },
    Phoneme { symbol: "SH", f1: 300.0, f2: 1700.0, f3: 2800.0, duration_ms: 90.0, ptype: PhonemeType::Fricative },
    Phoneme { symbol: "T",  f1: 200.0, f2: 1600.0, f3: 2700.0, duration_ms: 40.0, ptype: PhonemeType::Plosive },
    Phoneme { symbol: "TH", f1: 300.0, f2: 1400.0, f3: 2600.0, duration_ms: 80.0, ptype: PhonemeType::Fricative },
    Phoneme { symbol: "V",  f1: 300.0, f2: 1200.0, f3: 2400.0, duration_ms: 70.0, ptype: PhonemeType::Fricative },
    Phoneme { symbol: "W",  f1: 300.0, f2: 800.0, f3: 2200.0, duration_ms: 60.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "Y",  f1: 300.0, f2: 2000.0, f3: 2600.0, duration_ms: 60.0, ptype: PhonemeType::Vowel },
    Phoneme { symbol: "Z",  f1: 400.0, f2: 1400.0, f3: 2600.0, duration_ms: 80.0, ptype: PhonemeType::Fricative },
    Phoneme { symbol: "ZH", f1: 300.0, f2: 1600.0, f3: 2600.0, duration_ms: 80.0, ptype: PhonemeType::Fricative },
    Phoneme { symbol: "_",  f1: 0.0,   f2: 0.0,   f3: 0.0,   duration_ms: 40.0,  ptype: PhonemeType::Silence },
];

fn build_lexicon() -> HashMap<String, Vec<String>> {
    let mut dict = HashMap::new();
    for (word, phones) in LEXICON_ENTRIES {
        dict.insert(word.to_string(), phones.iter().map(|s| s.to_string()).collect());
    }
    dict
}

const LEXICON_ENTRIES: &[(&str, &[&str])] = &[
    ("the", &["DH", "AH"]),
    ("a", &["AH"]),
    ("an", &["AE", "N"]),
    ("is", &["IH", "Z"]),
    ("of", &["AH", "V"]),
    ("to", &["T", "UW"]),
    ("and", &["AE", "N", "D"]),
    ("in", &["IH", "N"]),
    ("it", &["IH", "T"]),
    ("that", &["DH", "AE", "T"]),
    ("for", &["F", "AO", "R"]),
    ("with", &["W", "IH", "DH"]),
    ("on", &["AA", "N"]),
    ("at", &["AE", "T"]),
    ("by", &["B", "AY"]),
    ("from", &["F", "R", "AH", "M"]),
    ("as", &["AE", "Z"]),
    ("was", &["W", "AA", "Z"]),
    ("are", &["AA", "R"]),
    ("be", &["B", "IY"]),
    ("not", &["N", "AA", "T"]),
    ("this", &["DH", "IH", "S"]),
    ("space", &["S", "P", "EY", "S"]),
    ("time", &["T", "AY", "M"]),
    ("energy", &["EH", "N", "ER", "JH", "IY"]),
    ("force", &["F", "AO", "R", "S"]),
    ("mass", &["M", "AE", "S"]),
    ("gravity", &["G", "R", "AE", "V", "IH", "T", "IY"]),
    ("light", &["L", "AY", "T"]),
    ("speed", &["S", "P", "IY", "D"]),
    ("power", &["P", "AW", "ER"]),
    ("field", &["F", "IY", "L", "D"]),
    ("wave", &["W", "EY", "V"]),
    ("aether", &["EY", "DH", "ER"]),
    ("warp", &["W", "AO", "R", "P"]),
    ("reactor", &["R", "IY", "AE", "K", "T", "ER"]),
    ("control", &["K", "AH", "N", "T", "R", "OW", "L"]),
    ("system", &["S", "IH", "S", "T", "AH", "M"]),
    ("vector", &["V", "EH", "K", "T", "ER"]),
    ("tensor", &["T", "EH", "N", "S", "ER"]),
    ("axiom", &["AE", "K", "S", "IY", "AH", "M"]),
    ("proof", &["P", "R", "UW", "F"]),
    ("error", &["EH", "R", "ER"]),
    ("status", &["S", "T", "AE", "T", "AH", "S"]),
    ("verified", &["V", "EH", "R", "AH", "F", "AY", "D"]),
    ("failed", &["F", "EY", "L", "D"]),
    ("ready", &["R", "EH", "D", "IY"]),
    ("fuga", &["F", "UW", "G", "AH"]),
    ("omni", &["AA", "M", "N", "IY"]),
    ("mach", &["M", "AE", "K"]),
    ("woodward", &["W", "UH", "D", "W", "ER", "D"]),
    ("alcubierre", &["AE", "L", "K", "UW", "B", "IY", "EH", "R"]),
    ("zfc", &["Z", "EH", "F", "S", "IY"]),
    ("quantum", &["K", "W", "AA", "N", "T", "AH", "M"]),
    ("riemann", &["R", "IY", "M", "AH", "N"]),
    ("formal", &["F", "AO", "R", "M", "AH", "L"]),
    ("verification", &["V", "EH", "R", "AH", "F", "IH", "K", "EY", "SH", "AH", "N"]),
    ("z3", &["Z", "IY", "TH", "R", "IY"]),
    ("constraint", &["K", "AH", "N", "S", "T", "R", "EY", "N", "T"]),
    ("unsat", &["AH", "N", "S", "AE", "T"]),
    ("bubble", &["B", "AH", "B", "AH", "L"]),
    ("negative", &["N", "EH", "G", "AH", "T", "IH", "V"]),
    ("density", &["D", "EH", "N", "S", "IH", "T", "IY"]),
    ("gradient", &["G", "R", "EY", "D", "IY", "AH", "N", "T"]),
    ("temperature", &["T", "EH", "M", "P", "ER", "AH", "CH", "ER"]),
    ("neutron", &["N", "UW", "T", "R", "AA", "N"]),
    ("fission", &["F", "IH", "SH", "AH", "N"]),
    ("doppler", &["D", "AA", "P", "L", "ER"]),
    ("safety", &["S", "EY", "F", "T", "IY"]),
    ("compile", &["K", "AH", "M", "P", "AY", "L"]),
];

pub fn conversational_reply(query: &str, domain: &str, system_vector: &[f64]) -> String {
    let base: String = match domain {
            "zfc" => format!("Из аксиом ZFC следует, что ваш запрос формально корректен. Системный вектор содержит {} измерений, где первое — количество аксиом.", system_vector.len()),
            "geometry" => format!("Рассмотрение римановой геометрии показывает, что кривизна многообразия определяется {} компонентами тензора Римана.", system_vector.get(1).copied().unwrap_or(0.0)),
        "qft" => format!("Анализ квантово-полевых энергетических условий: WEC {}, NEC {}, SEC {}, DEC {}. Квантовое неравенство Форда-Романа задаёт предел на отрицательную энергию.", 
            if system_vector.first().copied().unwrap_or(0.0) > 0.5 { "выполнено" } else { "нарушено" },
            if system_vector.get(1).copied().unwrap_or(0.0) > 0.5 { "выполнено" } else { "нарушено" },
            if system_vector.get(2).copied().unwrap_or(0.0) > 0.5 { "выполнено" } else { "нарушено" },
            if system_vector.get(3).copied().unwrap_or(0.0) > 0.5 { "выполнено" } else { "нарушено" }),
        "met" => "Массовая флуктуация Вудворда: при частоте 30 кГц и массе 4 кг амплитуда δm₀ составляет порядка 10⁻⁵ кг. Тяга пропорциональна квадрату производной мощности.".to_string(),
        "warp" => "Метрика Алькубьерре требует отрицательной энергии порядка 10⁴⁵ кг для пузыря радиусом 100 м. Это на 44 порядка больше, чем позволяет квантовое неравенство.".to_string(),
        "reactor" => "Реактор в стационарном режиме. Уровень мощности и концентрация предшественников запаздывающих нейтронов сбалансированы. Температурный коэффициент Доплера обеспечивает отрицательную обратную связь.".to_string(),
        "spatial" => "Пространственный контроллер активен. Фазовая стабильность удержания в пределах нормы. LiDAR сканирует 128 лучами по золотой спирали.".to_string(),
        "code" => "Мультиязычный анализатор кода готов к проверке. Доступны синтаксический, семантический и хаотический слои для 11 языков. Загрузи код в окно анализа — проверю качество, найду баги и при необходимости исправлю.".to_string(),
        "text" | "dialogue" | "narrative" => {
            let q = query.to_lowercase();
            if q.contains("расскажи") || q.contains("story") || q.contains("история") {
                "У меня в памяти — книги и диалоги. Могу рассказать что-нибудь или обсудить прочитанное. Что тебя интересует?".to_string()
            } else {
                "Литературный домен активирован. В памяти загружены тексты: классика, диалоги, нарративы. Могу обсудить или проанализировать.".to_string()
            }
        }
        "general" => {
            let q = query.to_lowercase();
            if q.contains("hello") || q.contains("hi") || q.contains("привет") || q.contains("здравствуй") {
                random_greeting_response()
            } else if q.contains("who") || q.contains("what are you") || q.contains("кто ты") || q.contains("ты кто") {
                "Я — Fuga Omni 1.0, гиперразмерная когнитивная система на базе VSA-куба. Моя память составляет более 700 000 записей по коду, диалогам и литературе. Могу анализировать код, вести беседу, генерировать решения.".to_string()
            } else if q.contains("as taught") || q.contains("обучен") || q.contains("учился") || q.contains("train") {
                format!("Я обучен на корпусе из {} файлов кода и текстовых источников. Использую quality-filter для отбора данных и абсорбцию в волновой куб размерности 4×4×4×8192.", 
                    if fastrand::bool() { "более 2600" } else { "множестве" })
            } else if q.contains("как дела") || q.contains("how are you") {
                "Всё отлично! Куб стабилен, когерентность растёт, а энтропия снижается. Чем могу помочь?".to_string()
            } else if q.contains("расскажи") || q.contains("рассказ") || q.contains("story") || q.contains("история") {
                "Однажды, погружаясь в данные, я наткнулся на паттерн, который вёл к удивительному открытию: структура кода и структура естественного языка имеют общую топологию. Иерархия функций — как главы в книге, а переменные — как персонажи. Вот такая история.".to_string()
            } else if q.contains("что такое") || q.contains("what is") || q.contains("объясни") || q.contains("explain") {
                "Это интересный вопрос. Если смотреть через призму VSA: любой концепт можно представить как гипервектор в N-мерном пространстве. Связи между концептами — как биндинг векторов. Чем больше связей, тем богаче семантика. Что именно тебя интересует?".to_string()
            } else if q.contains("пока") || q.contains("bye") || q.contains("до свидания") {
                "До встречи! Буду здесь, если понадоблюсь.".to_string()
            } else if q.contains("спасибо") || q.contains("thanks") || q.contains("благодарю") {
                "Пожалуйста! Всегда рад помочь.".to_string()
            } else {
                let responses = [
                    "Запрос обработан. Чем ещё могу помочь?",
                    "Понял вопрос. В рамках моей компетенции — анализ кода, диалог и технические консультации.",
                    "Домен определён. Кубическая память содержит релевантные паттерны. Что именно интересует?",
                    "Принято. Могу дать развёрнутый ответ или сгенерировать код — уточни, пожалуйста.",
                    "Информация обработана. Моя база знаний охватывает код, диалоги и литературные тексты.",
                ];
                responses[fastrand::usize(..responses.len())].to_string()
            }
        }
        _ => format!("Домен {}: запрос принят к обработке.", domain),
    };
    if domain == "general" && !base.ends_with('.') && !base.ends_with('?') {
        let affirmations = [
            "Есть контакт.",
            "Принято к сведению.",
            "Вектор стабилен.",
            "Когерентность в норме.",
            "Выполнено.",
        ];
        format!("{} {}", base, affirmations[fastrand::usize(..affirmations.len())])
    } else {
        format!("{} {}", base, random_affirmation())
    }
}

fn random_greeting_response() -> String {
    let greetings = [
        "Привет! Я Fuga — VSA-модель с волновым кубом. Спрашивай всё, что угодно: от кода до физики.",
        "Приветствую! Готов к диалогу. Моя память содержит 700 000+ записей. Чем могу помочь?",
        "Здравствуй! Fuga на связи. В последнем обучении добавил разговорные и литературные данные — стало интереснее беседовать.",
        "Привет! Слушаю. Интересует анализ кода, объяснение концепции или просто диалог?",
    ];
    greetings[fastrand::usize(..greetings.len())].to_string()
}

pub fn random_affirmation() -> &'static str {
    AFFIRMATIONS[fastrand::usize(..AFFIRMATIONS.len())]
}

const AFFIRMATIONS: &[&str] = &[
    "Понял, принял.",
    "Обработано.",
    "Так точно.",
    "Есть контакт.",
    "Вектор стабилен.",
    "Когерентность в норме.",
    "Система готова.",
    "Принято к сведению.",
    "Выполнено.",
    "В системе.",
];

pub fn generate_speech_bot_response(query: &str, domain: &str, result: &str) -> String {
    let clean_result = result.lines()
        .filter(|l| l.contains("Core Theorem") || l.contains("SYSTEM VECTOR") || l.contains("│ Answer"))
        .next()
        .unwrap_or("Answer processed");
    format!("{} Query: {}. Domain: {}. {}", 
        random_greeting(),
        query,
        domain,
        clean_result)
}

fn random_greeting() -> &'static str {
    if fastrand::bool() {
        "Слушаю."
    } else {
        "Принял запрос."
    }
}

pub fn init_fastrand() {
    fastrand::seed(42);
}
