# C++ ядро fuga (бета-тест)

Байтовый стек обучения и декодеров, портированный из Rust (`src/ai/*`).

## Статус: БЕТА-ТЕСТ C++

Ядро обучения и декодеров переведено на C++ без внешних зависимостей
(std:: только; g++ ≥ 12, C++20). Формат чекпоинтов FBW1 **bin-совместим**
с Rust (`save_byte_w`/`load_byte_w`): C++ читает Rust-обученные sidecar-файлы.

## Состав

| Файл | Что делает | Rust-референс |
|---|---|---|
| `fuga_core.h` | byte_basis, encode_bytes_sdr, structure_sdr_from_sdrs, SdrEncoder (8192→512), LatentPredictor (Widrow-Hoff + OWM-P), FBW1 IO | sdr.rs, latent_jepa.rs, htm_temporal.rs |
| `train.cpp` | обучение из JSONL: локальный W + патчевый W_patch, sidecar-сейв | full_byte_train.rs |
| `decode.cpp` | декодеры: naive, recurrent (SSM-lite h(t), mix, gap-φ), entropy (BLT two-speed) | tm_generate.rs |

## Сборка

```bash
cd cpp && make            # или: g++ -O3 -std=c++20 train.cpp -o fuga_train
```

## Использование

```bash
# обучение (локальный W + W_patch), 300K байт-шагов
./fuga_train --jsonl ../corpus_doc_code_pairs.jsonl --max-bytes 300000 \
             --out /tmp/fuga_cpp_w300k.bin

# декод Rust-обученным W (формат FBW1 совместим)
./fuga_decode --w ../fuga_byte_w_800.bin --decoder recurrent \
              --mix 0.4 --phi 0.9 --ctx 4 --seed "fn main() {" --max 200

# декод C++-обученным W
./fuga_decode --w /tmp/fuga_cpp_w300k.bin --decoder recurrent --seed "fn main() {"
```

## Честные ограничения (текущая бета)

- RNG: C++ использует **splitmix64** (детерминированный); Rust — StdRng (ChaCha12).
  Форматы совместимы, битовая идентичность SDR/латентов с Rust НЕ гарантируется —
  W обученный C++ ядром корректно декодирует C++ декодером, и наоборот.
- Скорость: **~170 byte-steps/s** на ноуте (20K шагов ≈ 2 мин) — bottleneck
  `SdrEncoder.encode` (164 set-бита × 512 латентов + хэш на пару, 2× на слой,
  2 уровня). Rust ~850 B/s (SIMD/cache-locality + единый кэш). Кэши включены:
  byte_basis (256 SDR) и encode_bytes_sdr для 2-байтовых патчей (65536 SDR).
  Оптимизация (кэш x-encode окна, SIMD) — следующая итерация после бета.
- `decode.cpp` entropy требует `--vocab` (список патчей из корпуса) — генератор
  живёт в Python-оркестраторе (`py/train_cpp.py`).

## Что дальше (не в этой бете)

- Python-оркестратор: гейты (compile+relevance), агентский контур, запуск обучения
- GPU: Vulkan/CUDA compute для Widrow-Hoff дельты (Rust-путь уже есть: `src/ai/gpu_ops.rs`)
- Кэш структура-свёртки окна; ~2-3× к скорости Rust
