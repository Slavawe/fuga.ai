// check_macro_norm.rs — верификация, что --macro-pc реально изменил W_macro
// (A/B 15.10: val_domains слеп к макро-каналу — меряет W_local).
// Usage: check_macro_norm <a.fuga> <b.fuga>
use fuga::ai::htm_temporal::load_unified_macro;

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() >= 3, "Usage: check_macro_norm <a.fuga> <b.fuga>");
    let a = load_unified_macro(&args[1]).expect("a: macro не читается");
    let b = load_unified_macro(&args[2]).expect("b: macro не читается");
    let mut maxdiff: f32 = 0.0;
    let mut diff_cnt = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > 1e-9 {
            diff_cnt += 1;
        }
        if d > maxdiff {
            maxdiff = d;
        }
    }
    println!("  {} : ||W_macro||={:.4}", args[1], norm(&a));
    println!("  {} : ||W_macro||={:.4}", args[2], norm(&b));
    println!("  несовпадающих элементов: {} / {} (maxdiff={:.3e})", diff_cnt, a.len(), maxdiff);
    if diff_cnt == 0 {
        println!("  ВЕРДИКТ: W_macro ИДЕНТИЧНЫ — флаг --macro-pc НЕ встал в путь обучения!");
    } else {
        println!("  ВЕРДИКТ: W_macro РАЗЛИЧАЮТСЯ — флаг активен, нужен макро-валидатор для честного A/B");
    }
}
