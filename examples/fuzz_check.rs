use fuga::PhaseCrystal;
use fuga::core::tokenizer_bridge::encode_bytes_nopos;
use fuga::core::hypervector::Hypervector;
fn soft_overlap(a: &Hypervector, b: &Hypervector) -> f64 {
    let mut o=0u32;
    for w in 0..a.words.len().min(b.words.len()){o+=(a.words[w]&b.words[w]).count_ones();}
    let an=a.words.iter().map(|w|w.count_ones()).sum::<u32>();
    let bn=b.words.iter().map(|w|w.count_ones()).sum::<u32>();
    if an.max(bn)<=0{0.0}else{o as f64/an.max(bn) as f64}
}
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let c = PhaseCrystal::load(&path).unwrap();
    let chunks: Vec<(usize,String)> = c.entries.iter().enumerate()
        .filter(|(_,e)|e.key_text.contains('#')).take(200)
        .map(|(i,e)|(i,e.text.clone())).collect();
    if chunks.is_empty(){println!("NO chunks");return;}
    let mut s_sum=0.0; let mut s_n=0;
    for (k,(i,txt)) in chunks.iter().enumerate() {
        let q = encode_bytes_nopos(txt.as_bytes(), c.dim);
        s_sum += soft_overlap(&q, &c.entries[*i].hv); s_n += 1;
    }
    let mut rk=[0usize;5]; let mut t_sum=0.0; let mut t_n=0; let mut t_ok=0;
    for (k,(_,txt)) in chunks.iter().enumerate() {
        let w:Vec<&str> = txt.split_whitespace().collect();
        if w.len()<12{continue;}
        let snip = w.iter().skip(w.len()/2-5).take(10).cloned().collect::<Vec<_>>().join(" ");
        let q = encode_bytes_nopos(snip.as_bytes(), c.dim);
        let mut scored: Vec<(f64,usize)> = c.entries.iter().enumerate()
            .map(|(i,e)|(soft_overlap(&q,&e.hv),i)).collect();
        scored.sort_by(|a,b|b.0.partial_cmp(&a.0).unwrap());
        for (r,(_,idx)) in scored.iter().take(5).enumerate(){ if *idx==chunks[k].0 {rk[r]+=1;break;} }
        t_sum += scored[0].0; t_n += 1;
        if scored[0].1==chunks[k].0 { t_ok += 1; }
    }
    let noise = "zzzqxwv asdfgh jklpoiu mnbvcx rtyuiop 1234567890 qwertyuiopasdfghjkl";
    let qn = encode_bytes_nopos(noise.as_bytes(), c.dim);
    let nb = c.entries.iter().map(|e|soft_overlap(&qn,&e.hv)).fold(0.0f64,f64::max);
    let nres = c.query(noise);
    println!("SELF avg={:.3}", s_sum/s_n as f64);
    println!("SNIP10mid top1..top5={:?} top1={}/{} avg={:.3}", rk, t_ok, t_n, t_sum/t_n as f64);
    println!("NOISE best={:.3} CLI_silence={}", nb, nres.is_none());
}
