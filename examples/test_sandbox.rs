fn main() {
    let data = std::fs::read("/tmp/opencode/v4_full.state").unwrap();
    let mut pos = 8usize;
    let rd_u32 = |p: &mut usize| { let v = u32::from_le_bytes(data[*p..*p+4].try_into().unwrap()); *p += 4; v };
    let rd_u64 = |p: &mut usize| { let v = u64::from_le_bytes(data[*p..*p+8].try_into().unwrap()); *p += 8; v };
    let rd_str = |p: &mut usize| { let n = rd_u32(p) as usize; let s = String::from_utf8_lossy(&data[*p..*p+n]).to_string(); *p += n; s };
    let dim = rd_u32(&mut pos) as usize;
    let route_cap = rd_u32(&mut pos) as usize;
    let processed = rd_u32(&mut pos) as usize;
    let bytes_in = rd_u64(&mut pos);
    println!("dim={} route_cap={} processed={} bytes_in={:.2}GB", dim, route_cap, processed, bytes_in as f64/1e9);
    pos += 3 * dim * 8;
    let nskip = rd_u32(&mut pos) as usize;
    for _ in 0..nskip { rd_str(&mut pos); }
    println!("skipped={}", nskip);
    let nent = rd_u32(&mut pos) as usize;
    let wc = (dim + 63) / 64;
    pos += nent * (wc*8 + 8 + 4 + 4 + 2 + 1 + 4);
    // need to skip text lengths too — let's just read entries carefully
    println!("nent={} (approx; text lens skipped)", nent);
    let ndone = rd_u32(&mut pos) as usize;
    println!("done shards={}", ndone);
    for _ in 0..ndone.min(60) { println!("  {}", rd_str(&mut pos)); }
}
