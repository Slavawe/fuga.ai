//! Simulation / 3D / reactor commands.
//!
//! Extracted from `src/main.rs` during monolith decomposition.

use std::process;

use crate::cli::args::{parse_dim, parse_flag_value};
use fuga::weaver::token_id;
use fuga::{FugaAI, WaveCube};

pub fn run_sim(args: &[String]) {
    use fuga::CubicController;
    use fuga::Pipe;
    use fuga::Valve;
    use fuga::sim::*;

    let stage = args.get(2).map(|s| s.as_str()).unwrap_or("1");
    let dim: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8192);

    match stage {
        "1" => {
            println!("╔══════════════════════════════════════════╗");
            println!("║  Stage 1: Valve Control Loop            ║");
            println!("║  Sensor → WaveCube → Valve              ║");
            println!("╚══════════════════════════════════════════╝");
            println!("  Dimension: {}D", dim);
            println!();

            let mut pipe = Pipe::new(1.0);
            let mut valve = Valve::new(2.0);
            let mut ctrl = CubicController::<8>::new(dim);
            ctrl.setpoint = 5.0;
            ctrl.kp = 1.5;
            ctrl.ki = 0.1;

            pipe.inflow = 1.0;

            let dt = 0.01;
            let steps = 2000;
            let mut log = Vec::with_capacity(steps);

            let base = pipe.inflow / valve.max_flow;
            for i in 0..steps {
                let signal = ctrl.update(pipe.pressure, dt);
                valve.set((base - signal).clamp(0.0, 1.0));
                pipe.step(dt, &valve);
                let stability = ctrl.phase_stability();
                log.push((i as f64 * dt, pipe.pressure, valve.position, stability));

                if i % 1000 == 0 {
                    println!(
                        "  t={:.2}s  P={:.2}Pa  valve={:.1}%  phi_stab={:.3}",
                        i as f64 * dt,
                        pipe.pressure,
                        valve.position * 100.0,
                        stability
                    );
                }
            }

            let setpoint = ctrl.setpoint;
            let final_p = pipe.pressure;
            let overshoot = ((final_p - setpoint) / setpoint * 100.0).abs();
            let settle = log
                .iter()
                .rposition(|(_, p, _, _)| (p - setpoint).abs() >= 0.1)
                .map(|i| i + 1)
                .unwrap_or(0);

            println!();
            println!("  === Results ===");
            println!("  Setpoint:      {:.2} Pa", setpoint);
            println!("  Final P:       {:.2} Pa", final_p);
            println!("  Overshoot:     {:.1}%", overshoot);
            let settle_s = settle as f64 * dt;
            println!("  Settle time:   {:.3}s", settle_s);
            println!("  Phase lock:    {:.3}", ctrl.phase_stability());

            if settle_s < 15.0 && overshoot < 15.0 {
                println!("  STAGE 1 PASS — stable in {:.3}s", settle_s);
            } else {
                println!("  STAGE 1 — oscillation detected");
            }
        }

        "2" => {
            println!("╔══════════════════════════════════════════╗");
            println!("║  Stage 2: Heater + Pump Inertia         ║");
            println!("║  Temperature rate-of-change prediction   ║");
            println!("╚══════════════════════════════════════════╝");
            println!("  Dimension: {}D", dim);
            println!();

            let mut heater = Heater::new(50.0);
            let mut ctrl = CubicController::<8>::new(dim);
            ctrl.setpoint = 60.0;
            ctrl.kp = 0.4;
            ctrl.ki = 0.05;

            let dt = 0.05;
            let steps = 1500;

            for i in 0..steps {
                let pid_signal = ctrl.update(heater.temperature, dt) + 0.5;
                heater.power = 200000.0 * pid_signal;
                heater.step(dt, 0.5, 20.0);

                if i % 100 == 0 {
                    let stability = ctrl.phase_stability();
                    println!(
                        "  t={:.1}s  T={:.1}C  power={:.0}W  phi_stab={:.3}",
                        i as f64 * dt,
                        heater.temperature,
                        heater.power,
                        stability
                    );
                }
            }

            let dtemp = heater.temperature - ctrl.setpoint;
            println!();
            println!("  === Results ===");
            println!("  Setpoint:     {:.0}C", ctrl.setpoint);
            println!("  Final T:      {:.1}C", heater.temperature);
            println!("  Steady err:   {:.2}C", dtemp);
            println!("  Phase lock:   {:.3}", ctrl.phase_stability());

            if dtemp.abs() < 5.0 {
                println!("  STAGE 2 PASS");
            } else {
                println!("  STAGE 2 — temperature not reaching setpoint");
            }
        }

        "3" => {
            println!("╔══════════════════════════════════════════╗");
            println!("║  Stage 3: Phase Transition (Water→Steam)║");
            println!("║  Boiling detection & phase lock          ║");
            println!("╚══════════════════════════════════════════╝");
            println!("  Dimension: {}D", dim);
            println!();

            let mut boiler = Boiler::new(3.0);
            let mut ctrl = CubicController::<8>::new(dim);
            ctrl.setpoint = 100.0;
            ctrl.kp = 0.8;
            ctrl.ki = 0.05;

            let dt = 0.02;
            let steps = 3000;
            let mut phase_transition_log = Vec::new();

            for i in 0..steps {
                let heat = if i < 1000 { 80.0 } else { 40.0 };
                let measurement = boiler.water_temp;
                let valve_signal = ctrl.update(measurement, dt);
                boiler.step(dt, heat, valve_signal * 0.5);

                let phase = boiler.phase();
                if let Phase::Boiling { vapor_fraction } = &phase {
                    if phase_transition_log.is_empty() {
                        println!("  Boiling onset at t={:.2}s!", i as f64 * dt);
                    }
                    phase_transition_log.push((i as f64 * dt, *vapor_fraction));
                }

                if i % 400 == 0 {
                    let stability = ctrl.phase_stability();
                    println!(
                        "  t={:.2}s  T={:.1}C  P={:.0}Pa  vapor={:.2}kg  phi_stab={:.3}",
                        i as f64 * dt,
                        boiler.water_temp,
                        boiler.pressure,
                        boiler.vapor_mass,
                        stability
                    );
                }
            }

            println!();
            println!("  === Results ===");
            println!("  Water:   {:.2}kg", boiler.water_mass);
            println!("  Vapor:   {:.2}kg", boiler.vapor_mass);
            println!("  Temp:    {:.1}C", boiler.water_temp);
            println!("  Phase:   {:?}", boiler.phase());

            if !phase_transition_log.is_empty() {
                println!("  Phase lock during boiling: {:.3}", ctrl.phase_stability());
                println!("  STAGE 3 PASS — phase transition handled");
            } else {
                println!("  STAGE 3 — no phase transition occurred");
            }
        }

        _ => {
            println!("  Usage: sim <stage> [dim]");
            println!("  Stages: 1=valve, 2=heater, 3=boiler");
            println!("  Usage: perceive [dim] [steps]");
            println!("  Embodied: Rapier3D → LiDAR raycast → WaveCube encode");
            println!("  Usage: room [dim] [steps]");
            println!("  Phase Lock: 360 LiDAR in empty room");
            println!("  Usage: room-view");
            println!("  3D viewer: closed-loop navigation with phase HUD");
        }
    }
}

pub fn run_room_phase_lock(dim: usize, steps: usize) {
    use fuga::core::hypervector::Hypervector;
    use fuga::spatial::room::Room;
    use fuga::spatial::sensor::SphericalSensor;

    let half_extent = 5.0;
    let num_rays = 128;
    let mut room = Room::new(half_extent);
    let sensor = SphericalSensor::new(num_rays, half_extent * 1.8);
    let mut cube = WaveCube::<3, 4>::new(dim);
    let mut phase_history: Vec<f64> = Vec::with_capacity(200);

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Stage 1: Room Phase Lock                       ║");
    println!("║  Empty room → 360 LiDAR → WaveCube {}D  ║", dim);
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!(
        "  Room: {}x{}x{} (half-extent)",
        half_extent, half_extent, half_extent
    );
    println!("  Sensor: {} rays (golden spiral, full sphere)", num_rays);
    println!("  Body: sphere r=0.3m at origin\n");

    for i in 0..steps {
        let pos = room.sphere_pos();
        let distances = sensor.cast_all(&pos, &room);

        let min_dist = distances.iter().copied().fold(f32::INFINITY, f32::min);
        let max_dist = distances.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let encoded = encode_distances(&distances, dim, half_extent as f64);
        let hv = Hypervector::from_i8_bits(
            dim,
            &encoded
                .iter()
                .map(|&x| if x > 0.5 { 1i8 } else { -1i8 })
                .collect::<Vec<_>>(),
        );
        let cell = (i % 4, (i / 4) % 4, (i / 16) % 4);
        cube.write_cell(cell.0, cell.1, cell.2, &hv);

        if i % 10 == 0 {
            cube.wave_flow_x(1);
            cube.wave_flow_y(1);
            cube.wave_flow_z(1);
        }

        let coherence = cube.coherence();
        phase_history.push(coherence);
        if phase_history.len() > 100 {
            phase_history.remove(0);
        }

        let stability = if phase_history.len() >= 10 {
            let recent: Vec<f64> = phase_history.iter().rev().take(10).copied().collect();
            let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
            let var: f64 =
                recent.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / recent.len() as f64;
            (1.0 - var.sqrt() * 10.0).max(0.0).min(1.0)
        } else {
            1.0
        };
        let entropy = cube.global_entropy();

        if i % 50 == 0 || i == steps - 1 {
            print!(
                "  t={:>5.1}s  min={:<5.2}m  max={:<5.2}m  phi={:<.4}  H={:<.4}  C={:<.4}",
                i as f64 * 0.05,
                min_dist,
                max_dist,
                stability,
                entropy,
                coherence
            );
            if stability > 0.9999 {
                println!("  phi=1.000");
            } else {
                println!();
            }
        }

        room.step(0.05);
    }

    println!();
    println!("  === Results ===");
    println!(
        "  Final phase lock: {:.6}",
        phase_history.last().copied().unwrap_or(0.0)
    );
    println!("  Cube coherence:   {:.6}", cube.coherence());
    println!("  Cube entropy:     {:.6}", cube.global_entropy());
    let max_stab = phase_history.iter().copied().fold(0.0_f64, f64::max);
    println!("  Peak phi_stab:     {:.6}", max_stab);

    if cube.coherence() > 0.5 {
        println!("  ROOM PHASE LOCK — spatial anchor acquired");
    } else {
        println!("  Room phase unstable — geometry not resolved");
    }
}

pub fn encode_distances(distances: &[f32], dim: usize, room_size: f64) -> Vec<f64> {
    let mut vec = vec![0.0; dim];
    let rays = distances.len();
    for (i, &d) in distances.iter().enumerate() {
        let idx = (i as f64 / rays as f64 * dim as f64) as usize % dim;
        vec[idx] = (d as f64 / room_size).min(1.0);
    }
    vec
}

pub fn run_perceive(dim: usize, steps: usize) {
    use fuga::core::hypervector::Hypervector;
    use fuga::spatial::SpatialPerception;

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Embodied Perception Pipeline                   ║");
    println!("║  Rapier3D → Raycast LiDAR → WaveCube {}D  ║", dim);
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let mut perception = SpatialPerception::new(dim);
    let mut cube = WaveCube::<3, 8>::new(dim);
    let dt = 1.0 / 60.0;

    println!(" World: ground + 1 ball + 1 cube + 1 wall");
    println!(" Sensor: 64 rays, max 20m, FOV 135x90");
    println!(" Agent: ball at z=5m, gravity pulls down\n");

    let _gravity = 9.81;
    let mut prev_entropy = 0.5;

    for i in 0..steps {
        let obs = perception.step(dt);
        let encoded = perception.encode_obs(&obs);

        let hv = Hypervector::from_i8_bits(
            dim,
            &encoded
                .iter()
                .map(|&x| if x > 0.5 { 1i8 } else { -1i8 })
                .collect::<Vec<_>>(),
        );
        let side = 8;
        let x = ((obs.agent_pos[1].abs() * 2.0) as usize).min(side - 1);
        let y = ((obs.agent_vel[1].abs() * 3.0) as usize).min(side - 1);
        let z = i % side;
        cube.write_cell(x, y, z, &hv);

        if i % 10 == 0 {
            cube.wave_flow_x(1);
            cube.wave_flow_y(1);
            cube.wave_flow_z(1);
        }

        let entropy = cube.global_entropy();
        let phase_drift = (entropy - prev_entropy).abs();
        let stability = (1.0 - phase_drift * 20.0).max(0.0);
        prev_entropy = entropy;

        if i % 60 == 0 {
            let hits: Vec<f64> = obs
                .depth_map
                .iter()
                .filter(|&&d| d < 19.9)
                .copied()
                .collect();
            let depth_avg = if hits.is_empty() {
                20.0
            } else {
                hits.iter().sum::<f64>() / hits.len() as f64
            };
            let coherence = cube.coherence();

            println!(
                "  t={:>6.1}s  z={:<6.2}m  vz={:<6.2}m/s  depth={:<5.2}m  phi={:<.3}  coh={:<.3}",
                i as f64 * dt,
                obs.agent_pos[1],
                obs.agent_vel[1],
                depth_avg,
                stability,
                coherence
            );
        }

        if obs.agent_pos[1] < 0.6 {
            let impact_v = obs.agent_vel[1];
            println!();
            println!(
                "  Ball hit ground at t={:.2}s — phase disrupted",
                i as f64 * dt
            );
            println!(
                "  Impact vz: {:.2} m/s, momentum: {:.2} kg.m/s",
                impact_v,
                1.0 * impact_v.abs()
            );
            println!("  Cube entropy: {:.4}", cube.global_entropy());
            println!("  Cube coherence: {:.4}", cube.coherence());
            println!("  phi_stab at impact: {:.3}", stability);
            break;
        }
    }
}

pub fn run_view_3d() {
    use fuga::physics::fluid::{FluidField, color_from_density};
    use fuga::physics::neutron::{NeutronDiffusion, color_from_flux};
    use fuga::render::Render3D;
    use fuga::spatial::SpatialPerception;

    let dim = 8192;
    let mut perception = SpatialPerception::new(dim);
    let mut render = Render3D::new("Fuga 3D — Rapier3D + Neutron Diffusion + CFD");
    let mut neutron = NeutronDiffusion::new(16, 16, 16);
    let mut fluid = FluidField::new(20, 20);
    let dt = 1.0 / 60.0;
    let mut _tick = 0u64;

    while render.is_open() {
        render.clear(0x1a1a2e);

        render.draw_ground_grid(0x3a3a5e);

        let obs = perception.step(dt);
        let pos = obs.agent_pos;
        let origin = [pos[0] as f32, pos[1] as f32 + 0.6, pos[2] as f32];

        if _tick % 3 == 0 {
            let s = &perception.sensors[0];
            render.draw_ray(&origin, &s.direction, 3.0, 0x00FF0055);
        }

        _tick += 1;

        render.draw_sphere_wire(pos[0] as f32, pos[1] as f32, pos[2] as f32, 0.5, 0x4FC3F7);
        render.draw_cube_wire(3.0, 0.5, 0.0, 1.0, 0x8BC34A);
        render.draw_cube_wire(-3.0, 1.0, 0.0, 2.0, 0xFF7043);

        neutron.step(0.01);
        for i in -1..=1 {
            for k in -1..=1 {
                let fx = i as f32 * 1.5;
                let fz = k as f32 * 1.5;
                let flux = neutron.flux_at(fx, 0.5, fz);
                if flux > 0.01 {
                    let c = color_from_flux(flux);
                    render.draw_cube_wire(fx, 0.05, fz, 0.3, c);
                }
            }
        }

        fluid.add_source(0.5, 0.0, 0.02);
        fluid.step(0.02);
        for i in 0..5 {
            for k in 0..5 {
                let fx = (i as f32 / 5.0) * 6.0 - 3.0;
                let fz = (k as f32 / 5.0) * 6.0 - 3.0;
                let d = fluid.density_at(fx, fz);
                if d > 0.01 {
                    let c = color_from_density(d);
                    render.draw_cube_wire(fx, 0.01, fz, 0.2, c);
                }
            }
        }

        render.update();
        std::thread::sleep(std::time::Duration::from_secs_f64(dt));
    }
}

pub fn run_room_view_3d() {
    use fuga::render::Render3D;
    use fuga::spatial::controller::RoomController;
    use fuga::spatial::room::Room;
    use fuga::spatial::sensor::SphericalSensor;

    let half_extent = 5.0;
    let num_rays = 128;
    let dim = 8192;
    let mut room = Room::new(half_extent);
    let sensor = SphericalSensor::new(num_rays, half_extent * 1.8);
    let mut ctrl = RoomController::new(dim, half_extent as f64);
    let dt = 1.0 / 60.0;

    let mut render = Render3D::new("Fuga Room — Phase Lock Navigation");
    let mut data_log: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut trail: Vec<[f32; 3]> = Vec::new();

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Stage 2: Phase-Locked Trajectory              ║");
    println!("║  Lissajous path + wall repulsion + WaveCube    ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("  Room: {}x{}x{}", half_extent, half_extent, half_extent);
    println!("  Path: Lissajous (1.8.sin(omega t), 1.8.cos(2 omega t))");
    println!("  Target phi: >= 0.95    Target H: <= 0.15\n");

    let mut render_timer = 0u64;

    while render.is_open() {
        render.clear(0x0d0d1a);

        let pos = room.sphere_pos();
        let vel = room.sphere_vel();
        let distances = sensor.cast_all(&pos, &room);

        let t = render_timer as f64 * dt;
        let omega = 0.3;
        let tx = 1.8 * (omega * t).sin();
        let tz = 1.8 * (2.0 * omega * t).cos();
        ctrl.set_target(tx, tz);

        let force = ctrl.compute(pos, vel, &distances, render_timer as usize);
        room.apply_force(&force);
        room.step(dt);

        let nearest = distances.iter().copied().fold(f32::INFINITY, f32::min) as f64;

        trail.push([pos[0] as f32, pos[1] as f32, pos[2] as f32]);
        if trail.len() > 100 {
            trail.remove(0);
        }

        render.draw_ground_grid(0x1a1a3a);

        let hs = half_extent;
        let wall_color = 0x3a3a6a;
        render.draw_line(&[-hs, -hs, -hs], &[hs, -hs, -hs], wall_color);
        render.draw_line(&[hs, -hs, -hs], &[hs, -hs, hs], wall_color);
        render.draw_line(&[hs, -hs, hs], &[-hs, -hs, hs], wall_color);
        render.draw_line(&[-hs, -hs, hs], &[-hs, -hs, -hs], wall_color);
        render.draw_line(&[-hs, hs, -hs], &[hs, hs, -hs], wall_color);
        render.draw_line(&[hs, hs, -hs], &[hs, hs, hs], wall_color);
        render.draw_line(&[hs, hs, hs], &[-hs, hs, hs], wall_color);
        render.draw_line(&[-hs, hs, hs], &[-hs, hs, -hs], wall_color);
        render.draw_line(&[-hs, -hs, -hs], &[-hs, hs, -hs], wall_color);
        render.draw_line(&[hs, -hs, -hs], &[hs, hs, -hs], wall_color);
        render.draw_line(&[hs, -hs, hs], &[hs, hs, hs], wall_color);
        render.draw_line(&[-hs, -hs, hs], &[-hs, hs, hs], wall_color);

        let origin = [pos[0] as f32, pos[1] as f32 + 0.35, pos[2] as f32];
        let max_dist = half_extent * 1.8f32.sqrt();
        for (j, dir) in sensor.directions.iter().enumerate() {
            let d = distances[j];
            let frac = (d / max_dist).min(1.0);
            let r = (255.0 * (1.0 - frac)) as u32;
            let g = (255.0 * frac) as u32;
            let ray_color = (r.min(255) << 16) | (g.min(255) << 8);
            render.draw_ray(&origin, dir, d, ray_color);
            let hit = [
                origin[0] + dir[0] * d,
                origin[1] + dir[1] * d,
                origin[2] + dir[2] * d,
            ];
            render.draw_dot(&hit, 1.5, ray_color);
        }

        for (i, tp) in trail.iter().enumerate() {
            let alpha = (i as f32 / trail.len() as f32 * 0.6) as u32;
            render.draw_dot(tp, 1.2, (alpha << 24) | 0x4FC3F7);
        }

        render.draw_sphere_wire(pos[0] as f32, pos[1] as f32, pos[2] as f32, 0.3, 0x4FC3F7);
        render.draw_dot(
            &[pos[0] as f32, pos[1] as f32, pos[2] as f32],
            3.0,
            0x4FC3F7,
        );

        let fscale = 0.5;
        let fend = [
            pos[0] as f32 + force[0] as f32 * fscale,
            pos[1] as f32 + force[1] as f32 * fscale,
            pos[2] as f32 + force[2] as f32 * fscale,
        ];
        render.draw_arrow(
            &[pos[0] as f32, pos[1] as f32, pos[2] as f32],
            &fend,
            0xFFD700,
        );

        if render_timer as usize % 300 < 15 {
            render.draw_sphere_wire(tx as f32, 0.0, tz as f32, 0.25, 0xFFD700);
            render.draw_dot(&[tx as f32, 0.0, tz as f32], 4.0, 0xFFD700);
        }

        let stab = ctrl.phase_stability();
        let ent = ctrl.entropy();
        let coh = ctrl.coherence();

        if render_timer % 30 == 0 {
            let dist_to_target = ((tx - pos[0]).powi(2) + (tz - pos[2]).powi(2)).sqrt() as f64;
            data_log.push((stab, ent, coh, nearest));
            if data_log.len() % 2 == 0 || stab > 0.99 || nearest < 1.0 {
                print!(
                    "\r  t={:>5.1}s  pos=({:>5.2},{:>5.2})  nearest={:<5.2}m  dist_t={:<5.2}  phi={:<.4}  H={:<.4}  C={:<.4}  force=({:>+5.2},{:>+5.2})",
                    render_timer as f64 * dt,
                    pos[0],
                    pos[2],
                    nearest,
                    dist_to_target,
                    stab,
                    ent,
                    coh,
                    force[0],
                    force[2]
                );
                if nearest < 0.6 {
                    print!(" WALL");
                }
                if dist_to_target < 0.5 {
                    print!(" ON PATH");
                }
                println!();
            }
        }

        render.update();
        render_timer += 1;
        std::thread::sleep(std::time::Duration::from_secs_f64(dt * 0.5));
    }

    println!("\n\n  === Results ===");
    let avg_stab: f64 =
        data_log.iter().map(|(s, _, _, _)| s).sum::<f64>() / data_log.len().max(1) as f64;
    let avg_ent: f64 =
        data_log.iter().map(|(_, e, _, _)| e).sum::<f64>() / data_log.len().max(1) as f64;
    let min_nearest: f64 = data_log
        .iter()
        .map(|(_, _, _, n)| n)
        .copied()
        .fold(f64::INFINITY, f64::min);
    println!("  Avg phi_stab:     {:.4}", avg_stab);
    println!("  Avg entropy:    {:.4}", avg_ent);
    println!("  Min wall dist:  {:.4}m", min_nearest);

    if avg_stab >= 0.95 && avg_ent <= 0.15 && min_nearest > 0.05 {
        println!("  STAGE 2 PASS — closed-loop navigation stable");
    } else {
        println!("  STAGE 2 FAIL");
        if avg_stab < 0.95 {
            println!("      reason: phi_stab {:.4} < 0.95", avg_stab);
        }
        if avg_ent > 0.15 {
            println!("      reason: entropy {:.4} > 0.15", avg_ent);
        }
        if min_nearest <= 0.05 {
            println!("      reason: wall collision (min {:.4}m)", min_nearest);
        }
    }
}

pub fn run_reactor(steps: usize) {
    use fuga::physics::reactor::ReactorCore;
    let mut core = ReactorCore::default();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Stage 3: Reactor Core — Point Kinetics         ║");
    println!("║  ln 235U thermal · 2 group rods · Doppler       ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("  Parameters:");
    println!("    beta  = {:.4}  (delayed neutron fraction)", core.beta);
    println!("    Lambda = {:.2e} s  (neutron lifetime)", core.lambda);
    println!("    lambda = {:.4} s-1  (precursor decay)", core.decay);
    println!("    alpha_T = {:.1} pcm/K  (Doppler coeff)", core.alpha_t);
    println!();

    for step in 0..steps {
        let withdrawal = if step < 50 {
            0.3
        } else if step < 100 {
            0.5
        } else if step < 200 {
            0.7
        } else {
            0.85
        };
        core.rod_set(&[withdrawal, withdrawal]);
        core.step(0.001);
        if step % 50 == 0 {
            let power_mw = core.n * 3000.0;
            print!(
                "\r  t={:>6.3}s  rho={:>+7.1}pcm  n={:>10.6}  P={:>9.3}MW  T={:>7.2}K  rods={:.2}/{:.2}",
                core.time,
                core.rho,
                core.n,
                power_mw,
                core.t,
                core.rods[0].position,
                core.rods[1].position
            );
            if core.n > 0.9 {
                print!(" CRITICAL");
            }
            if core.n > 1.5 {
                print!(" EXCURSION");
            }
            println!();
        }
        if core.n > 10.0 {
            println!("\n  SCRAM triggered!");
            core.scram();
            break;
        }
    }
    println!("\n\n  === Final State ===");
    println!(
        "  t = {:.3}s   n = {:.6}   T = {:.2}K   rho = {:.1} pcm",
        core.time, core.n, core.t, core.rho
    );

    println!("  Power: {:.2} MW", core.n * 3000.0);
}

pub fn run_reactor_view_3d() {
    use fuga::physics::reactor::ReactorCore;
    use fuga::render::Render3D;
    let mut core = ReactorCore::default();
    let mut render = Render3D::new("Fuga Reactor — Core View");
    let mut step: usize = 0;
    let num_fuel = 5;
    let pitch = 1.2;
    let off = (num_fuel as f32 - 1.0) * pitch / 2.0;

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Stage 3: Reactor Core 3D Viewer               ║");
    println!("║  Fuel rods · control rods · neutron flux        ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    while render.is_open() {
        render.clear(0x0a0a1a);
        let w = (0.3 + (step as f64 * 0.0005).sin() * 0.2).clamp(0.1, 0.9);
        core.rod_set(&[w, w]);
        core.step(0.001);
        let pf = (core.n * 2.0).min(1.0);

        render.draw_ground_grid(0x151530);

        for iz in 0..num_fuel {
            for ix in 0..num_fuel {
                let fx = ix as f32 * pitch - off;
                let fz = iz as f32 * pitch - off;
                let dist = ((ix as f64 - (num_fuel as f64 - 1.0) / 2.0).powi(2)
                    + (iz as f64 - (num_fuel as f64 - 1.0) / 2.0).powi(2))
                .sqrt();
                let flux = (dist / (num_fuel as f64 * 0.6)).min(1.0);
                let bright =
                    (120.0 + (flux * std::f64::consts::PI).cos().max(0.0) * 80.0 * pf) as u32;
                let c = (bright.min(255) << 16) | ((bright.min(255) * 2 / 3) << 8);
                render.draw_line(&[fx, -1.5, fz], &[fx, 1.5, fz], c);
            }
        }

        let rp = core.rods[0].position as f32;
        for iz in [0, num_fuel - 1] {
            for ix in [0, num_fuel - 1] {
                let rx = ix as f32 * pitch - off;
                let rz = iz as f32 * pitch - off;
                let ch = 2.5 * (1.0 - rp);
                render.draw_line(&[rx, -1.5, rz], &[rx, -1.5 + ch, rz], 0xFF4444);
                if ch > 0.1 {
                    render.draw_dot(&[rx, -1.5 + ch, rz], 2.0, 0xFF6666);
                }
            }
        }

        for _ in 0..(pf * 200.0) as usize {
            let (fx, fy, fz): (f32, f32, f32) = (
                rand::random::<f32>() - 0.5,
                rand::random::<f32>() - 0.5,
                rand::random::<f32>() - 0.5,
            );
            let b = 50 + (rand::random::<f32>() * 150.0) as u32;
            render.draw_dot(
                &[fx * 6.0, fy * 4.0, fz * 6.0],
                (pf as f32) * 0.6 + 0.5,
                (b << 8) | (b >> 1),
            );
        }

        render.draw_cube_wire(0.0, 0.0, 0.0, 6.0, 0x2a2a5a);

        if step % 60 == 0 {
            print!(
                "\r  t={:>6.3}s  n={:>10.6}  P={:>9.3}MW  T={:>7.2}K  rods={:.3}/{:.3}",
                core.time,
                core.n,
                core.n * 3000.0,
                core.t,
                core.rods[0].position,
                core.rods[1].position
            );
            if core.n > 0.9 {
                print!(" CRITICAL");
            }
            if core.n > 1.5 {
                print!(" EXCURSION");
            }
            if core.n > 5.0 {
                print!(" SCRAM");
            }
            println!();
        }

        if render.window.is_key_down(minifb::Key::R) {
            core.rod_set(&[(core.rods[0].position + 0.01).min(1.0); 2]);
        }
        if render.window.is_key_down(minifb::Key::F) {
            core.rod_set(&[(core.rods[0].position - 0.01).max(0.0); 2]);
        }
        if render.window.is_key_down(minifb::Key::Space) {
            core.scram();
        }
        if core.n > 10.0 {
            core.scram();
        }

        render.update();
        step += 1;
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}

