use num_complex::Complex64 as C64;
use rapidfem::config::{self, PortConfig};
use rapidfem::mesh_io::load_mesh;
use rapidfem::basis::Nedelec2Basis;
use rapidfem::waveguide::{RectWaveguide, AbsorbingBoundary, LumpedPort, detect_rect_port};
use rapidfem::port::Port;
use rapidfem::assembly::frequency_sweep;
use rapidfem::sparam::sparam_waveport;
use rapidfem::interp;
use rapidfem::constants::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rapidfem <config.toml>");
        std::process::exit(1);
    }

    let config = config::load_config(&args[1]).expect("Failed to load config");
    let mesh = load_mesh(&config.mesh.file).expect("Failed to load mesh");
    let basis = Nedelec2Basis::new(&mesh);
    let frequencies = config.frequency.frequencies();

    eprintln!("RapidFEM — {} tets, {} DOFs, {} frequencies",
        mesh.n_tets(), basis.n_field, frequencies.len());

    // Build ports from config
    let mut port_tri_indices_owned: Vec<Vec<usize>> = Vec::new();
    let mut port_refs: Vec<Box<dyn Port>> = Vec::new();

    for pc in &config.ports {
        match pc {
            PortConfig::Rectangular { tag, width, height, mode, er, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping port", tag);
                    continue;
                }
                let (cs, det_w, det_h) = detect_rect_port(&mesh, &tri_ids);
                let w = if *width > 0.0 { *width } else { det_w };
                let h = if *height > 0.0 { *height } else { det_h };
                let port_num = port_refs.len() + 1;
                let port = RectWaveguide {
                    port_number: port_num,
                    power: *power,
                    mode: (mode[0], mode[1]),
                    er: *er,
                    polarization: 1.0,
                    dims: (w, h),
                    cs,
                };
                eprintln!("  Port {}: rectangular, tag={}, TE{}{}, dims=({:.2}mm, {:.2}mm), er={:.1}",
                    port_num, tag, mode[0], mode[1], w*1e3, h*1e3, er);
                port_tri_indices_owned.push(tri_ids);
                port_refs.push(Box::new(port));
            }
            PortConfig::Lumped { tag, z0, direction, width, height, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping port", tag);
                    continue;
                }
                let port_num = port_refs.len() + 1;
                // Detect width/height from mesh if not provided
                let (_, det_w, det_h) = detect_rect_port(&mesh, &tri_ids);
                let w = if *width > 0.0 { *width } else { det_w };
                let h = if *height > 0.0 { *height } else { det_h };
                let port = LumpedPort {
                    port_number: port_num,
                    power: *power,
                    z0: *z0,
                    width: w,
                    height: h,
                    direction: *direction,
                };
                eprintln!("  Port {}: lumped, tag={}, Z0={:.0}Ω, dir=({:.1},{:.1},{:.1})",
                    port_num, tag, z0, direction[0], direction[1], direction[2]);
                port_tri_indices_owned.push(tri_ids);
                port_refs.push(Box::new(port));
            }
            PortConfig::Abc { tag, order, abctype } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping ABC", tag);
                    continue;
                }
                let abc_char = abctype.chars().next().unwrap_or('B');
                let abc = AbsorbingBoundary::new(*order, abc_char);
                eprintln!("  ABC: tag={}, order={}, type={}", tag, order, abctype);
                port_tri_indices_owned.push(tri_ids);
                port_refs.push(Box::new(abc));
            }
        }
    }

    let port_dyn_refs: Vec<&dyn Port> = port_refs.iter().map(|b| b.as_ref()).collect();
    let port_tri_refs: Vec<&[usize]> = port_tri_indices_owned.iter().map(|v| v.as_slice()).collect();

    // PEC
    let mut pec_tris: Vec<usize> = Vec::new();
    for &tag in &config.pec.tags {
        pec_tris.extend_from_slice(mesh.tris_for_tag(tag));
    }

    // Materials
    let materials: Vec<rapidfem::materials::Material> = config.materials.iter().map(|mc| {
        let tet_indices = mesh.vtag_to_tet.get(&mc.volume_tag)
            .map(|v| v.clone())
            .unwrap_or_default();
        if tet_indices.is_empty() {
            eprintln!("  WARNING: volume tag {} has no tets", mc.volume_tag);
        } else {
            eprintln!("  Material: tag={}, er={:.2}, ur={:.2}, tand={:.4}, cond={:.2e}, {} tets",
                mc.volume_tag, mc.er, mc.ur, mc.tand, mc.conductivity, tet_indices.len());
        }
        rapidfem::materials::Material {
            er: mc.er,
            ur: mc.ur,
            tand: mc.tand,
            cond: mc.conductivity,
            tet_indices,
        }
    }).collect();

    let materials_opt = if materials.is_empty() { None } else { Some(materials.as_slice()) };

    // Eigenmode analysis (if configured)
    if let Some(ref eig_config) = config.eigenmode {
        eprintln!("\n--- Eigenmode Analysis ---");
        let modes = rapidfem::eigenmode::solve_eigenmode(
            &mesh, &basis, &pec_tris, materials_opt,
            eig_config.target_frequency, eig_config.n_modes,
        );

        eprintln!("\n  {} modes found:", modes.len());
        for (i, mode) in modes.iter().enumerate() {
            let f_ghz = mode.frequency.re / 1e9;
            let q_str = if mode.q_factor.is_finite() { format!("{:.1}", mode.q_factor) } else { "∞".to_string() };
            eprintln!("    Mode {}: f = {:.6} GHz, Q = {}", i+1, f_ghz, q_str);
        }

        // Export eigenmode fields to VTK
        if let Some(ref vtk_path) = config.output.vtk {
            for (i, mode) in modes.iter().enumerate() {
                let mode_path = vtk_path.replace(".vtk", &format!("_mode{}.vtk", i+1))
                    .replace(".vtu", &format!("_mode{}.vtu", i+1));
                let label = format!("Mode {} f={:.4}GHz", i+1, mode.frequency.re/1e9);
                rapidfem::vtk_export::write_vtk(&mode_path, &mesh, &basis, &mode.field, &label)
                    .expect("Failed to write eigenmode VTK");
                eprintln!("    Wrote {}", mode_path);
            }
        }

        if config.frequency.values.is_empty() && config.frequency.range.is_empty() {
            return; // eigenmode-only run
        }
    }

    // Solve
    let t0 = std::time::Instant::now();
    let results = frequency_sweep(
        &mesh, &basis, &port_dyn_refs, &port_tri_refs, &pec_tris,
        &frequencies, materials_opt,
    );

    // Extract S-parameters
    let n_driven = port_dyn_refs.iter().filter(|p| p.is_driven()).count();
    let driven_indices: Vec<usize> = (0..port_dyn_refs.len()).filter(|&i| port_dyn_refs[i].is_driven()).collect();
    let mut all_sparams: Vec<Vec<Vec<C64>>> = Vec::new();

    for (fi, freq_result) in results.iter().enumerate() {
        let k0 = 2.0 * PI * frequencies[fi] / C0;
        let mut freq_s = vec![vec![C64::new(0.0, 0.0); n_driven]; n_driven];

        let grid = interp::TetGrid::new(&mesh);
        for (exc_idx, sol) in freq_result.solutions.iter().enumerate() {
            let fieldf = |x: f64, y: f64, z: f64| -> (C64, C64, C64) {
                match grid.find_containing_tet(&mesh, x, y, z) {
                    Some(tet) => interp::eval_field_in_tet(&mesh, &basis, sol, tet, x, y, z),
                    None => (C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)),
                }
            };
            for (obs_idx, &obs_pi) in driven_indices.iter().enumerate() {
                let obs_tris: Vec<[usize; 3]> = port_tri_refs[obs_pi].iter().map(|&ti| mesh.tris[ti]).collect();
                let active = obs_idx == exc_idx;
                let s = sparam_waveport(&mesh.nodes, &obs_tris, port_dyn_refs[obs_pi], k0, active, &fieldf, 4);
                freq_s[obs_idx][exc_idx] = s;
            }
        }
        all_sparams.push(freq_s);
    }

    let total = t0.elapsed().as_secs_f64();
    eprintln!("\nTotal: {:.3}s for {} frequency points", total, frequencies.len());

    // Print results
    for (fi, &freq) in frequencies.iter().enumerate() {
        let s = &all_sparams[fi];
        eprint!("  f={:.4e}: ", freq);
        for i in 0..n_driven {
            for j in 0..n_driven {
                eprint!("|S{}{}|={:.4} ", i+1, j+1, s[i][j].norm());
            }
        }
        eprintln!();
    }

    // VTK field export (first frequency, first port excitation)
    if let Some(ref path) = config.output.vtk {
        if let Some(first_result) = results.first() {
            if let Some(first_sol) = first_result.solutions.first() {
                let label = format!("f={:.4e}Hz", frequencies[0]);
                rapidfem::vtk_export::write_vtk(path, &mesh, &basis, first_sol, &label)
                    .expect("Failed to write VTK file");
                eprintln!("Wrote VTK: {}", path);
            }
        }
    }

    // Touchstone export
    if let Some(ref path) = config.output.touchstone {
        rapidfem::touchstone::write_touchstone(path, &frequencies, &all_sparams, n_driven, config.output.z0)
            .expect("Failed to write Touchstone file");
        eprintln!("Wrote {}", path);
    }
}
