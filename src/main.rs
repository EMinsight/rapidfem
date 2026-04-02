use num_complex::Complex64 as C64;
use rapidfem::config::{self, PortConfig};
use rapidfem::mesh_io::load_mesh;
use rapidfem::basis::Nedelec2Basis;
use rapidfem::waveguide::{RectWaveguide, AbsorbingBoundary, detect_rect_port};
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
    let mut rect_ports: Vec<RectWaveguide> = Vec::new();
    let mut abc_ports: Vec<AbsorbingBoundary> = Vec::new();
    let mut port_tri_indices_owned: Vec<Vec<usize>> = Vec::new();
    let mut port_refs: Vec<Box<dyn Port>> = Vec::new();

    for pc in &config.ports {
        match pc {
            PortConfig::Rectangular { tag, width, height } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                let (cs, det_w, det_h) = detect_rect_port(&mesh, &tri_ids);
                let w = if *width > 0.0 { *width } else { det_w };
                let h = if *height > 0.0 { *height } else { det_h };
                let port = RectWaveguide {
                    port_number: rect_ports.len() + 1,
                    power: 1.0, mode: (1, 0), er: 1.0,
                    polarization: 1.0, dims: (w, h), cs,
                };
                eprintln!("  Port {}: rectangular, tag={}, dims=({:.2}mm, {:.2}mm)",
                    port.port_number, tag, w*1e3, h*1e3);
                port_tri_indices_owned.push(tri_ids);
                rect_ports.push(port);
            }
            PortConfig::Abc { tag, order } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                let abc = AbsorbingBoundary::new(*order, 'B');
                eprintln!("  ABC: tag={}, order={}", tag, order);
                port_tri_indices_owned.push(tri_ids);
                abc_ports.push(abc);
            }
        }
    }

    // Build trait object references
    for p in &rect_ports { port_refs.push(Box::new(RectWaveguide {
        port_number: p.port_number, power: p.power, mode: p.mode, er: p.er,
        polarization: p.polarization, dims: p.dims,
        cs: rapidfem::waveguide::CoordinateSystem::new(p.cs.origin, p.cs.xax, p.cs.yax, p.cs.zax),
    })); }
    for a in &abc_ports { port_refs.push(Box::new(AbsorbingBoundary::new(a.order, a.abctype))); }

    let port_dyn_refs: Vec<&dyn Port> = port_refs.iter().map(|b| b.as_ref()).collect();
    let port_tri_refs: Vec<&[usize]> = port_tri_indices_owned.iter().map(|v| v.as_slice()).collect();

    // PEC
    let mut pec_tris: Vec<usize> = Vec::new();
    for &tag in &config.pec.tags {
        pec_tris.extend_from_slice(mesh.tris_for_tag(tag));
    }

    // Solve
    let t0 = std::time::Instant::now();
    let results = frequency_sweep(
        &mesh, &basis, &port_dyn_refs, &port_tri_refs, &pec_tris,
        &frequencies, None,
    );

    // Extract S-parameters
    let n_driven = port_dyn_refs.iter().filter(|p| p.is_driven()).count();
    let driven_indices: Vec<usize> = (0..port_dyn_refs.len()).filter(|&i| port_dyn_refs[i].is_driven()).collect();
    let mut all_sparams: Vec<Vec<Vec<C64>>> = Vec::new();

    for (fi, freq_result) in results.iter().enumerate() {
        let k0 = 2.0 * PI * frequencies[fi] / C0;
        let mut freq_s = vec![vec![C64::new(0.0, 0.0); n_driven]; n_driven];

        for (exc_idx, sol) in freq_result.solutions.iter().enumerate() {
            let fieldf = |x: f64, y: f64, z: f64| -> (C64, C64, C64) {
                match interp::find_containing_tet(&mesh, x, y, z) {
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

    // Touchstone export
    if let Some(ref path) = config.output.touchstone {
        rapidfem::touchstone::write_touchstone(path, &frequencies, &all_sparams, n_driven, 50.0)
            .expect("Failed to write Touchstone file");
        eprintln!("Wrote {}", path);
    }
}
