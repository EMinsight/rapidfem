use num_complex::Complex64 as C64;
use rapidfem::config::{self, PortConfig};
use rapidfem::mesh_io::load_mesh;
use rapidfem::basis::Nedelec2Basis;
use rapidfem::waveguide::{RectWaveguide, AbsorbingBoundary, LumpedPort, SurfaceImpedance, LumpedElement, CoaxPort, UserDefinedPort, FloquetPort, detect_rect_port, lumped_port_dims, cs_from_origin_zaxis};
use rapidfem::port::Port;
use rapidfem::assembly::frequency_sweep_with_pml;
use rapidfem::sparam::{sparam_waveport, sparam_voltage_line};
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
            PortConfig::Floquet { tag, scan_theta_deg, scan_phi_deg, mode_nr, er, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping FloquetPort", tag);
                    continue;
                }
                // Compute port-face area and CS from the triangulation
                let (cs_detected, det_w, det_h) = detect_rect_port(&mesh, &tri_ids);
                let area = det_w * det_h;
                let port_num = port_refs.len() + 1;
                let port = FloquetPort {
                    port_number: port_num,
                    power: *power, er: *er, area,
                    scan_theta: scan_theta_deg.to_radians(),
                    scan_phi: scan_phi_deg.to_radians(),
                    mode_nr: *mode_nr,
                    cs: cs_detected,
                };
                eprintln!("  Port {}: floquet, tag={}, mode={} ({}), θ={:.1}°, φ={:.1}°, A={:.2}mm²",
                    port_num, tag, mode_nr,
                    if *mode_nr == 1 { "TE/S" } else { "TM/P" },
                    scan_theta_deg, scan_phi_deg, area*1e6);
                if *scan_theta_deg > 1e-6 {
                    eprintln!("    WARNING: oblique incidence drops transverse phase factor in mode field — approximate");
                }
                port_tri_indices_owned.push(tri_ids);
                port_refs.push(Box::new(port));
            }
            PortConfig::UserDefined { tag, e_field, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping UserDefined", tag);
                    continue;
                }
                let port_num = port_refs.len() + 1;
                let port = UserDefinedPort::from_constant(port_num, *power, *e_field);
                eprintln!("  Port {}: user_defined, tag={}, E=({:.3},{:.3},{:.3}), P={:.2}W",
                    port_num, tag, e_field[0], e_field[1], e_field[2], power);
                port_tri_indices_owned.push(tri_ids);
                port_refs.push(Box::new(port));
            }
            PortConfig::Coax { tag, ri, ro, origin, z_axis, er, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping CoaxPort", tag);
                    continue;
                }
                // Auto-detect origin (face centroid) and z_axis (face normal) if not given
                let (cs_detected, _, _) = detect_rect_port(&mesh, &tri_ids);
                let org = origin.unwrap_or(cs_detected.origin);
                let zax = z_axis.unwrap_or(cs_detected.zax);
                let cs = cs_from_origin_zaxis(org, zax);
                let port_num = port_refs.len() + 1;
                let port = CoaxPort {
                    port_number: port_num,
                    power: *power, er: *er, ri: *ri, ro: *ro, cs,
                };
                eprintln!("  Port {}: coax, tag={}, Ri={:.3}mm, Ro={:.3}mm, εr={:.2}, Z₀={:.2}Ω",
                    port_num, tag, ri*1e3, ro*1e3, er, port.port_z());
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
                // EMerge convention: height = extent along `direction`, width = extent orthogonal
                // (NOT the geometric broad/narrow ordering). See microwave_bc.py:1314-1317.
                let (det_w, det_h) = lumped_port_dims(&mesh, &tri_ids, direction);
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
            PortConfig::Pmc { tag } => {
                let tri_ids = mesh.tris_for_tag(*tag);
                eprintln!("  PMC: tag={}, {} triangles (natural BC)", tag, tri_ids.len());
            }
            PortConfig::LumpedElement { tag, r, l, c, width, height, direction } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping LumpedElement", tag);
                    continue;
                }
                let (det_w, det_h) = lumped_port_dims(&mesh, &tri_ids, direction);
                let w = if *width > 0.0 { *width } else { det_w };
                let h = if *height > 0.0 { *height } else { det_h };
                let bc = LumpedElement { r: *r, l: *l, c: *c, width: w, height: h };
                eprintln!("  LumpedElement: tag={}, R={:.2}Ω, L={:.2e}H, C={:?}F, w={:.2}mm, h={:.2}mm",
                    tag, r, l, c, w*1e3, h*1e3);
                port_tri_indices_owned.push(tri_ids);
                port_refs.push(Box::new(bc));
            }
            PortConfig::SurfaceImpedance { tag, conductivity, mur, er, thickness, zs } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping SurfaceImpedance", tag);
                    continue;
                }
                let bc = if let Some(zs_arr) = zs {
                    let mut s = SurfaceImpedance::from_zs(C64::new(zs_arr[0], zs_arr[1]));
                    s.mur = *mur; s.er = *er; s.thickness = *thickness;
                    s
                } else {
                    let mut s = SurfaceImpedance::from_conductivity(*conductivity);
                    s.mur = *mur; s.er = *er; s.thickness = *thickness;
                    s
                };
                eprintln!("  SurfaceImpedance: tag={}, σ={:.2e}S/m, μr={:.2}, εr={:.2}, t={:?}",
                    tag, conductivity, mur, er, thickness);
                port_tri_indices_owned.push(tri_ids);
                port_refs.push(Box::new(bc));
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
        let dispersion = if let Some(d) = &mc.debye {
            rapidfem::materials::Dispersion::Debye {
                er_inf: d.er_inf, er_static: d.er_static, tau_s: d.tau_s,
            }
        } else if let Some(d) = &mc.drude {
            rapidfem::materials::Dispersion::Drude {
                er_inf: d.er_inf, plasma_freq_hz: d.plasma_freq_hz, damping_freq_hz: d.damping_freq_hz,
            }
        } else {
            rapidfem::materials::Dispersion::None
        };
        if dispersion.is_dispersive() {
            eprintln!("    (dispersive: εr(f) recomputed per frequency)");
        }
        rapidfem::materials::Material {
            er: mc.er,
            ur: mc.ur,
            tand: mc.tand,
            cond: mc.conductivity,
            tet_indices,
            er_diag: mc.er_diag,
            ur_diag: mc.ur_diag,
            dispersion,
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

    // PML regions
    let pml_regions: Vec<rapidfem::materials::PmlRegion> = config.pml.iter().map(|pc| {
        let tet_indices = mesh.vtag_to_tet.get(&pc.volume_tag)
            .map(|v| v.clone())
            .unwrap_or_default();
        if tet_indices.is_empty() {
            eprintln!("  WARNING: PML volume tag {} has no tets", pc.volume_tag);
        } else {
            eprintln!("  PML: tag={}, dir=({:.0},{:.0},{:.0}), inner={:.3}m, t={:.3}m, n={:.1}, δmax={:.1}, {} tets",
                pc.volume_tag, pc.direction[0], pc.direction[1], pc.direction[2],
                pc.inner_face, pc.thickness, pc.exponent, pc.delta_max, tet_indices.len());
        }
        rapidfem::materials::PmlRegion {
            tet_indices,
            er_base: pc.er_base,
            ur_base: pc.ur_base,
            direction: pc.direction,
            inner_face: pc.inner_face,
            thickness: pc.thickness,
            exponent: pc.exponent,
            delta_max: pc.delta_max,
        }
    }).collect();
    let pml_opt = if pml_regions.is_empty() { None } else { Some(pml_regions.as_slice()) };

    // Solve
    let t0 = std::time::Instant::now();
    let results = frequency_sweep_with_pml(
        &mesh, &basis, &port_dyn_refs, &port_tri_refs, &pec_tris,
        &frequencies, materials_opt, pml_opt,
    );

    // EMerge-compatible lumped port integration lines: one line per min-projection node,
    // line goes from node to (node + direction * height). S-param averages over lines.
    // See microwave_3d.py:_define_lumped_port_integration_points (lines 524-562).
    let mut lumped_lines: std::collections::HashMap<usize, Vec<Vec<[f64; 3]>>> = std::collections::HashMap::new();
    for (pi, port) in port_dyn_refs.iter().enumerate() {
        if port.is_lumped() {
            let tri_ids = port_tri_refs[pi];
            let (dir, _z0, _v_inc) = port.lumped_voltage_params().unwrap();
            let height = port.port_height().expect("lumped port must have height");

            // Collect unique vertices on the port face
            let mut verts: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for &ti in tri_ids {
                for &vi in &mesh.tris[ti] { verts.insert(vi); }
            }

            // Find vertices with minimum projection onto direction (within tolerance)
            let mut min_proj = f64::INFINITY;
            for &vi in &verts {
                let p = mesh.nodes[vi];
                let proj = p[0]*dir[0] + p[1]*dir[1] + p[2]*dir[2];
                if proj < min_proj { min_proj = proj; }
            }
            let proj_tol = 1e-9 * height.max(1.0);
            let start_verts: Vec<usize> = verts.iter().copied().filter(|&vi| {
                let p = mesh.nodes[vi];
                let proj = p[0]*dir[0] + p[1]*dir[1] + p[2]*dir[2];
                (proj - min_proj).abs() < proj_tol
            }).collect();

            // Build one 21-point line per start vertex, end = start + direction * height
            let n_pts = 21;
            let mut lines: Vec<Vec<[f64; 3]>> = Vec::with_capacity(start_verts.len());
            for &vi in &start_verts {
                let s = mesh.nodes[vi];
                let mut pts = Vec::with_capacity(n_pts);
                for i in 0..n_pts {
                    let t = i as f64 / (n_pts - 1) as f64;
                    pts.push([s[0] + t*dir[0]*height, s[1] + t*dir[1]*height, s[2] + t*dir[2]*height]);
                }
                lines.push(pts);
            }
            eprintln!("  Lumped port {}: {} integration lines × {} pts, height={:.4}mm",
                port.port_number(), lines.len(), n_pts, height*1e3);
            lumped_lines.insert(pi, lines);
        }
    }

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
                let active = obs_idx == exc_idx;

                let s = if let (true, Some(lines), Some((_, z0, v_inc))) = (
                    port_dyn_refs[obs_pi].is_lumped(),
                    lumped_lines.get(&obs_pi),
                    port_dyn_refs[obs_pi].lumped_voltage_params(),
                ) {
                    // Lumped port: average S-param across multiple integration lines (EMerge convention)
                    let _ = z0;
                    let n_lines = lines.len() as f64;
                    let mut s_sum = C64::new(0.0, 0.0);
                    for line_pts in lines {
                        s_sum += sparam_voltage_line(v_inc, z0, active, &fieldf, line_pts);
                    }
                    s_sum / C64::from(n_lines)
                } else {
                    // Waveport: mode matching
                    let obs_tris: Vec<[usize; 3]> = port_tri_refs[obs_pi].iter().map(|&ti| mesh.tris[ti]).collect();
                    sparam_waveport(&mesh.nodes, &obs_tris, port_dyn_refs[obs_pi], k0, active, &fieldf, 4)
                };
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

    // Error estimation (if adaptive config present)
    if let Some(ref adaptive) = config.adaptive {
        if let Some(first_result) = results.first() {
            if let Some(first_sol) = first_result.solutions.first() {
                let k0 = 2.0 * PI * frequencies[0] / C0;
                // Build material tensors for error estimator
                let n_tets = mesh.n_tets();
                let (er_tensors, _) = if let Some(mats) = materials_opt {
                    rapidfem::materials::build_material_tensors(n_tets, mats, frequencies[0])
                } else {
                    let id: [[num_complex::Complex64; 3]; 3] = [
                        [num_complex::Complex64::new(1.0,0.0), num_complex::Complex64::new(0.0,0.0), num_complex::Complex64::new(0.0,0.0)],
                        [num_complex::Complex64::new(0.0,0.0), num_complex::Complex64::new(1.0,0.0), num_complex::Complex64::new(0.0,0.0)],
                        [num_complex::Complex64::new(0.0,0.0), num_complex::Complex64::new(0.0,0.0), num_complex::Complex64::new(1.0,0.0)],
                    ];
                    (vec![id; n_tets], vec![id; n_tets])
                };

                let estimate = rapidfem::error_estimator::estimate_error(
                    &mesh, &basis, first_sol, k0, &er_tensors, adaptive.theta,
                );

                // Write error VTK
                let error_path = config.output.vtk.as_deref()
                    .unwrap_or("error.vtk")
                    .replace(".vtk", "_error.vtk");
                rapidfem::vtk_export::write_vtk_error(&error_path, &mesh, &estimate)
                    .expect("Failed to write error VTK");
                eprintln!("Wrote error VTK: {}", error_path);

                // Write gmsh size field
                let size_path = config.mesh.file.replace(".msh", "_size.pos");
                rapidfem::error_estimator::write_size_field(
                    &size_path, &mesh, &estimate, adaptive.refinement_ratio,
                ).expect("Failed to write size field");
                eprintln!("Wrote size field: {}", size_path);

                eprintln!("To refine, load the size field in your gmsh script:");
                eprintln!("  gmsh.merge(\"{}\")  // then set as background field", size_path);
            }
        }
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

    // Group delay export τ_g = -dφ/dω with phase unwrapping per S-pair
    if let Some(ref path) = config.output.group_delay {
        if frequencies.len() >= 2 && n_driven >= 1 {
            let nf = frequencies.len();
            let mut file = std::fs::File::create(path).expect("Cannot create group delay CSV");
            use std::io::Write;
            write!(file, "freq_hz").unwrap();
            for i in 0..n_driven {
                for j in 0..n_driven {
                    write!(file, ",tau_g_{}{}_s", i + 1, j + 1).unwrap();
                }
            }
            writeln!(file).unwrap();

            // Per S-pair, unwrap phase across frequency
            let mut phase: Vec<Vec<f64>> = vec![vec![0.0; nf]; n_driven * n_driven];
            for i in 0..n_driven {
                for j in 0..n_driven {
                    let idx = i * n_driven + j;
                    let mut prev = all_sparams[0][i][j].arg();
                    phase[idx][0] = prev;
                    for fi in 1..nf {
                        let mut cur = all_sparams[fi][i][j].arg();
                        let diff = cur - prev;
                        if diff > PI { cur -= 2.0 * PI; }
                        else if diff < -PI { cur += 2.0 * PI; }
                        phase[idx][fi] = cur;
                        prev = cur;
                    }
                }
            }

            for fi in 0..nf {
                write!(file, "{:.6e}", frequencies[fi]).unwrap();
                for i in 0..n_driven {
                    for j in 0..n_driven {
                        let idx = i * n_driven + j;
                        // Central diff for interior, one-sided at ends
                        let tau = if fi == 0 {
                            -(phase[idx][1] - phase[idx][0]) / (2.0 * PI * (frequencies[1] - frequencies[0]))
                        } else if fi == nf - 1 {
                            -(phase[idx][nf-1] - phase[idx][nf-2]) / (2.0 * PI * (frequencies[nf-1] - frequencies[nf-2]))
                        } else {
                            -(phase[idx][fi+1] - phase[idx][fi-1]) / (2.0 * PI * (frequencies[fi+1] - frequencies[fi-1]))
                        };
                        write!(file, ",{:.6e}", tau).unwrap();
                    }
                }
                writeln!(file).unwrap();
            }
            eprintln!("Wrote group delay: {}", path);
        }
    }

    // Far-field radiation pattern (first frequency, first excitation)
    if let Some(ref ff_path) = config.output.farfield {
        if let Some(first_result) = results.first() {
            if let Some(first_sol) = first_result.solutions.first() {
                eprintln!("\n--- Far-field computation ---");

                // Find NFFT surface: use configured tag, or auto-detect ABC tag
                let nfft_tag = config.output.nfft_tag.unwrap_or_else(|| {
                    // Find the ABC port tag
                    for pc in &config.ports {
                        if let PortConfig::Abc { tag, .. } = pc {
                            return *tag;
                        }
                    }
                    eprintln!("  WARNING: no ABC tag found for NFFT, using tag 2");
                    2
                });

                let nfft_tris = mesh.tris_for_tag(nfft_tag).to_vec();
                if nfft_tris.is_empty() {
                    eprintln!("  ERROR: NFFT tag {} has no triangles", nfft_tag);
                } else {
                    eprintln!("  NFFT surface: tag={}, {} triangles", nfft_tag, nfft_tris.len());

                    let pattern = rapidfem::farfield::compute_farfield(
                        &mesh, &basis, first_sol, &nfft_tris,
                        frequencies[0], 91, 72, 4,
                    );

                    // Write full pattern CSV
                    rapidfem::farfield::write_pattern_csv(ff_path, &pattern)
                        .expect("Failed to write far-field CSV");
                    eprintln!("  Wrote far-field: {}", ff_path);

                    // Write plane cuts
                    let cuts_path = ff_path.replace(".csv", "_cuts.csv");
                    rapidfem::farfield::write_plane_cuts_csv(&cuts_path, &pattern)
                        .expect("Failed to write plane cuts CSV");
                    eprintln!("  Wrote plane cuts: {}", cuts_path);
                }
            }
        }
    }
}
