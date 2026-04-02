//! VTK field export for ParaView visualization.
//!
//! Writes unstructured grid (.vtk) with E-field interpolated to mesh nodes.

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::interp;

/// Export E-field solution to a legacy VTK file (.vtk).
///
/// Interpolates the Nedelec-2 FEM solution to mesh nodes and writes:
/// - E_real: real part of E-field (3-component vector)
/// - E_imag: imaginary part of E-field (3-component vector)
/// - E_mag: field magnitude (scalar)
pub fn write_vtk(
    path: &str,
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    solution: &[C64],
    label: &str,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    let n_nodes = mesh.n_nodes();
    let n_tets = mesh.n_tets();

    // Build node-to-tet mapping: for each node, pick one adjacent tet
    let mut node_to_tet = vec![usize::MAX; n_nodes];
    for (itet, tet) in mesh.tets.iter().enumerate() {
        for &ni in tet {
            if node_to_tet[ni] == usize::MAX {
                node_to_tet[ni] = itet;
            }
        }
    }

    // Interpolate E-field at each node
    let mut e_real = vec![[0.0f64; 3]; n_nodes];
    let mut e_imag = vec![[0.0f64; 3]; n_nodes];
    let mut e_mag = vec![0.0f64; n_nodes];

    for ni in 0..n_nodes {
        let tet_idx = node_to_tet[ni];
        if tet_idx == usize::MAX { continue; }

        let p = mesh.nodes[ni];
        let (ex, ey, ez) = interp::eval_field_in_tet(mesh, basis, solution, tet_idx, p[0], p[1], p[2]);
        e_real[ni] = [ex.re, ey.re, ez.re];
        e_imag[ni] = [ex.im, ey.im, ez.im];
        e_mag[ni] = (ex.norm_sqr() + ey.norm_sqr() + ez.norm_sqr()).sqrt();
    }

    // Write legacy VTK format
    writeln!(file, "# vtk DataFile Version 3.0")?;
    writeln!(file, "RapidFEM {}", label)?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET UNSTRUCTURED_GRID")?;

    // Points
    writeln!(file, "POINTS {} double", n_nodes)?;
    for ni in 0..n_nodes {
        let p = mesh.nodes[ni];
        writeln!(file, "{:.10e} {:.10e} {:.10e}", p[0], p[1], p[2])?;
    }

    // Cells (tetrahedra: 4 nodes each, preceded by count 4)
    writeln!(file, "CELLS {} {}", n_tets, n_tets * 5)?;
    for tet in &mesh.tets {
        writeln!(file, "4 {} {} {} {}", tet[0], tet[1], tet[2], tet[3])?;
    }

    // Cell types (10 = VTK_TETRA)
    writeln!(file, "CELL_TYPES {}", n_tets)?;
    for _ in 0..n_tets {
        writeln!(file, "10")?;
    }

    // Point data
    writeln!(file, "POINT_DATA {}", n_nodes)?;

    // E-field real part
    writeln!(file, "VECTORS E_real double")?;
    for ni in 0..n_nodes {
        writeln!(file, "{:.10e} {:.10e} {:.10e}", e_real[ni][0], e_real[ni][1], e_real[ni][2])?;
    }

    // E-field imaginary part
    writeln!(file, "VECTORS E_imag double")?;
    for ni in 0..n_nodes {
        writeln!(file, "{:.10e} {:.10e} {:.10e}", e_imag[ni][0], e_imag[ni][1], e_imag[ni][2])?;
    }

    // E-field magnitude
    writeln!(file, "SCALARS E_mag double 1")?;
    writeln!(file, "LOOKUP_TABLE default")?;
    for ni in 0..n_nodes {
        writeln!(file, "{:.10e}", e_mag[ni])?;
    }

    Ok(())
}
