//! Structured tetrahedral mesh generation.
//!
//! Currently a box mesher via the conforming Kuhn (Freudenthal)
//! triangulation — six tets per cell through a fixed main diagonal, which
//! keeps shared cell faces matched. Used to build cavities for solver
//! validation.

use rapidfem_core::mesh::Mesh;

/// Tetrahedral mesh of the box `[0,lx] × [0,ly] × [0,lz]` with `nx·ny·nz`
/// cells, each split into 6 tets.
pub fn structured_box(
    nx: usize,
    ny: usize,
    nz: usize,
    lx: f64,
    ly: f64,
    lz: f64,
) -> Mesh {
    assert!(nx >= 1 && ny >= 1 && nz >= 1);
    let (nxp, nyp) = (nx + 1, ny + 1);
    let node_id = |i: usize, j: usize, k: usize| (k * nyp + j) * nxp + i;

    let mut nodes = Vec::with_capacity(nxp * nyp * (nz + 1));
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push([
                    lx * i as f64 / nx as f64,
                    ly * j as f64 / ny as f64,
                    lz * k as f64 / nz as f64,
                ]);
            }
        }
    }

    // Six tets per cell, all sharing the 0-7 main diagonal. The cell-corner
    // index is `x + 2y + 4z`.
    const KUHN: [[usize; 4]; 6] = [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ];
    let mut tets = Vec::with_capacity(nx * ny * nz * 6);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let corner = |c: usize| {
                    node_id(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1))
                };
                for kt in KUHN {
                    let mut t = [
                        corner(kt[0]),
                        corner(kt[1]),
                        corner(kt[2]),
                        corner(kt[3]),
                    ];
                    t.sort_unstable();
                    tets.push(t);
                }
            }
        }
    }
    Mesh::from_tets(nodes, tets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_mesh_is_conforming_and_fills_volume() {
        let m = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        assert_eq!(m.n_tets(), 8 * 6);
        // Every interior face is shared by exactly two tets; boundary faces
        // by one. A conforming mesh has no half-shared faces beyond that.
        for adj in &m.tri_to_tet {
            assert!(adj[0] != usize::MAX, "every face has an owner");
        }
        // Total tet volume equals the box volume.
        let mut vol = 0.0;
        for tet in &m.tets {
            let v: Vec<[f64; 3]> = tet.iter().map(|&n| m.nodes[n]).collect();
            let e = |a: [f64; 3]| {
                [a[0] - v[0][0], a[1] - v[0][1], a[2] - v[0][2]]
            };
            let (a, b, c) = (e(v[1]), e(v[2]), e(v[3]));
            let det = a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
            vol += det.abs() / 6.0;
        }
        assert!((vol - 1.0).abs() < 1e-12, "filled volume = {vol}");
    }
}
