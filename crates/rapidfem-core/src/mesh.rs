// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
// Copyright (C) Robert Fennis (original EMerge source)
//
// This file is part of rapidfem and contains code ported from EMerge
// (https://github.com/FennisRobert/EMerge), originally licensed under
// GPL-2.0-or-later with the Gmsh additional permission; redistributed
// here under GPL-3.0-or-later with that permission preserved.
// See LICENSE and NOTICE for the full terms.

//! Mesh data structure: nodes, edges, tris, tets, and connectivity.
//! Mirrors emerge/_emerge/mesh3d.py.

use hashbrown::HashMap;

/// Edge local index ordering within a tetrahedron (1-indexed node pairs).
/// Mirrors EMerge's _idset1 = ((1,2),(1,3),(1,4),(2,3),(4,2),(3,4))
/// Note: (4,2) not (2,4), this ordering is critical for basis function orientation.
pub const TET_EDGE_LOCAL: [[usize; 2]; 6] = [
    [0, 1], // (1,2) in 1-indexed
    [0, 2], // (1,3)
    [0, 3], // (1,4)
    [1, 2], // (2,3)
    [3, 1], // (4,2), note reversed!
    [2, 3], // (3,4)
];

/// Face local index ordering within a tetrahedron (1-indexed node triples).
/// Mirrors EMerge's _idset2 = ((1,2,3),(1,3,4),(1,4,2),(2,3,4))
/// Note: (1,4,2) not (1,2,4), this ordering is critical.
pub const TET_FACE_LOCAL: [[usize; 3]; 4] = [
    [0, 1, 2], // (1,2,3)
    [0, 2, 3], // (1,3,4)
    [0, 3, 1], // (1,4,2), note reversed!
    [1, 2, 3], // (2,3,4)
];

pub struct Mesh {
    /// Node coordinates: nodes[i] = [x, y, z]
    pub nodes: Vec<[f64; 3]>,
    /// Edges: edges[e] = [n1, n2] sorted (min, max)
    pub edges: Vec<[usize; 2]>,
    /// Triangles: tris[t] = [n1, n2, n3] sorted
    pub tris: Vec<[usize; 3]>,
    /// Tetrahedra: tets[t] = [n1, n2, n3, n4] in gmsh node order (sorted)
    pub tets: Vec<[usize; 4]>,

    /// Per-tet: 6 edge indices in TET_EDGE_LOCAL order
    pub tet_to_edge: Vec<[usize; 6]>,
    /// Per-tet: 4 tri indices in TET_FACE_LOCAL order
    pub tet_to_tri: Vec<[usize; 4]>,
    /// Per-tri: 3 edge indices
    pub tri_to_edge: Vec<[usize; 3]>,
    /// Per-tri: up to 2 adjacent tet indices (usize::MAX = no neighbor)
    pub tri_to_tet: Vec<[usize; 2]>,

    /// Edge lengths
    pub edge_lengths: Vec<f64>,

    /// Inverse maps for fast lookup during construction
    pub inv_edges: HashMap<(usize, usize), usize>,
    pub inv_tris: HashMap<(usize, usize, usize), usize>,

    /// Gmsh face tag → list of triangle indices
    pub ftag_to_tri: HashMap<i32, Vec<usize>>,
    /// Gmsh volume tag → list of tet indices
    pub vtag_to_tet: HashMap<i32, Vec<usize>>,
}

impl Mesh {
    /// Build all connectivity from raw nodes and tets.
    /// Extracts edges and triangles from tetrahedra, builds inverse maps.
    pub fn from_tets(nodes: Vec<[f64; 3]>, tets: Vec<[usize; 4]>) -> Self {
        let n_tets = tets.len();
        let mut inv_edges: HashMap<(usize, usize), usize> = HashMap::new();
        let mut inv_tris: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut edges: Vec<[usize; 2]> = Vec::new();
        let mut tris: Vec<[usize; 3]> = Vec::new();
        let mut tet_to_edge = vec![[0usize; 6]; n_tets];
        let mut tet_to_tri = vec![[0usize; 4]; n_tets];

        for (ti, tet) in tets.iter().enumerate() {
            // Extract 6 edges
            for (ei, &[li, lj]) in TET_EDGE_LOCAL.iter().enumerate() {
                let (a, b) = (tet[li], tet[lj]);
                let key = if a < b { (a, b) } else { (b, a) };
                let edge_idx = *inv_edges.entry(key).or_insert_with(|| {
                    let idx = edges.len();
                    edges.push([key.0, key.1]);
                    idx
                });
                tet_to_edge[ti][ei] = edge_idx;
            }

            // Extract 4 faces
            for (fi, &[li, lj, lk]) in TET_FACE_LOCAL.iter().enumerate() {
                let mut face = [tet[li], tet[lj], tet[lk]];
                face.sort();
                let key = (face[0], face[1], face[2]);
                let tri_idx = *inv_tris.entry(key).or_insert_with(|| {
                    let idx = tris.len();
                    tris.push(face);
                    idx
                });
                tet_to_tri[ti][fi] = tri_idx;
            }
        }

        // Build tri_to_edge, must match EMerge's ordering:
        // tri_to_edge[0] = edge(sorted[0], sorted[1])
        // tri_to_edge[1] = edge(sorted[1], sorted[2])
        // tri_to_edge[2] = edge(sorted[0], sorted[2])
        let n_tris = tris.len();
        let mut tri_to_edge = vec![[0usize; 3]; n_tris];
        for (ti, tri) in tris.iter().enumerate() {
            let edge_pairs = [
                (tri[0].min(tri[1]), tri[0].max(tri[1])), // edge(0,1)
                (tri[1].min(tri[2]), tri[1].max(tri[2])), // edge(1,2), EMerge order!
                (tri[0].min(tri[2]), tri[0].max(tri[2])), // edge(0,2)
            ];
            for (ei, &key) in edge_pairs.iter().enumerate() {
                tri_to_edge[ti][ei] = inv_edges[&key];
            }
        }

        // Build tri_to_tet. An interior face is shared by exactly two tets,
        // a boundary face by one. A face shared by three or more tets means
        // a non-manifold mesh; report it rather than silently overwriting
        // slot [1], which would corrupt the DG face-jump terms downstream.
        let mut tri_to_tet = vec![[usize::MAX; 2]; n_tris];
        let mut non_manifold = 0usize;
        for (ti, tet_tris) in tet_to_tri.iter().enumerate() {
            for &tri_idx in tet_tris {
                if tri_to_tet[tri_idx][0] == usize::MAX {
                    tri_to_tet[tri_idx][0] = ti;
                } else if tri_to_tet[tri_idx][1] == usize::MAX {
                    tri_to_tet[tri_idx][1] = ti;
                } else {
                    non_manifold += 1;
                }
            }
        }
        if non_manifold > 0 {
            eprintln!(
                "WARNING: non-manifold mesh: {} face-tet incidences beyond \
                 the two-per-face limit were dropped; face-jump terms on \
                 those faces will be wrong",
                non_manifold
            );
        }

        // Compute edge lengths
        let edge_lengths: Vec<f64> = edges.iter().map(|&[a, b]| {
            let dx = nodes[b][0] - nodes[a][0];
            let dy = nodes[b][1] - nodes[a][1];
            let dz = nodes[b][2] - nodes[a][2];
            (dx*dx + dy*dy + dz*dz).sqrt()
        }).collect();

        Mesh {
            nodes, edges, tris, tets,
            tet_to_edge, tet_to_tri, tri_to_edge, tri_to_tet,
            edge_lengths, inv_edges, inv_tris,
            ftag_to_tri: HashMap::new(),
            vtag_to_tet: HashMap::new(),
        }
    }

    pub fn n_nodes(&self) -> usize { self.nodes.len() }
    pub fn n_edges(&self) -> usize { self.edges.len() }
    pub fn n_tris(&self) -> usize { self.tris.len() }
    pub fn n_tets(&self) -> usize { self.tets.len() }

    /// Get boundary triangles (only one adjacent tet).
    pub fn boundary_tris(&self) -> Vec<usize> {
        (0..self.n_tris())
            .filter(|&i| self.tri_to_tet[i][1] == usize::MAX)
            .collect()
    }

    /// Get triangles for a face tag.
    pub fn tris_for_tag(&self, tag: i32) -> &[usize] {
        self.ftag_to_tri.get(&tag).map_or(&[], |v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_manifold_face_keeps_the_first_two_tets() {
        // Three tets all listing nodes {0,1,2} as a face is a non-manifold
        // connectivity. from_tets must keep the first two incidences in
        // tri_to_tet rather than overwriting slot [1] with the third.
        let nodes = vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0],
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]];
        let mesh = Mesh::from_tets(nodes, tets);
        assert_eq!(mesh.n_tets(), 3);

        let shared = mesh.inv_tris[&(0, 1, 2)];
        let adj = mesh.tri_to_tet[shared];
        assert_eq!(adj[0], 0, "first tet kept");
        assert_eq!(adj[1], 1, "second tet kept, not overwritten by the third");
    }
}
