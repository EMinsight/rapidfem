//! Load .msh files via mshio and build a Mesh.
//! Mirrors the gmsh extraction in mesh3d._pre_update().

use crate::mesh::Mesh;
use mshio::mshfile::ElementType;
use std::collections::HashMap;

/// Load a gmsh .msh file from disk and build a Mesh.
/// Convenience wrapper around `parse_mesh_bytes`.
pub fn load_mesh(path: &str) -> Result<Mesh, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Cannot read {}: {}", path, e))?;
    parse_mesh_bytes(&bytes).map_err(|e| format!("{}: {}", path, e))
}

/// Parse a gmsh .msh file from in-memory bytes and build a Mesh with full connectivity.
/// This is the core loader; `load_mesh` is a thin file-system wrapper around it.
/// Used by Python (numpy bytes) and WASM (Uint8Array) bindings.
pub fn parse_mesh_bytes(bytes: &[u8]) -> Result<Mesh, String> {
    let msh = mshio::parse_msh_bytes(bytes)
        .map_err(|e| format!("Cannot parse mesh bytes: {:?}", e))?;

    // Extract nodes — build a global node list and tag-to-index map
    let msh_nodes = msh.data.nodes
        .as_ref().ok_or("No nodes in mesh")?;

    // Node tags in gmsh are 1-based. We need to map tag → contiguous index.
    let total_nodes = msh_nodes.num_nodes as usize;
    let mut nodes = Vec::with_capacity(total_nodes);
    let mut tag_to_idx: HashMap<u64, usize> = HashMap::with_capacity(total_nodes);

    for block in &msh_nodes.node_blocks {
        for (i, node) in block.nodes.iter().enumerate() {
            let idx = nodes.len();
            // Node tag: if sparse tags exist, use the HashMap; otherwise sequential
            let tag = if let Some(ref tag_map) = block.node_tags {
                // tag_map maps tag → local index, but we need the inverse
                // Actually, node_tags maps tag → index within the block
                // We need to find the tag for index i
                // This is inefficient but mshio doesn't give us a better way
                let mut found_tag = 0u64;
                for (&t, &local_i) in tag_map.iter() {
                    if local_i == i {
                        found_tag = t;
                        break;
                    }
                }
                found_tag
            } else {
                // Sequential tags: first block starts at min_node_tag
                // Actually, each NodeBlock just stores nodes sequentially
                // and the tags are implicit from position within the Nodes struct
                // For sequential tags, tag = min_node_tag + global_offset
                // But we don't track that easily. Let's just use the linear index.
                // Actually mshio gives us min_node_tag and max_node_tag on the Nodes struct.
                // For simple meshes, tags are 1..N.
                msh_nodes.min_node_tag + idx as u64
            };
            tag_to_idx.insert(tag, idx);
            nodes.push([node.x, node.y, node.z]);
        }
    }

    // Extract elements
    let msh_elements = msh.data.elements
        .as_ref().ok_or("No elements in mesh")?;

    let mut tets: Vec<[usize; 4]> = Vec::new();
    let mut tet_entity_tags: Vec<i32> = Vec::new();
    let mut surface_tris: Vec<(i32, [usize; 3])> = Vec::new();

    for block in &msh_elements.element_blocks {
        let etype = block.element_type;
        let entity_tag = block.entity_tag;

        match etype {
            ElementType::Tet4 => {
                for el in &block.elements {
                    let n: Vec<usize> = el.nodes.iter()
                        .map(|&t| *tag_to_idx.get(&t).expect("Unknown node tag in tet"))
                        .collect();
                    let tet = [n[0], n[1], n[2], n[3]];
                    tets.push(tet);
                    tet_entity_tags.push(entity_tag);
                }
            }
            ElementType::Tri3 => {
                for el in &block.elements {
                    let n: Vec<usize> = el.nodes.iter()
                        .map(|&t| *tag_to_idx.get(&t).expect("Unknown node tag in tri"))
                        .collect();
                    let mut tri = [n[0], n[1], n[2]];
                    tri.sort();
                    surface_tris.push((entity_tag, tri));
                }
            }
            _ => {} // Skip other element types (lines, points, etc.)
        }
    }

    if tets.is_empty() {
        return Err("No tetrahedra found in mesh".to_string());
    }

    // Build mesh with connectivity
    let mut mesh = Mesh::from_tets(nodes, tets);

    // Map entity tags to physical tags using the entities section
    let mut entity_to_physical_2d: HashMap<i32, i32> = HashMap::new();
    let mut entity_to_physical_3d: HashMap<i32, i32> = HashMap::new();

    if let Some(ref entities) = msh.data.entities {
        for surf in &entities.surfaces {
            if !surf.physical_tags.is_empty() {
                entity_to_physical_2d.insert(surf.tag, surf.physical_tags[0] as i32);
            }
        }
        for vol in &entities.volumes {
            if !vol.physical_tags.is_empty() {
                entity_to_physical_3d.insert(vol.tag, vol.physical_tags[0] as i32);
            }
        }
    }

    // Build ftag_to_tri: physical tag → mesh triangle indices
    for (entity_tag, tri_nodes) in &surface_tris {
        let physical_tag = entity_to_physical_2d.get(entity_tag).copied().unwrap_or(*entity_tag);
        let key = (tri_nodes[0], tri_nodes[1], tri_nodes[2]);
        if let Some(&tri_idx) = mesh.inv_tris.get(&key) {
            mesh.ftag_to_tri.entry(physical_tag).or_default().push(tri_idx);
        }
    }

    // Build vtag_to_tet
    for (ti, &entity_tag) in tet_entity_tags.iter().enumerate() {
        let physical_tag = entity_to_physical_3d.get(&entity_tag).copied().unwrap_or(entity_tag);
        mesh.vtag_to_tet.entry(physical_tag).or_default().push(ti);
    }

    eprintln!("Mesh: {} nodes, {} edges, {} tris, {} tets",
        mesh.n_nodes(), mesh.n_edges(), mesh.n_tris(), mesh.n_tets());

    Ok(mesh)
}
