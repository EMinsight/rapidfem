fn main() {
    eprintln!("rapidfem — EMerge port to Rust");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rapidfem <mesh.msh>");
        return;
    }

    let mesh = rapidfem::mesh_io::load_mesh(&args[1]).expect("Failed to load mesh");
    let basis = rapidfem::basis::Nedelec2Basis::new(&mesh);

    eprintln!("DOFs: {} (2×{} edges + 2×{} tris)",
        basis.n_field, mesh.n_edges(), mesh.n_tris());
}
