// build.rs — compiles the Apple Accelerate shim on macOS only.
//
// The shim is a thin C wrapper around Apple's SparseFactor / SparseSolve /
// SparseCleanup so the Rust side doesn't have to chase the C++-overloaded
// symbols those expose. On other platforms this file does nothing.

fn main() {
    #[cfg(target_os = "macos")]
    {
        let shim = "src/solver/accelerate_shim.c";
        println!("cargo:rerun-if-changed={shim}");
        cc::Build::new()
            .file(shim)
            .flag_if_supported("-O3")
            .compile("rapidfem_accelerate_shim");
        // Link Apple Accelerate framework — it provides the sparse solvers.
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
