/// Unit tests for waveguide.rs against EMerge reference values.
use rapidfem::waveguide::{RectWaveguide, CoordinateSystem};
use rapidfem::constants::*;

fn make_test_port() -> RectWaveguide {
    // EMerge port 1 for Box(22.86mm, 30mm, 10.16mm) at origin, wg.front face
    let cs = CoordinateSystem::new(
        [0.01143, 0.0, 0.00508],  // origin
        [1.0, 0.0, 0.0],          // xax
        [0.0, 0.0, 1.0],          // yax (EMerge: [0, -0, 1] ≈ [0, 0, 1])
        [0.0, -1.0, 0.0],         // zax
    );
    RectWaveguide {
        port_number: 1,
        power: 1.0,
        mode: (1, 0),
        er: 1.0,
        polarization: 1.0,
        dims: (22.86e-3, 10.16e-3),
        cs,
    }
}

#[test]
fn test_waveguide_beta_gamma_zmode() {
    let port = make_test_port();
    let k0 = 2.0 * PI * 10.0e9 / C0;

    let beta = port.get_beta(k0);
    let gamma = port.get_gamma(k0);
    let zmode = port.z_mode(k0);
    let amp = port.get_amplitude(k0);
    let qm = port.qmode(k0);

    eprintln!("beta = {:.15e}", beta);
    eprintln!("gamma = {:.15e}", gamma);
    eprintln!("Zmode = {:.15e}", zmode);
    eprintln!("amplitude = {:.15e}", amp);
    eprintln!("qmode = {:.15e}", qm);

    assert!((beta - 1.582382563130197e+02).abs() < 1e-6);
    assert!((gamma.im - 1.582382563130197e+02).abs() < 1e-6);
    assert!((zmode - 4.989743759681103e+02).abs() < 1e-6);
    assert!((amp - 2.547183965900364e+03).abs() < 1e-3);
    assert!((qm - 1.150863557661104e+00).abs() < 1e-9);

    eprintln!("beta/gamma/Zmode/amp/qmode: PASS");
}

#[test]
fn test_waveguide_mode_field() {
    let port = make_test_port();
    let k0 = 2.0 * PI * 10.0e9 / C0;

    let (ex, ey, ez) = port.port_mode_3d_global(0.01, 0.0, 0.005, k0);

    eprintln!("mode_field_global(0.01, 0, 0.005) = ({:.15e}, {:.15e}, {:.15e})", ex, ey, ez);

    // EMerge: (0, 0, 2.875035710170894e+03)
    assert!(ex.abs() < 1e-10, "Ex should be 0, got {}", ex);
    assert!(ey.abs() < 1e-10, "Ey should be 0, got {}", ey);
    assert!((ez - 2.875035710170894e+03).abs() < 1e-3, "Ez wrong: {}", ez);

    eprintln!("port_mode_3d_global: PASS");
}

#[test]
fn test_waveguide_uinc() {
    let port = make_test_port();
    let k0 = 2.0 * PI * 10.0e9 / C0;

    let ui = port.get_uinc(0.01, 0.0, 0.005, k0);

    eprintln!("get_uinc(0.01, 0, 0.005) = ({:.6e}, {:.6e}, {:.6e})", ui[0], ui[1], ui[2]);

    // EMerge: (0, 0, -9.098812752302133e+05j)
    assert!(ui[0].norm() < 1e-6);
    assert!(ui[1].norm() < 1e-6);
    assert!((ui[2].im - (-9.098812752302133e+05)).abs() < 1e0, "Uinc_z wrong: {}", ui[2]);

    eprintln!("get_uinc: PASS");
}
