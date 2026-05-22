# RapidFEM Time-Domain Backend — GPU Plan (mixed precision, OpenCL)

## Goal

An optional GPU execution path for the time-domain (DGTD) backend. The
matrix-free operator, the explicit LSERK4 transient, and (conditionally)
the exponential propagator run on the GPU, with the field state and the
operator data resident in device memory so a transient loop crosses the
PCIe bus only for snapshots.

The path is **optional and feature-gated**: with no GPU or no OpenCL
runtime the existing CPU path runs unchanged, and the wheel stays
installable for everyone.

## Where this fits

Follow-up to `td-backend-plan.md` and `td-production-plan.md`. Those
delivered the CPU DGTD backend: the matrix-free operator, the Krylov
exponential propagator, the explicit LSERK4 integrator (v0.9.0), and a
matvec performance pass (coarse-chunked `apply`, ~2x at high core counts).

The CPU benchmark established the regime: on a real unstructured mesh at
~1.3M state DOF, the explicit LSERK4 path is ~2.9x cheaper per unit of
simulated time than the exponential propagator, and `apply` (the matvec)
is the hot path of that explicit path (five matvecs per step). GPU
acceleration targets that hot path.

## Guiding principles

1. **The CPU f64 path is the golden reference.** Every GPU kernel is
   cross-validated against it. It is never removed.
2. **Data residency over kernel offload.** The win is keeping the field
   state and operator data on the device across the whole step loop; a
   per-step host round-trip would cancel it. Snapshots cross on the output
   cadence only.
3. **Mixed precision is two regimes, not one knob.** See below.
4. **Validate before building on top.** Each phase has an analytic or
   CPU-cross-check gate that must pass before the next.
5. **Optional and non-invasive.** Feature-gated, runtime-detected, with a
   CPU fallback. The CPU backend is not touched.

## The mixed-precision policy

`alles f32` does not work. Mixed precision here means **two precision
regimes, one per integrator.**

The **explicit LSERK4 path tolerates f32 natively**: it is a fourth-order
scheme, f32 round-off sits below its truncation error. The **exponential
propagator fights f32**: its reason to exist is exactness, and an f32
matvec caps the reachable Krylov accuracy at roughly 1e-6 to 1e-7, so
`KRYLOV_TOL = 1e-10` becomes unreachable. The holistic answer: the
explicit path is f32-dominant, the exponential propagator stays f64.

### Precision per quantity

| Quantity | Precision | Rationale |
|---|---|---|
| EM field state `y` (E, H, P) | f32 | the large array, bandwidth, device residency |
| `apply` / matvec arithmetic | f32 | throughput (consumer GPU: f32 ~60x its f64 rate) |
| geometric factors, lift, diff matrices, material tensors | f32 | operator inputs, geometry precision is ample |
| LSERK4 registers `y`, `p` (time accumulation) | f32 or f64 (set by the budget, see below) | round-off drift over many steps |
| Arnoldi basis, CGS2 orthogonalisation | f64 | f32 loses orthogonality fast; bandwidth-bound, so f64 on GPU is still fast |
| matvec inside Arnoldi | f32 `apply` with a cast at the boundary, or f64 | the P3 two-regime decision |
| dense `expm(H)`, Hessenberg `H` | f64 | small, accuracy-critical |
| CFL power iteration | f32 | only needs the spectral radius approximately |
| sparse assembly (`state_space`) | f64, CPU | not on the hot path, exactness for export |

### The tolerance constant

The acceptable relative L2 error of the f32 / GPU path against the CPU
f64 reference is a **single named constant** in
`crates/rapidfem-td/src/constants.rs` (working name `GPU_REL_TOL`).
Every GPU validation gate references it. Tuning the accuracy budget is a
one-line edit, never a scattered per-test literal. Its initial value is
fixed from the P0.2 CPU-f32 study.

### Foundation: the Field / Accum seam

The `Field` / `Accum` type aliases (already in `constants.rs`) are the
foundation. In the GPU build `Field` becomes f32; the kernels carry a
matching `#define REAL float` (the FluidX3D `fpxx` pattern). `Accum`
stays f64. The matvec boundary in `expmv_into` is where they meet.

### Two named f32 risks

- **DG flux cancellation.** The numerical flux forms jumps
  `[E] = E_minus - E_plus` between neighbouring elements. For a smooth,
  well-resolved field the two traces are close, so the f32 difference
  loses relative digits. The literature finds f32 DGTD generally
  tolerable (the jump is small, so its absolute contribution is small,
  and the upwind penalty is dissipative), but this needs a validation
  gate.
- **Explicit accumulation drift.** An f32 state over 1e5 to 1e6 steps
  drifts by roughly `sqrt(N) * eps`. If that exceeds the budget, the
  LSERK4 `y` / `p` registers move to f64 while the matvec stays f32.

## Architecture

- New module tree `crates/rapidfem-td/src/gpu/` (or a `rapidfem-gpu`
  crate): the OpenCL host layer (device discovery, context, command
  queue, a buffer abstraction, kernel build-from-source) and the kernel
  sources as `.cl` strings.
- Rust binding: `opencl3` (maintained, OpenCL 3.0, thin).
- **GPU field layout**: a struct-of-arrays layout, not the CPU
  node-major interleaved `[node*6 + field]`, so neighbouring work-items
  read contiguous memory (coalescing). Conversion happens at the
  host/device boundary.
- **Kernel organisation**: one work-group per element (or a few), work-
  items mapped to (element, node); the reference matrices (diff, lift)
  staged in local memory; the volume curl becomes a small local-memory
  matmul. This is the established nodal-DG GPU pattern (Hesthaven and
  Warburton; Kloeckner).

## Phases

| Phase | Content | Gate |
|---|---|---|
| **P0 Foundations** | OpenCL host layer; buffer and kernel abstraction; the precision policy and `GPU_REL_TOL`; operator-data upload (geometric factors, lift, materials, topology) | A trivial kernel runs on the target GPU. And: the CPU f32 build (`Field = f32`) passes the test suite with loosened tolerances, which separates f32 numerical error from GPU bugs before any kernel exists. |
| **P1 GPU `apply`** | GPU field layout (SoA); volume-curl kernel; flux kernel (numerical flux, neighbour gather, BC-as-flux, the hardest); materials and the dispersive polarisation current | GPU `apply` vs CPU f64 `apply` within `GPU_REL_TOL` on the validation meshes; benchmark vs CPU `apply`. |
| **P2 GPU LSERK4 transient** | LSERK4 stage kernels; the state-resident transient loop (only snapshots cross); driven soft source on GPU; CFL calibration on GPU; the exponential-warmup hybrid | GPU transient vs CPU transient within `GPU_REL_TOL`; benchmark. The main expected win. |
| **Decision gate** | With P2 benchmarked, decide P3: is the explicit GPU path fast enough on its own, or is a GPU exponential propagator worth it, and in which precision regime (f64 on GPU vs left on the CPU)? | A measured decision, recorded here. |
| **P3 GPU exponential propagator** | Arnoldi basis device-resident (f64); CGS2 orthogonalisation kernels (f64, bandwidth-bound); the matvec boundary; `expm` on the host | Energy conservation; vs CPU; the accuracy ceiling documented. |
| **P4 MOR on GPU** | reuses the P3 Arnoldi | reduced model vs CPU. Lower priority. |
| **P5 Python API** | `ProblemTD` device selection; runtime OpenCL detection; CPU fallback; `transient` / `stepper` dispatch | the TD examples run with the GPU device flag; the fallback works. |
| **P6 Packaging and CI** | feature gate `--features gpu`; the ICD loader is a system library (no build-time toolkit); a GPU CI runner, or at minimum a build test of the gpu feature | `pip install rapidfem` with GPU auto-detection and CPU fallback. |

## Validation strategy

The CPU f64 path is the reference throughout. P0.2 is the key step: a
CPU build with `Field = f32` (over the existing type seam) runs the test
suite and isolates the cost of f32 alone, before any kernel exists. That
fixes `GPU_REL_TOL`. After that, every kernel has its own CPU-vs-GPU
gate against that one constant.

The accuracy study deferred earlier is now on the critical path: it is
P0.2, and it sets the budget the whole plan validates against.

## Risks and open decisions

- **The f32 accuracy budget** (`GPU_REL_TOL`): set by P0.2, the single
  knob the gates reference.
- **The P3 two-regime decision**: deferred to the post-P2 gate by design.
- **DG flux cancellation in f32**: see above; gated in P1.
- **Kernel maintenance**: the OpenCL C kernels are source separate from
  the Rust `apply`. Two codebases for the operator. Mitigation: the CPU
  Rust `apply` stays the reference and the kernel is validated against
  it on every change.
- **CI**: real GPU tests need a GPU runner; without one, CI build-tests
  the gpu feature only.

## Scope boundaries

Deliberately out of scope: local time stepping (a separate effort), the
frequency-domain backend, and P4 (lower priority). The explicit path
(P1, P2, P5) is the core deliverable; P3 is the harder, separately
decided extension.
