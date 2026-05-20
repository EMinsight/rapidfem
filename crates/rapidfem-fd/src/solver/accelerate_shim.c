/*
 * Thin C shim around Apple's Accelerate sparse Bunch-Kaufman solver.
 *
 * Apple exposes `SparseFactor` / `SparseSolve` as `__attribute__((overloadable))`
 * inline functions; calling them via Rust FFI directly means picking the
 * right mangled symbol, which is fragile across SDK versions. The shim wraps
 * them with three plain C entry points that the Rust side binds with stable
 * names.
 *
 * Used for the macOS solver path — complex-symmetric A (size N) is
 * reformulated as a real symmetric INDEFINITE matrix M of size 2N (see
 * `accelerate.rs` for the block layout), then factored here with
 * `SparseFactorizationLDLTSBK` (supernodal Bunch-Kaufman).
 */

#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    SparseOpaqueSymbolicFactorization symb;
    SparseOpaqueFactorization_Double  fact;
    int     n;
    long   *col_starts;   /* length n+1, owned */
    int    *row_idx;      /* length nnz, owned */
    double *values;       /* length nnz, owned */
} AccelLDLT;

static SparseSymbolicFactorOptions default_sf_options(void) {
    SparseSymbolicFactorOptions o;
    o.control              = SparseDefaultControl;
    o.orderMethod          = SparseOrderDefault;   /* AMD-style ordering */
    o.order                = NULL;
    o.ignoreRowsAndColumns = NULL;
    o.malloc               = malloc;
    o.free                 = free;
    o.reportError          = NULL;
    return o;
}

/* Factorize a real-symmetric INDEFINITE matrix in CSC upper-triangle. */
void *accel_ldlt_factorize(int n, const long *col_starts, const int *row_idx, const double *values) {
    if (n <= 0) return NULL;
    long nnz = col_starts[n];

    AccelLDLT *h = (AccelLDLT *)calloc(1, sizeof(*h));
    if (!h) return NULL;
    h->n          = n;
    h->col_starts = (long *)malloc(sizeof(long) * (size_t)(n + 1));
    h->row_idx    = (int  *)malloc(sizeof(int)  * (size_t)nnz);
    h->values     = (double *)malloc(sizeof(double) * (size_t)nnz);
    if (!h->col_starts || !h->row_idx || !h->values) goto fail;
    memcpy(h->col_starts, col_starts, sizeof(long) * (size_t)(n + 1));
    memcpy(h->row_idx,    row_idx,    sizeof(int)  * (size_t)nnz);
    memcpy(h->values,     values,     sizeof(double) * (size_t)nnz);

    SparseAttributes_t attr = {0};
    attr.kind     = SparseSymmetric;
    attr.triangle = SparseUpperTriangle;

    SparseMatrixStructure structure;
    structure.rowCount     = n;
    structure.columnCount  = n;
    structure.columnStarts = h->col_starts;
    structure.rowIndices   = h->row_idx;
    structure.attributes   = attr;
    structure.blockSize    = 1;

    SparseMatrix_Double mat;
    mat.structure = structure;
    mat.data      = h->values;

    /* Symbolic: SparseFactor(type, MatrixStructure, options) — picks the
     * overload returning SparseOpaqueSymbolicFactorization. SBK is the
     * supernodal Bunch-Kaufman variant appropriate for the symmetric
     * indefinite real-block matrix. */
    SparseSymbolicFactorOptions sfopts = default_sf_options();
    h->symb = SparseFactor(SparseFactorizationLDLTSBK, structure, sfopts);
    if (h->symb.status != SparseStatusOK) goto fail;

    /* Numeric: SparseFactor(symbolic, Matrix) — overload returning
     * SparseOpaqueFactorization_Double, reusing the symbolic factor. */
    h->fact = SparseFactor(h->symb, mat);
    if (h->fact.status != SparseStatusOK) goto fail_symb;

    return (void *)h;

fail_symb:
    SparseCleanup(h->symb);
fail:
    if (h) {
        free(h->col_starts);
        free(h->row_idx);
        free(h->values);
        free(h);
    }
    return NULL;
}

/* Solve M·x = b for the cached factorisation. */
int accel_ldlt_solve(void *handle, const double *b, double *x) {
    if (!handle) return -1;
    AccelLDLT *h = (AccelLDLT *)handle;
    DenseVector_Double rhs;
    rhs.count = h->n;
    rhs.data  = (double *)b;
    DenseVector_Double sol;
    sol.count = h->n;
    sol.data  = x;
    SparseSolve(h->fact, rhs, sol);
    return 0;
}

void accel_ldlt_destroy(void *handle) {
    if (!handle) return;
    AccelLDLT *h = (AccelLDLT *)handle;
    SparseCleanup(h->fact);
    SparseCleanup(h->symb);
    free(h->col_starts);
    free(h->row_idx);
    free(h->values);
    free(h);
}
