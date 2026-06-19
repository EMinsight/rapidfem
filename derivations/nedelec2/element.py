# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
#
# Clean-room symbolic derivation of the 2nd-order H(curl) tetrahedral element.
# Independent of any third-party code: the 20-DOF basis is constructed here
# from its textbook definition (Whitney edge function times nodal barycentric
# weights), per Savage & Peterson, "Higher-order vector finite elements for
# tetrahedral cells", IEEE Trans. MTT 44 (1996) 874-879; see also Jin, "The
# Finite Element Method in Electromagnetics".
"""Symbolic Nedelec-2 (20-DOF) element stiffness and mass matrices.

The 20 basis functions on a tet, with W_ab = L_a grad L_b - L_b grad L_a and
the Whitney function scaled by an edge length / nodal barycentric weight:

  edge e=(a,b), length l_e:
      phi_e1 = l_e * L_a * W_ab
      phi_e2 = l_e * L_b * W_ab
  face f=(n0,n1,n2):
      phi_f1 = dist(n0,n2) * L_n1 * (L_n0 grad L_n2 - L_n2 grad L_n0)
      phi_f2 = dist(n0,n1) * L_n2 * (L_n0 grad L_n1 - L_n1 grad L_n0)

DOF order matches the assembler: [6 edge m1][4 face m1][6 edge m2][4 face m2].

A vector field is stored as a list of terms (coeff, exps, vec):
    field = sum_t  coeff_t * (L1^e1 L2^e2 L3^e3 L4^e4) * vec_t
with grad L_i constant vectors. This makes curl and the volume integrals
exact polynomial operations in the barycentric coordinates, integrated via
the simplex identity from barycentric.py.
"""
from __future__ import annotations

import sympy as sp

from barycentric import volume_coeff_closed  # exact integral / (6V), pure rational

# Edge / face local-node layout used by the assembler's unit-tet test.
LOCAL_EDGE_MAP = [(0, 1), (0, 2), (0, 3), (1, 2), (3, 1), (2, 3)]
LOCAL_TRI_MAP = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 2, 3)]


def barycentric_gradients(verts):
    """Return (sixV, grads) where grads[i] is the exact constant vector grad L_i.

    L_i = a_i + b_i x + c_i y + d_i z solves M [a;b;c;d] = e_i with
    M rows [1, x_i, y_i, z_i]. grad L_i = (b_i, c_i, d_i). 6V = det(M-ish).
    """
    M = sp.Matrix(
        [[sp.Integer(1), v[0], v[1], v[2]] for v in verts]
    )  # 4x4, row i = vertex i
    Minv = M.inv()
    # column j of Minv gives coefficients (a_j,b_j,c_j,d_j) of L_j? Solve M*C=I.
    # L_j(x) = sum_k C[k,j] * [1,x,y,z]_k  => grad = rows 1..3 of column j.
    grads = []
    for j in range(4):
        col = Minv[:, j]
        grads.append(sp.Matrix([col[1], col[2], col[3]]))
    sixV = abs(M.det())
    return sixV, grads


def dist(verts, i, j):
    d = sp.Matrix(verts[i]) - sp.Matrix(verts[j])
    return sp.sqrt((d.T * d)[0])


def monomial_from_nodes(nodes):
    """Return exponent tuple over (L1,L2,L3,L4) for a product of L_node's."""
    e = [0, 0, 0, 0]
    for n in nodes:
        e[n] += 1
    return tuple(e)


def whitney(grads, a, b):
    """W_ab = L_a grad L_b - L_b grad L_a, as field terms (coeff, exps, vec)."""
    ea = monomial_from_nodes([a])
    eb = monomial_from_nodes([b])
    return [
        (sp.Integer(1), eb, grads[a]),    # L_b * grad L_a  -> wait: L_a grad L_b
    ]


def edge_function(grads, a, b, length):
    """phi = length * L_w * W_ab, with W_ab = L_a gradL_b - L_b gradL_a.

    L_w is the nodal weight (a for mode 1, b for mode 2) supplied by caller via
    `weight_node`. Returns list of (coeff, exps, vec).
    """
    raise NotImplementedError  # replaced by weighted_whitney below


def weighted_whitney(grads, weight_node, a, b, scale):
    """scale * L_weight * (L_a gradL_b - L_b gradL_a)  ->  field terms."""
    terms = []
    # term1: + L_weight * L_a * gradL_b
    e1 = monomial_from_nodes([weight_node, a])
    terms.append((scale, e1, grads[b]))
    # term2: - L_weight * L_b * gradL_a
    e2 = monomial_from_nodes([weight_node, b])
    terms.append((-scale, e2, grads[a]))
    return terms


def build_basis(verts):
    """Return list of 20 fields, each a list of (coeff, exps, vec)."""
    sixV, grads = barycentric_gradients(verts)
    edge_m1, edge_m2, face_m1, face_m2 = [], [], [], []
    for (a, b) in LOCAL_EDGE_MAP:
        le = dist(verts, a, b)
        edge_m1.append(weighted_whitney(grads, a, a, b, le))  # l*L_a*W_ab
        edge_m2.append(weighted_whitney(grads, b, a, b, le))  # l*L_b*W_ab
    for (n0, n1, n2) in LOCAL_TRI_MAP:
        l_f1 = dist(verts, n0, n2)
        l_f2 = dist(verts, n0, n1)
        # phi_f1 = l*L_n1*(L_n0 gradL_n2 - L_n2 gradL_n0)
        face_m1.append(weighted_whitney(grads, n1, n0, n2, l_f1))
        # phi_f2 = l*L_n2*(L_n0 gradL_n1 - L_n1 gradL_n0)
        face_m2.append(weighted_whitney(grads, n2, n0, n1, l_f2))
    basis = edge_m1 + face_m1 + edge_m2 + face_m2
    assert len(basis) == 20
    return basis, sixV, grads


def curl_field(field, grads):
    """curl(sum coeff * L^e * vec) = sum coeff * (grad(L^e) x vec).

    grad(L^e) = sum_k e_k L^(e-1_k) grad L_k. Returns field terms (coeff,exps,vec).
    """
    out = []
    for coeff, exps, vec in field:
        for k in range(4):
            if exps[k] == 0:
                continue
            new_e = list(exps)
            new_e[k] -= 1
            cross = grads[k].cross(vec)  # grad L_k x vec
            out.append((coeff * exps[k], tuple(new_e), cross))
    return out


def integrate_dot(field_a, field_b, tensor, sixV):
    """integral_tet (field_a . tensor . field_b) dV, exact.

    Each pair of terms contributes coeff_a*coeff_b*(vec_a^T tensor vec_b) times
    integral L^(ea+eb) dV = volume_coeff(ea+eb) * 6V.
    """
    total = sp.Integer(0)
    for ca, ea, va in field_a:
        for cb, eb, vb in field_b:
            quad = (va.T * tensor * vb)[0]
            if quad == 0:
                continue
            exps = tuple(ea[k] + eb[k] for k in range(4))
            vcoeff = _volume_coeff_exps(exps)
            total += ca * cb * quad * vcoeff * sixV
    return sp.simplify(total)


def _volume_coeff_exps(exps):
    """volume integral identity for given (L1..L4) exponents, as /(6V) rational."""
    # build index args (a,b,c,d style) from exponents; reuse closed form directly
    num = 1
    for e in exps:
        num *= sp.factorial(e)
    total = sum(exps)
    return sp.Rational(num, 1) / sp.factorial(total + 3)


def element_matrices(verts, eps, mu_inv):
    """Return (D, F) 20x20 sympy matrices: stiffness (curl-curl) and mass."""
    basis, sixV, grads = build_basis(verts)
    curls = [curl_field(f, grads) for f in basis]
    D = sp.zeros(20, 20)
    F = sp.zeros(20, 20)
    for i in range(20):
        for j in range(i, 20):
            dij = integrate_dot(curls[i], curls[j], mu_inv, sixV)
            fij = integrate_dot(basis[i], basis[j], eps, sixV)
            D[i, j] = D[j, i] = dij
            F[i, j] = F[j, i] = fij
    return D, F


def frob(M):
    s = sp.Integer(0)
    for i in range(M.rows):
        for j in range(M.cols):
            s += M[i, j] ** 2
    return sp.sqrt(s)


if __name__ == "__main__":
    # Unit tet, identity tensors -> compare to the assembler golden norms.
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    I3 = sp.eye(3)
    D, F = element_matrices(verts, I3, I3)
    dnorm = sp.nsimplify(frob(D))
    fnorm = sp.nsimplify(frob(F))
    d_num = float(dnorm)
    f_num = float(fnorm)
    print(f"||D||_F = {d_num:.15e}")
    print(f"||F||_F = {f_num:.15e}")
    d_ref = 1.931105037709411e+00
    f_ref = 6.434570001645180e-02
    print(f"d_ref   = {d_ref:.15e}  rel err {abs(d_num-d_ref)/d_ref:.2e}")
    print(f"f_ref   = {f_ref:.15e}  rel err {abs(f_num-f_ref)/f_ref:.2e}")

    # Localize the mass discrepancy by DOF-block Frobenius norms.
    # order: [0:6]=edge m1, [6:10]=face m1, [10:16]=edge m2, [16:20]=face m2
    edge_idx = list(range(0, 6)) + list(range(10, 16))
    face_idx = list(range(6, 10)) + list(range(16, 20))

    def block_norm(M, rows, cols):
        s = sp.Integer(0)
        for i in rows:
            for j in cols:
                s += M[i, j] ** 2
        return float(sp.sqrt(s))

    print("--- mass F block norms (sympy) ---")
    print(f"edge-edge: {block_norm(F, edge_idx, edge_idx):.15e}")
    print(f"edge-face: {block_norm(F, edge_idx, face_idx):.15e}")
    print(f"face-face: {block_norm(F, face_idx, face_idx):.15e}")
    print("--- stiffness D block norms (sympy) ---")
    print(f"edge-edge: {block_norm(D, edge_idx, edge_idx):.15e}")
    print(f"edge-face: {block_norm(D, edge_idx, face_idx):.15e}")
    print(f"face-face: {block_norm(D, face_idx, face_idx):.15e}")
