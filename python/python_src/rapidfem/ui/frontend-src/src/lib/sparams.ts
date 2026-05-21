/** 2x2 complex matrix utilities + S→Z→Y conversion + L_eq extraction. */

import type { SMatrix } from './api';

export type C = { re: number; im: number };

const cmul = (a: C, b: C): C => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
const cadd = (a: C, b: C): C => ({ re: a.re + b.re, im: a.im + b.im });
const csub = (a: C, b: C): C => ({ re: a.re - b.re, im: a.im - b.im });
const cinv = (z: C): C => {
	const m2 = z.re * z.re + z.im * z.im;
	return { re: z.re / m2, im: -z.im / m2 };
};
const cneg = (z: C): C => ({ re: -z.re, im: -z.im });

export function inv2x2(m: C[][]): C[][] {
	const a = m[0][0], b = m[0][1], c = m[1][0], d = m[1][1];
	const det = csub(cmul(a, d), cmul(b, c));
	const Di = cinv(det);
	return [
		[cmul(d, Di), cmul(cneg(b), Di)],
		[cmul(cneg(c), Di), cmul(a, Di)]
	];
}

const matmul2 = (A: C[][], B: C[][]): C[][] => [
	[cadd(cmul(A[0][0], B[0][0]), cmul(A[0][1], B[1][0])),
	 cadd(cmul(A[0][0], B[0][1]), cmul(A[0][1], B[1][1]))],
	[cadd(cmul(A[1][0], B[0][0]), cmul(A[1][1], B[1][0])),
	 cadd(cmul(A[1][0], B[0][1]), cmul(A[1][1], B[1][1]))]
];

/** Z = sqrt(z0) (I + S) (I - S)^-1 sqrt(z0). */
export function sToZ(S: SMatrix, z0: number): C[][] {
	const I: C[][] = [[{ re: 1, im: 0 }, { re: 0, im: 0 }],
	                  [{ re: 0, im: 0 }, { re: 1, im: 0 }]];
	const IpS = [[cadd(I[0][0], S[0][0]), cadd(I[0][1], S[0][1])],
	             [cadd(I[1][0], S[1][0]), cadd(I[1][1], S[1][1])]];
	const ImS = [[csub(I[0][0], S[0][0]), csub(I[0][1], S[0][1])],
	             [csub(I[1][0], S[1][0]), csub(I[1][1], S[1][1])]];
	const Z = matmul2(IpS, inv2x2(ImS));
	return [
		[{ re: Z[0][0].re * z0, im: Z[0][0].im * z0 },
		 { re: Z[0][1].re * z0, im: Z[0][1].im * z0 }],
		[{ re: Z[1][0].re * z0, im: Z[1][0].im * z0 },
		 { re: Z[1][1].re * z0, im: Z[1][1].im * z0 }]
	];
}

export function sToY(S: SMatrix, z0: number): C[][] {
	return inv2x2(sToZ(S, z0));
}

/** Equivalent series-L from π-equivalent: L = 1 / (ω · Im(Y21)).
 *  Sign flip of L_eq marks the inductor's self-resonance frequency. */
export function L_eq_pH(S: SMatrix, freq_hz: number, z0 = 50): number {
	const omega = 2 * Math.PI * freq_hz;
	const Y = sToY(S, z0);
	return 1e12 / (omega * Y[1][0].im);
}

/** Find SRF by linear interpolation on Im(Y21) zero-crossing.
 *  Returns null if the sweep doesn't cross. */
export function find_srf(
	freqs_hz: number[],
	smats: SMatrix[],
	z0 = 50
): number | null {
	if (smats.length < 2) return null;
	const imY21 = smats.map((S) => sToY(S, z0)[1][0].im);
	const sign0 = Math.sign(imY21[0]);
	for (let k = 1; k < imY21.length; k++) {
		if (Math.sign(imY21[k]) !== sign0) {
			const f1 = freqs_hz[k - 1], f2 = freqs_hz[k];
			const y1 = imY21[k - 1], y2 = imY21[k];
			return f1 + (f2 - f1) * y1 / (y1 - y2);
		}
	}
	return null;
}

/** Characteristic impedance of a uniform 2-port transmission line.
 *  From Z-params: Z0 = sqrt(Z11^2 - Z12^2) (Pozar §4.4 / ABCD relation
 *  B/C = Z0^2 for a symmetric reciprocal line; equivalently
 *  Z0^2 = Z11·Z22 - Z12·Z21, which simplifies for uniform symmetric TLs).
 *  Returns the real part of the principal square root in ohms. */
export function Z0_TL(S: SMatrix, z0_ref = 50): number {
	const Z = sToZ(S, z0_ref);
	const z11_sq = cmul_local(Z[0][0], Z[0][0]);
	const z12_sq = cmul_local(Z[0][1], Z[1][0]);
	const inside: C = { re: z11_sq.re - z12_sq.re, im: z11_sq.im - z12_sq.im };
	return Math.sqrt(Math.max(0, inside.re));
}

function cmul_local(a: C, b: C): C {
	return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}
