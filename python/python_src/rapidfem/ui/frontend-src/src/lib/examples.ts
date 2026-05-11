/** Result-metric tags used by ResultsPanel to decide which derived numbers
 *  to compute alongside the S-parameter plots. The frontend lets the user
 *  pick any subset; defaults to none. */
export type Metric =
	| 'L_eq'    // π-equivalent series inductance — for inductors
	| 'Q'       // quality factor at port 1 — for resonators
	| 'Z0';     // characteristic impedance — for transmission lines
