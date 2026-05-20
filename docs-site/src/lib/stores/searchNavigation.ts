// Coordination store for "jump to a symbol" navigation.
// Search results, the API table-of-contents, and in-page links all set a
// target here; ClassDoc / FunctionDoc / ModuleDoc watch it to expand and
// scroll themselves into view.

import { writable } from 'svelte/store';

export type SymbolType = 'module' | 'class' | 'function' | 'method';

export interface SearchTarget {
	name: string;
	type: SymbolType;
	/** Owning class name — set for methods. */
	parentClass?: string;
	/** Where the navigation originated. */
	source: 'search' | 'toc';
}

export const searchTarget = writable<SearchTarget | null>(null);

export function clearSearchTarget() {
	searchTarget.set(null);
}
