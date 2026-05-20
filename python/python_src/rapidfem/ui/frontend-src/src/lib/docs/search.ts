// Fuzzy search over the extracted API of the currently loaded version.

import Fuse from 'fuse.js';
import type { APIPackage } from '$lib/docs/api/types';
import type { SymbolType } from '$lib/docs/stores/searchNavigation';

export interface SearchItem {
	name: string;
	type: SymbolType;
	/** Owning module path. */
	module: string;
	/** Owning class name — set for methods. */
	parentClass?: string;
	description: string;
	signature?: string;
}

/** Flatten an API package into a searchable list of symbols. */
export function buildSearchItems(pkg: APIPackage): SearchItem[] {
	const items: SearchItem[] = [];

	for (const module of Object.values(pkg.modules)) {
		items.push({
			name: module.name,
			type: 'module',
			module: module.name,
			description: module.description
		});

		for (const cls of module.classes) {
			items.push({
				name: cls.name,
				type: 'class',
				module: module.name,
				description: cls.description
			});

			for (const method of cls.methods) {
				if (method.name === '__init__') continue;
				items.push({
					name: method.name,
					type: 'method',
					module: module.name,
					parentClass: cls.name,
					description: method.description,
					signature: method.signature ?? undefined
				});
			}
		}

		for (const func of module.functions) {
			items.push({
				name: func.name,
				type: 'function',
				module: module.name,
				description: func.description,
				signature: func.signature ?? undefined
			});
		}
	}

	return items;
}

/** Create a Fuse index. Name matches are weighted far above descriptions. */
export function createSearch(items: SearchItem[]): Fuse<SearchItem> {
	return new Fuse(items, {
		keys: [
			{ name: 'name', weight: 0.8 },
			{ name: 'description', weight: 0.2 }
		],
		threshold: 0.4,
		ignoreLocation: true,
		minMatchCharLength: 2
	});
}

export function runSearch(fuse: Fuse<SearchItem>, query: string, limit = 20): SearchItem[] {
	const q = query.trim();
	if (q.length < 2) return [];
	return fuse.search(q, { limit }).map((r) => r.item);
}
