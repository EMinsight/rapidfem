import { error } from '@sveltejs/kit';
import { base } from '$app/paths';
import versionsData from '$lib/api/versions.json';
import type { APIPackage, VersionManifest } from '$lib/api/types';

export const prerender = true;

const manifest = versionsData as VersionManifest;

// Enumerate every version (plus the 'latest' alias) so the static adapter
// prerenders one API page per version.
export function entries() {
	const tags = manifest.versions.map((v) => v.tag);
	return ['latest', ...tags].map((version) => ({ version }));
}

export async function load({ params, fetch }) {
	const res = await fetch(`${base}/api/${params.version}.json`);
	if (!res.ok) {
		error(404, `No API data for version "${params.version}"`);
	}
	const api = (await res.json()) as APIPackage;
	return { api, manifest, version: params.version };
}
