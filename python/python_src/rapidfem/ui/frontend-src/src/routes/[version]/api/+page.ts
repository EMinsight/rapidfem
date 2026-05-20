import { error } from '@sveltejs/kit';
import { base } from '$app/paths';
import versionsData from '$lib/docs/api/versions.json';
import type { APIPackage, VersionManifest } from '$lib/docs/api/types';

// SPA route — SSR/prerender are off app-wide; this load runs in the browser.
const manifest = versionsData as VersionManifest;

export async function load({ params, fetch }) {
	const res = await fetch(`${base}/api/${params.version}.json`);
	if (!res.ok) {
		error(404, `No API data for version "${params.version}"`);
	}
	const api = (await res.json()) as APIPackage;
	return { api, manifest, version: params.version };
}
