// API data types — must match the JSON emitted by scripts/build.py.

export interface APIParameter {
	name: string;
	type: string | null;
	default: string | null;
	description: string;
}

export interface APIAttribute {
	name: string;
	description: string;
	type: string | null;
	value: string | null;
}

export interface APIMethod {
	name: string;
	description: string;
	docstring_html: string;
	source: string | null;
	signature: string | null;
	parameters: APIParameter[];
	returns: string | null;
	method_type: 'method' | 'classmethod' | 'staticmethod' | 'property';
}

export interface APIClass {
	name: string;
	description: string;
	docstring_html: string;
	source: string | null;
	bases: string[];
	methods: APIMethod[];
	attributes: APIAttribute[];
	parameters: APIParameter[];
}

export interface APIFunction {
	name: string;
	description: string;
	docstring_html: string;
	source: string | null;
	signature: string | null;
	parameters: APIParameter[];
	returns: string | null;
}

export interface APIModule {
	name: string;
	description: string;
	docstring_html: string;
	classes: APIClass[];
	functions: APIFunction[];
}

export interface APIPackage {
	package: string;
	display_name: string;
	version: string;
	modules: Record<string, APIModule>;
}

// versions.json manifest
export interface VersionEntry {
	tag: string;
	date: string;
}

export interface VersionManifest {
	latest: string;
	versions: VersionEntry[];
}
