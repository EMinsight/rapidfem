(function(){"use strict";const B={bg:"#131316"};function I(){return{theta:-Math.PI/4,phi:Math.PI/5,distance:300,target:[0,0,0]}}const V=`#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in float aScalar;
uniform mat4 uMVP;
uniform mat3 uNormalMat;
uniform float uZFlip;
out vec3 vNormal;
out float vScalar;
out vec3 vWorld;
void main() {
	vec3 n = aNormal;
	n.z *= uZFlip;
	vNormal = normalize(uNormalMat * n);
	vec3 pos = aPos;
	pos.z *= uZFlip;
	vWorld = pos;
	gl_Position = uMVP * vec4(pos, 1.0);
	vScalar = aScalar;
}`,D=`#version 300 es
precision highp float;
in vec3 vNormal;
in float vScalar;
in vec3 vWorld;
uniform vec3 uColor;
uniform vec3 uLightDir;
uniform float uAmbient;
uniform float uColormap;
uniform vec4 uClipPlane;          // (nx, ny, nz, d): discard fragments where dot(world, n) > d
uniform float uClipEnable;
out vec4 fragColor;

// Polynomial Inferno colormap — black → purple → red → orange → yellow → white.
// Matches our warm rapidpassives palette better than viridis.
vec3 inferno(float t) {
	t = clamp(t, 0.0, 1.0);
	const vec3 c0 = vec3(0.0002, 0.0016, -0.0194);
	const vec3 c1 = vec3(0.1065, 0.5639, 3.9327);
	const vec3 c2 = vec3(11.6024, -3.972, -15.9423);
	const vec3 c3 = vec3(-41.7039, 17.4363, 44.354);
	const vec3 c4 = vec3(77.1629, -33.4023, -81.8073);
	const vec3 c5 = vec3(-71.319, 32.6261, 73.2095);
	const vec3 c6 = vec3(25.1311, -12.2426, -23.0703);
	return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

void main() {
	if (uClipEnable > 0.5) {
		if (dot(vWorld, uClipPlane.xyz) > uClipPlane.w) discard;
	}
	float diff = max(dot(normalize(vNormal), uLightDir), 0.0);
	vec3 base = mix(uColor, inferno(vScalar), uColormap);
	vec3 lit = base * (uAmbient + (1.0 - uAmbient) * diff);
	fragColor = vec4(lit, 1.0);
}`,U=`#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }`,N=`#version 300 es
precision highp float;
uniform vec3 uColor;
out vec4 fragColor;
void main() { fragColor = vec4(uColor, 1.0); }`,z=`#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aABC;           // (A, B, C) phasor terms
uniform mat4 uMVP;
uniform float uZFlip;
uniform float uPointScale;                 // base size in pixels at unit clip-w
uniform float uPhase;                      // current ωt in radians
uniform float uLogFloor;                   // log10(min |E|) (log mode) or |E|min (lin mode)
uniform float uLogRange;                   // log10(max/min) (log mode) or (|E|max - |E|min) (lin mode)
uniform float uLogScale;                   // 1.0 = log color mapping, 0.0 = linear
out float vScalar;
void main() {
	vec3 pos = aPos;
	pos.z *= uZFlip;
	gl_Position = uMVP * vec4(pos, 1.0);

	// |E(t)|² = A cos²(ωt) + B sin²(ωt) − 2 C cos·sin
	float c = cos(uPhase);
	float s = sin(uPhase);
	float e2 = aABC.x * c * c + aABC.y * s * s - 2.0 * aABC.z * c * s;
	float mag = sqrt(max(e2, 0.0));
	float norm_log = (log(max(mag, 1e-30)) / 2.302585093 - uLogFloor) / max(uLogRange, 1e-9);
	float norm_lin = (mag - uLogFloor) / max(uLogRange, 1e-9);
	float norm = mix(norm_lin, norm_log, uLogScale);
	vScalar = clamp(norm, 0.0, 1.0);

	float w = max(gl_Position.w, 1e-6);
	gl_PointSize = clamp(uPointScale / w * (0.4 + 0.6 * vScalar), 4.0, 96.0);
}`,O=`#version 300 es
precision highp float;
in float vScalar;
out vec4 fragColor;
vec3 inferno(float t) {
	t = clamp(t, 0.0, 1.0);
	const vec3 c0 = vec3(0.0002, 0.0016, -0.0194);
	const vec3 c1 = vec3(0.1065, 0.5639, 3.9327);
	const vec3 c2 = vec3(11.6024, -3.972, -15.9423);
	const vec3 c3 = vec3(-41.7039, 17.4363, 44.354);
	const vec3 c4 = vec3(77.1629, -33.4023, -81.8073);
	const vec3 c5 = vec3(-71.319, 32.6261, 73.2095);
	const vec3 c6 = vec3(25.1311, -12.2426, -23.0703);
	return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}
void main() {
	vec2 uv = gl_PointCoord * 2.0 - 1.0;
	float r2 = dot(uv, uv);
	if (r2 > 1.0) discard;
	float falloff = pow(1.0 - r2, 2.0);
	vec3 col = inferno(vScalar);
	// Additive contribution: low-field points fade out, high-field accumulate brightness
	fragColor = vec4(col * (vScalar * falloff), 1.0);
}`;function S(t,e,i){const o=t.createShader(e);if(t.shaderSource(o,i),t.compileShader(o),!t.getShaderParameter(o,t.COMPILE_STATUS)){const r=t.getShaderInfoLog(o);throw t.deleteShader(o),new Error("Shader compile: "+r)}return o}function E(t,e,i){const o=S(t,t.VERTEX_SHADER,e),r=S(t,t.FRAGMENT_SHADER,i),n=t.createProgram();if(t.attachShader(n,o),t.attachShader(n,r),t.linkProgram(n),!t.getProgramParameter(n,t.LINK_STATUS))throw new Error("Program link: "+t.getProgramInfoLog(n));return t.deleteShader(o),t.deleteShader(r),n}function k(t){const e=parseInt(t.slice(1,3),16)/255,i=parseInt(t.slice(3,5),16)/255,o=parseInt(t.slice(5,7),16)/255;return[e,i,o]}function Y(t,e,i,o){const r=1/Math.tan(t/2),n=1/(i-o),a=new Float32Array(16);return a[0]=r/e,a[5]=r,a[10]=(o+i)*n,a[11]=-1,a[14]=2*o*i*n,a}function q(t,e,i){const o=t[0]-e[0],r=t[1]-e[1],n=t[2]-e[2];let a=Math.sqrt(o*o+r*r+n*n);const s=o/a,l=r/a,c=n/a,h=i[1]*c-i[2]*l,u=i[2]*s-i[0]*c,f=i[0]*l-i[1]*s;a=Math.sqrt(h*h+u*u+f*f);const v=h/a,p=u/a,g=f/a,b=l*g-c*p,A=c*v-s*g,x=s*p-l*v,d=new Float32Array(16);return d[0]=v,d[1]=b,d[2]=s,d[4]=p,d[5]=A,d[6]=l,d[8]=g,d[9]=x,d[10]=c,d[12]=-(v*t[0]+p*t[1]+g*t[2]),d[13]=-(b*t[0]+A*t[1]+x*t[2]),d[14]=-(s*t[0]+l*t[1]+c*t[2]),d[15]=1,d}function W(t,e){const i=new Float32Array(16);for(let o=0;o<4;o++)for(let r=0;r<4;r++)i[r*4+o]=t[o]*e[r*4]+t[4+o]*e[r*4+1]+t[8+o]*e[r*4+2]+t[12+o]*e[r*4+3];return i}function H(t){const e=new Float32Array(9);return e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[4],e[4]=t[5],e[5]=t[6],e[6]=t[8],e[7]=t[9],e[8]=t[10],e}function Z(t){const e=Math.cos(t.phi);return[t.target[0]+t.distance*e*Math.sin(t.theta),t.target[1]+t.distance*e*Math.cos(t.theta),t.target[2]+t.distance*Math.sin(t.phi)]}function G(t){const e=t.getContext("webgl2",{antialias:!0,alpha:!0,preserveDrawingBuffer:!0});if(!e)return null;const i=E(e,V,D),o=E(e,U,N),r=E(e,z,O),n=k(B.bg);return e.clearColor(n[0],n[1],n[2],1),e.enable(e.DEPTH_TEST),e.disable(e.CULL_FACE),{gl:e,program:i,uMVP:e.getUniformLocation(i,"uMVP"),uNormalMat:e.getUniformLocation(i,"uNormalMat"),uColor:e.getUniformLocation(i,"uColor"),uLightDir:e.getUniformLocation(i,"uLightDir"),uAmbient:e.getUniformLocation(i,"uAmbient"),uZFlip:e.getUniformLocation(i,"uZFlip"),uColormap:e.getUniformLocation(i,"uColormap"),uClipPlane:e.getUniformLocation(i,"uClipPlane"),uClipEnable:e.getUniformLocation(i,"uClipEnable"),clip_plane:[0,0,1,0],clip_enable:!1,lineProgram:o,uLineMVP:e.getUniformLocation(o,"uMVP"),uLineColor:e.getUniformLocation(o,"uColor"),pointProgram:r,uPointMVP:e.getUniformLocation(r,"uMVP"),uPointZFlip:e.getUniformLocation(r,"uZFlip"),uPointScale:e.getUniformLocation(r,"uPointScale"),uPointPhase:e.getUniformLocation(r,"uPhase"),uPointLogFloor:e.getUniformLocation(r,"uLogFloor"),uPointLogRange:e.getUniformLocation(r,"uLogRange"),uPointLogScale:e.getUniformLocation(r,"uLogScale"),meshes:[],lineMeshes:[],pointCloud:null,pointPhase:0,pointLogFloor:-30,pointLogRange:6,pointLogScale:1,bbox:{min:[0,0,0],max:[0,0,0]}}}function $(t){const{gl:e}=t;for(const i of t.meshes){e.deleteVertexArray(i.vao);for(const o of i.buffers)e.deleteBuffer(o)}for(const i of t.lineMeshes){e.deleteVertexArray(i.vao);for(const o of i.buffers)e.deleteBuffer(o)}e.deleteProgram(t.program),e.deleteProgram(t.lineProgram)}function j(t){const{gl:e}=t;for(const i of t.meshes){e.deleteVertexArray(i.vao);for(const o of i.buffers)e.deleteBuffer(o)}t.meshes=[];for(const i of t.lineMeshes){e.deleteVertexArray(i.vao);for(const o of i.buffers)e.deleteBuffer(o)}t.lineMeshes=[]}function M(t,e,i){const{gl:o}=t;if(t.pointCloud){o.deleteVertexArray(t.pointCloud.vao);for(const s of t.pointCloud.buffers)o.deleteBuffer(s)}if(e.length===0){t.pointCloud=null;return}const r=o.createVertexArray();o.bindVertexArray(r);const n=o.createBuffer();o.bindBuffer(o.ARRAY_BUFFER,n),o.bufferData(o.ARRAY_BUFFER,e,o.STATIC_DRAW),o.enableVertexAttribArray(0),o.vertexAttribPointer(0,3,o.FLOAT,!1,0,0);const a=o.createBuffer();o.bindBuffer(o.ARRAY_BUFFER,a),o.bufferData(o.ARRAY_BUFFER,i,o.STATIC_DRAW),o.enableVertexAttribArray(1),o.vertexAttribPointer(1,3,o.FLOAT,!1,0,0),o.bindVertexArray(null),t.pointCloud={vao:r,buffers:[n,a],count:e.length/3}}function X(t,e,i){t.pointLogFloor=e,t.pointLogRange=i}function K(t,e){t.pointLogScale=e==="log"?1:0}function J(t,e,i){t.pointLogFloor=e,t.pointLogRange=i}function Q(t,e,i,o,r=0,n,a){const{gl:s}=t,l=s.createVertexArray();s.bindVertexArray(l);const c=s.createBuffer();s.bindBuffer(s.ARRAY_BUFFER,c),s.bufferData(s.ARRAY_BUFFER,e,s.STATIC_DRAW),s.enableVertexAttribArray(0),s.vertexAttribPointer(0,3,s.FLOAT,!1,0,0);const h=s.createBuffer();s.bindBuffer(s.ARRAY_BUFFER,h),s.bufferData(s.ARRAY_BUFFER,i,s.STATIC_DRAW),s.enableVertexAttribArray(1),s.vertexAttribPointer(1,3,s.FLOAT,!1,0,0);const u=[c,h];let f=!1;s.disableVertexAttribArray(2),s.vertexAttrib1f(2,0),s.bindVertexArray(null),t.meshes.push({vao:l,buffers:u,count:e.length/3,color:o,tag:r,visible:!0,depth_offset:n,has_scalar:f})}function tt(t,e,i,o=0){const{gl:r}=t,n=r.createVertexArray();r.bindVertexArray(n);const a=r.createBuffer();r.bindBuffer(r.ARRAY_BUFFER,a),r.bufferData(r.ARRAY_BUFFER,e,r.STATIC_DRAW),r.enableVertexAttribArray(0),r.vertexAttribPointer(0,3,r.FLOAT,!1,0,0),r.bindVertexArray(null),t.lineMeshes.push({vao:n,buffers:[a],count:e.length/3,color:i,tag:o,visible:!0})}function C(t,e,i){for(const o of t.meshes)o.tag===e&&(o.visible=i);for(const o of t.lineMeshes)o.tag===e&&(o.visible=i)}function et(t,e,i){t.bbox.min=e,t.bbox.max=i}function ot(t,e,i,o,r=1){const{gl:n}=t;n.viewport(0,0,i,o),n.clear(n.COLOR_BUFFER_BIT|n.DEPTH_BUFFER_BIT);const a=i/o||1,s=t.bbox.max[0]-t.bbox.min[0],l=t.bbox.max[1]-t.bbox.min[1],c=t.bbox.max[2]-t.bbox.min[2],h=.5*Math.sqrt(s*s+l*l+c*c),u=Math.max(e.distance*.001,h*.001,1e-9),f=(e.distance+h)*8,v=Y(Math.PI/6,a,u,f),p=Z(e),g=q(p,e.target,[0,0,1]),b=W(v,g),A=H(g),x=p[0]-e.target[0],d=p[1]-e.target[1],F=p[2]-e.target[2]+e.distance*.3,y=Math.sqrt(x*x+d*d+F*F),w=[x/y,d/y,F/y];n.useProgram(t.program),n.uniformMatrix4fv(t.uMVP,!1,b),n.uniformMatrix3fv(t.uNormalMat,!1,A),n.uniform3f(t.uLightDir,w[0],w[1],w[2]),n.uniform1f(t.uAmbient,.8),n.uniform1f(t.uZFlip,r),n.uniform4f(t.uClipPlane,t.clip_plane[0],t.clip_plane[1],t.clip_plane[2],t.clip_plane[3]),n.uniform1f(t.uClipEnable,t.clip_enable?1:0);let P=!1;for(const m of t.meshes)m.visible&&(m.depth_offset?(P||(n.enable(n.POLYGON_OFFSET_FILL),P=!0),n.polygonOffset(m.depth_offset[0],m.depth_offset[1])):P&&(n.disable(n.POLYGON_OFFSET_FILL),P=!1),n.uniform3f(t.uColor,m.color[0],m.color[1],m.color[2]),n.uniform1f(t.uColormap,m.has_scalar?1:0),n.bindVertexArray(m.vao),n.drawArrays(n.TRIANGLES,0,m.count));if(P&&n.disable(n.POLYGON_OFFSET_FILL),t.lineMeshes.length>0){n.useProgram(t.lineProgram),n.uniformMatrix4fv(t.uLineMVP,!1,b);for(const m of t.lineMeshes)m.visible&&(n.uniform3f(t.uLineColor,m.color[0],m.color[1],m.color[2]),n.bindVertexArray(m.vao),n.drawArrays(n.LINES,0,m.count))}if(t.pointCloud&&t.pointCloud.count>0){n.useProgram(t.pointProgram),n.uniformMatrix4fv(t.uPointMVP,!1,b),n.uniform1f(t.uPointZFlip,r);const m=t.bbox.max[0]-t.bbox.min[0],L=t.bbox.max[1]-t.bbox.min[1],gt=Math.max(m,L,1e-9);n.uniform1f(t.uPointScale,gt*.4),n.uniform1f(t.uPointPhase,t.pointPhase),n.uniform1f(t.uPointLogFloor,t.pointLogFloor),n.uniform1f(t.uPointLogRange,t.pointLogRange),n.uniform1f(t.uPointLogScale,t.pointLogScale),n.disable(n.DEPTH_TEST),n.depthMask(!1),n.enable(n.BLEND),n.blendFunc(n.ONE,n.ONE),n.bindVertexArray(t.pointCloud.vao),n.drawArrays(n.POINTS,0,t.pointCloud.count),n.disable(n.BLEND),n.depthMask(!0),n.enable(n.DEPTH_TEST)}n.bindVertexArray(null)}function nt(t,e){const i=(t[0]+e[0])/2,o=(t[1]+e[1])/2,r=(t[2]+e[2])/2,n=e[0]-t[0],a=e[1]-t[1],s=e[2]-t[2],l=Math.max(Math.sqrt(n*n+a*a+s*s),1e-9),c=Math.PI/12,h=l*.6/Math.tan(c)*1.05;return{theta:Math.PI/4,phi:Math.PI/4,distance:h,target:[i,o,r]}}const it=1380339014,_=1,R=3,T=["geometry","mesh","field"],rt=2.4;function st(t){for(let e=t.cells.length-1;e>=0;e--)for(const i of t.cells[e].display_events)if(i.kind==="mesh"&&i.payload)return i.payload;return null}function at(t){for(let e=t.cells.length-1;e>=0;e--)for(const i of t.cells[e].display_events)if(i.kind==="result"&&i.payload)return i.payload;return null}async function lt(t,e){const i=new URL(t.url,e).href,o=await fetch(i);if(!o.ok)throw new Error(`bin fetch ${o.status}`);const r=await o.arrayBuffer(),n=new DataView(r);if(n.getUint32(0,!0)!==it)throw new Error("field bin: bad magic");const a=n.getUint32(8,!0),s=n.getUint32(12,!0),l=n.getUint32(16,!0),c=20,h=new Uint8Array(r,c,a*s),u=c+(h.byteLength+3&-4),f=new Float32Array(r,u),v=[];let p=0;for(let g=0;g<a;g++){const b=[];for(let A=0;A<s;A++)h[g*s+A]===0?b.push(null):(b.push(Array.from(f.subarray(p,p+l))),p+=l);v.push(b)}return v}function ct(t,e){const i=e.length/3,o=new Float32Array(i*9),r=new Float32Array(i*9),n=[0,0,0],a=[0,0,0],s=[0,0,0];for(let l=0;l<i;l++){const c=e[l*3],h=e[l*3+1],u=e[l*3+2];n[0]=t[c*3],n[1]=t[c*3+1],n[2]=t[c*3+2],a[0]=t[h*3],a[1]=t[h*3+1],a[2]=t[h*3+2],s[0]=t[u*3],s[1]=t[u*3+1],s[2]=t[u*3+2];const f=a[0]-n[0],v=a[1]-n[1],p=a[2]-n[2],g=s[0]-n[0],b=s[1]-n[1],A=s[2]-n[2],x=v*A-p*b,d=p*g-f*A,F=f*b-v*g,y=1/Math.max(Math.hypot(x,d,F),1e-20),w=x*y,P=d*y,m=F*y;o.set(n,l*9),o.set(a,l*9+3),o.set(s,l*9+6);for(let L=0;L<3;L++)r[l*9+L*3]=w,r[l*9+L*3+1]=P,r[l*9+L*3+2]=m}return{positions:o,normals:r}}function ht(t){const e=t.tets,i=t.tet_phys,o=i.length,r=(s,l,c)=>{const h=[s,l,c].sort((u,f)=>u-f);return(BigInt(h[0])*0x100000000n+BigInt(h[1]))*0x100000000n+BigInt(h[2])},n=new Map;for(let s=0;s<o;s++){const l=i[s];if(!l)continue;let c=n.get(l);c||(c=[],n.set(l,c)),c.push(s)}const a=[];for(const[,s]of n.entries()){const l=new Map;for(const c of s){const h=e[c*4],u=e[c*4+1],f=e[c*4+2],v=e[c*4+3],p=[[h,u,f],[h,u,v],[h,f,v],[u,f,v]];for(const g of p){const b=r(g[0],g[1],g[2]),A=l.get(b);A?A.count++:l.set(b,{count:1,tri:g})}}for(const c of l.values())c.count===1&&a.push(c.tri[0],c.tri[1],c.tri[2])}return a}function ft(t,e){const i=new Set,o=[],r=(a,s)=>{const l=a<s?a:s,c=a<s?s:a,h=BigInt(l)<<32n|BigInt(c);i.has(h)||(i.add(h),o.push(t[a*3],t[a*3+1],t[a*3+2],t[s*3],t[s*3+1],t[s*3+2]))},n=e.length/3;for(let a=0;a<n;a++){const s=e[a*3],l=e[a*3+1],c=e[a*3+2];r(s,l),r(l,c),r(c,s)}return Float32Array.from(o)}function ut(t,e){const i=t.nodes,o=t.tets,r=o.length/4,n=new Float32Array(r*3),a=new Float32Array(r*3);for(let s=0;s<r;s++){const l=o[s*4],c=o[s*4+1],h=o[s*4+2],u=o[s*4+3];for(let f=0;f<3;f++)n[s*3+f]=.25*(i[l*3+f]+i[c*3+f]+i[h*3+f]+i[u*3+f]),a[s*3+f]=.25*(e[l*3+f]+e[c*3+f]+e[h*3+f]+e[u*3+f])}return{positions:n,abc:a}}function dt(t,e){let i=0;for(let r=0;r<t.length;r+=3){const n=Math.max(t[r],t[r+1]);n>i&&(i=n)}const o=Math.sqrt(Math.max(i,1e-30));return e==="log"?{floor:Math.log10(o)-4,range:4}:{floor:0,range:o}}class mt extends HTMLElement{canvas=null;wrapper=null;loadingEl=null;labelEl=null;glState=null;camera=I();animId=0;mounted=!1;mesh=null;fields=null;hasField=!1;needsRender=!1;isDragging=!1;isRightDrag=!1;lastMouse={x:0,y:0};currentPhase="geometry";phaseStart=0;static get observedAttributes(){return["src","width","height","rotate","cycle","mode","interactive","transparent","speed","theta","phi","field-mode","field-freq","field-port"]}connectedCallback(){this.mounted=!0;const e=this.attachShadow({mode:"open"}),i=this.hasAttribute("transparent");this.wrapper=document.createElement("div"),this.wrapper.style.cssText=`position:relative;width:${this.getAttribute("width")||"100%"};height:${this.getAttribute("height")||"400px"};background:${i?"transparent":"#131316"};overflow:hidden;border-radius:inherit;`,this.canvas=document.createElement("canvas"),this.canvas.style.cssText=`display:block;width:100%;height:100%;cursor:${this.hasAttribute("interactive")?"grab":"default"};`,this.wrapper.appendChild(this.canvas),this.loadingEl=document.createElement("div"),this.loadingEl.style.cssText="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font:500 11px/1 monospace;color:#55535a;",this.wrapper.appendChild(this.loadingEl),this.labelEl=document.createElement("div"),this.labelEl.style.cssText="position:absolute;top:8px;left:10px;font:500 10px/1 monospace;color:#9a96a0;text-transform:uppercase;letter-spacing:0.5px;pointer-events:none;opacity:0;transition:opacity 0.3s;",this.wrapper.appendChild(this.labelEl);const o=document.createElement("a");if(o.href="https://fem.rapidpassives.org",o.target="_blank",o.rel="noopener",o.textContent="RapidFEM",o.style.cssText="position:absolute;bottom:6px;right:8px;font:500 9px/1 monospace;color:#55535a;text-decoration:none;opacity:0.7;transition:opacity 0.15s;",o.onmouseenter=()=>o.style.opacity="1",o.onmouseleave=()=>o.style.opacity="0.7",this.wrapper.appendChild(o),e.appendChild(this.wrapper),this.glState=G(this.canvas),!this.glState)return;this.hasAttribute("interactive")&&(this.canvas.addEventListener("pointerdown",a=>this.onPointerDown(a)),this.canvas.addEventListener("pointermove",a=>this.onPointerMove(a)),this.canvas.addEventListener("pointerup",()=>this.onPointerUp()),this.canvas.addEventListener("wheel",a=>this.onWheel(a),{passive:!1}),this.canvas.addEventListener("contextmenu",a=>a.preventDefault()),this.canvas.addEventListener("dblclick",()=>this.fitView())),new ResizeObserver(()=>{this.needsRender=!0}).observe(this.wrapper);const n=this.getAttribute("src");n&&this.load(n)}disconnectedCallback(){this.mounted=!1,this.animId++,this.glState&&$(this.glState)}attributeChangedCallback(e,i,o){this.mounted&&(e==="src"&&o?this.load(o):(e==="mode"||e==="field-mode"||e==="field-freq"||e==="field-port")&&(this.applyField(),this.applyPhase(this.resolvePhase()),this.needsRender=!0))}onPointerDown(e){this.isDragging=!0,this.isRightDrag=e.button===2,this.lastMouse={x:e.clientX,y:e.clientY},this.canvas?.setPointerCapture(e.pointerId),this.canvas&&(this.canvas.style.cursor="grabbing")}onPointerMove(e){if(!this.isDragging)return;const i=e.clientX-this.lastMouse.x,o=e.clientY-this.lastMouse.y;if(this.lastMouse={x:e.clientX,y:e.clientY},this.isRightDrag){const r=this.camera.distance*7e-4,n=Math.cos(this.camera.theta),a=Math.sin(this.camera.theta);this.camera={...this.camera,target:[this.camera.target[0]+(i*n-o*a*Math.sin(this.camera.phi))*r,this.camera.target[1]-(i*a+o*n*Math.sin(this.camera.phi))*r,this.camera.target[2]+o*Math.cos(this.camera.phi)*r]}}else this.camera={...this.camera,theta:this.camera.theta+i*.005,phi:Math.max(.05,Math.min(Math.PI/2-.05,this.camera.phi+o*.005))};this.needsRender=!0}onPointerUp(){this.isDragging=!1,this.isRightDrag=!1,this.canvas&&(this.canvas.style.cursor="grab")}onWheel(e){e.preventDefault(),this.camera={...this.camera,distance:this.camera.distance*(e.deltaY>0?1.1:1/1.1)},this.needsRender=!0}fitView(){if(!this.glState||!this.mesh)return;this.camera=nt(this.mesh.bbox.min,this.mesh.bbox.max);const e=parseFloat(this.getAttribute("theta")||"45")*Math.PI/180,i=parseFloat(this.getAttribute("phi")||"45")*Math.PI/180;this.camera={...this.camera,theta:e,phi:i},this.needsRender=!0}async load(e){this.loadingEl&&(this.loadingEl.textContent="Loading…",this.loadingEl.style.display="flex");try{const i=new URL(e,location.href).href,o=await fetch(i);if(!o.ok)throw new Error(`HTTP ${o.status}`);const r=await o.json();if(this.mesh=st(r),!this.mesh)throw new Error("no mesh payload in bake");const n=at(r);n&&n.fields&&n.fields.$bin?(this.loadingEl&&(this.loadingEl.textContent="Decoding field…"),this.fields=await lt(n.fields,i),this.hasField=this.fields.some(a=>a.some(s=>s!==null))):(this.fields=null,this.hasField=!1),this.rebuildScene(),this.fitView(),this.loadingEl&&(this.loadingEl.style.display="none"),this.phaseStart=performance.now()/1e3,this.applyPhase(this.resolvePhase()),this.startAnimation()}catch(i){console.error("[fem-viewer] load failed",i),this.loadingEl&&(this.loadingEl.textContent=`Error: ${i.message}`)}}rebuildScene(){if(!this.glState||!this.mesh)return;j(this.glState),et(this.glState,this.mesh.bbox.min,this.mesh.bbox.max);const e=ht(this.mesh);if(e.length){const{positions:i,normals:o}=ct(this.mesh.nodes,e);Q(this.glState,i,o,[.22,.22,.26],_);const r=ft(this.mesh.nodes,e);r.length&&tt(this.glState,r,[.34,.34,.4],R)}this.applyField()}applyField(){if(!this.glState||!this.mesh)return;if(!this.hasField||!this.fields){M(this.glState,new Float32Array(0),new Float32Array(0));return}const e=Math.min(parseInt(this.getAttribute("field-freq")||"-1",10),this.fields.length-1),i=e>=0?e:this.fields.length-1,o=parseInt(this.getAttribute("field-port")||"0",10),r=this.fields[i],n=r&&r[o];if(!n){M(this.glState,new Float32Array(0),new Float32Array(0));return}const{positions:a,abc:s}=ut(this.mesh,n);M(this.glState,a,s);const l=(this.getAttribute("field-mode")||"lin")==="log"?"log":"lin",c=dt(s,l);K(this.glState,l),l==="log"?X(this.glState,c.floor,c.range):J(this.glState,c.floor,c.range)}resolvePhase(){if(this.hasAttribute("cycle")){const i=performance.now()/1e3-this.phaseStart;let o=this.hasField?T:T.filter(n=>n!=="field");const r=Math.floor(i/rt)%o.length;return o[r]}const e=(this.getAttribute("mode")||"geometry").toLowerCase();return e==="mesh"||e==="field"?e:"geometry"}applyPhase(e){if(!this.glState)return;this.currentPhase=e;const i=e==="geometry",o=e==="mesh",r=e==="field"&&this.hasField;C(this.glState,_,i),C(this.glState,R,o),!r&&this.mesh?M(this.glState,new Float32Array(0),new Float32Array(0)):r&&this.applyField(),this.labelEl&&(this.labelEl.textContent=e,this.labelEl.style.opacity=this.hasAttribute("cycle")?"0.65":"0")}syncCanvas(){if(!this.canvas)return{w:0,h:0};const e=this.canvas.getBoundingClientRect(),i=Math.round(e.width),o=Math.round(e.height);if(i<=0||o<=0)return{w:i,h:o};const r=window.devicePixelRatio||1,n=Math.round(i*r),a=Math.round(o*r);return(this.canvas.width!==n||this.canvas.height!==a)&&(this.canvas.width=n,this.canvas.height=a),{w:i,h:o}}renderFrame(){if(!this.glState||!this.canvas||!this.mounted)return;const{w:e,h:i}=this.syncCanvas();if(e<=0||i<=0)return;this.hasAttribute("transparent");const o=parseFloat(this.getAttribute("speed")||"1");if(this.hasAttribute("rotate")&&!this.isDragging&&(this.camera={...this.camera,theta:this.camera.theta+.003*o}),this.hasAttribute("cycle")){const r=this.resolvePhase();r!==this.currentPhase&&this.applyPhase(r)}ot(this.glState,this.camera,e,i,1),this.needsRender=!1}startAnimation(){const e=++this.animId,i=this.hasAttribute("rotate")||this.hasAttribute("cycle");if(!i&&!this.hasAttribute("interactive")){this.renderFrame();return}const o=()=>{!this.mounted||e!==this.animId||((i||this.needsRender)&&this.renderFrame(),requestAnimationFrame(o))};requestAnimationFrame(o)}}typeof customElements<"u"&&!customElements.get("fem-viewer")&&customElements.define("fem-viewer",mt)})();
