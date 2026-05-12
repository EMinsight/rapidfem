(function(){"use strict";const _={bg:"#131316"};function R(){return{theta:-Math.PI/4,phi:Math.PI/5,distance:300,target:[0,0,0]}}const B=`#version 300 es
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
}`,I=`#version 300 es
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
}`,T=`#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }`,V=`#version 300 es
precision highp float;
uniform vec3 uColor;
out vec4 fragColor;
void main() { fragColor = vec4(uColor, 1.0); }`,D=`#version 300 es
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
}`,U=`#version 300 es
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
}`;function S(t,e,i){const o=t.createShader(e);if(t.shaderSource(o,i),t.compileShader(o),!t.getShaderParameter(o,t.COMPILE_STATUS)){const r=t.getShaderInfoLog(o);throw t.deleteShader(o),new Error("Shader compile: "+r)}return o}function w(t,e,i){const o=S(t,t.VERTEX_SHADER,e),r=S(t,t.FRAGMENT_SHADER,i),n=t.createProgram();if(t.attachShader(n,o),t.attachShader(n,r),t.linkProgram(n),!t.getProgramParameter(n,t.LINK_STATUS))throw new Error("Program link: "+t.getProgramInfoLog(n));return t.deleteShader(o),t.deleteShader(r),n}function N(t){const e=parseInt(t.slice(1,3),16)/255,i=parseInt(t.slice(3,5),16)/255,o=parseInt(t.slice(5,7),16)/255;return[e,i,o]}function z(t,e,i,o){const r=1/Math.tan(t/2),n=1/(i-o),s=new Float32Array(16);return s[0]=r/e,s[5]=r,s[10]=(o+i)*n,s[11]=-1,s[14]=2*o*i*n,s}function k(t,e,i){const o=t[0]-e[0],r=t[1]-e[1],n=t[2]-e[2];let s=Math.sqrt(o*o+r*r+n*n);const a=o/s,h=r/s,l=n/s,c=i[1]*l-i[2]*h,f=i[2]*a-i[0]*l,d=i[0]*h-i[1]*a;s=Math.sqrt(c*c+f*f+d*d);const v=c/s,p=f/s,m=d/s,b=h*m-l*p,A=l*v-a*m,x=a*p-h*v,u=new Float32Array(16);return u[0]=v,u[1]=b,u[2]=a,u[4]=p,u[5]=A,u[6]=h,u[8]=m,u[9]=x,u[10]=l,u[12]=-(v*t[0]+p*t[1]+m*t[2]),u[13]=-(b*t[0]+A*t[1]+x*t[2]),u[14]=-(a*t[0]+h*t[1]+l*t[2]),u[15]=1,u}function O(t,e){const i=new Float32Array(16);for(let o=0;o<4;o++)for(let r=0;r<4;r++)i[r*4+o]=t[o]*e[r*4]+t[4+o]*e[r*4+1]+t[8+o]*e[r*4+2]+t[12+o]*e[r*4+3];return i}function Y(t){const e=new Float32Array(9);return e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[4],e[4]=t[5],e[5]=t[6],e[6]=t[8],e[7]=t[9],e[8]=t[10],e}function q(t){const e=Math.cos(t.phi);return[t.target[0]+t.distance*e*Math.sin(t.theta),t.target[1]+t.distance*e*Math.cos(t.theta),t.target[2]+t.distance*Math.sin(t.phi)]}function Z(t){const e=t.getContext("webgl2",{antialias:!0,alpha:!0,preserveDrawingBuffer:!0});if(!e)return null;const i=w(e,B,I),o=w(e,T,V),r=w(e,D,U),n=N(_.bg);return e.clearColor(n[0],n[1],n[2],1),e.enable(e.DEPTH_TEST),e.disable(e.CULL_FACE),{gl:e,program:i,uMVP:e.getUniformLocation(i,"uMVP"),uNormalMat:e.getUniformLocation(i,"uNormalMat"),uColor:e.getUniformLocation(i,"uColor"),uLightDir:e.getUniformLocation(i,"uLightDir"),uAmbient:e.getUniformLocation(i,"uAmbient"),uZFlip:e.getUniformLocation(i,"uZFlip"),uColormap:e.getUniformLocation(i,"uColormap"),uClipPlane:e.getUniformLocation(i,"uClipPlane"),uClipEnable:e.getUniformLocation(i,"uClipEnable"),clip_plane:[0,0,1,0],clip_enable:!1,lineProgram:o,uLineMVP:e.getUniformLocation(o,"uMVP"),uLineColor:e.getUniformLocation(o,"uColor"),pointProgram:r,uPointMVP:e.getUniformLocation(r,"uMVP"),uPointZFlip:e.getUniformLocation(r,"uZFlip"),uPointScale:e.getUniformLocation(r,"uPointScale"),uPointPhase:e.getUniformLocation(r,"uPhase"),uPointLogFloor:e.getUniformLocation(r,"uLogFloor"),uPointLogRange:e.getUniformLocation(r,"uLogRange"),uPointLogScale:e.getUniformLocation(r,"uLogScale"),meshes:[],lineMeshes:[],pointCloud:null,pointPhase:0,pointLogFloor:-30,pointLogRange:6,pointLogScale:1,bbox:{min:[0,0,0],max:[0,0,0]}}}function G(t){const{gl:e}=t;for(const i of t.meshes){e.deleteVertexArray(i.vao);for(const o of i.buffers)e.deleteBuffer(o)}for(const i of t.lineMeshes){e.deleteVertexArray(i.vao);for(const o of i.buffers)e.deleteBuffer(o)}e.deleteProgram(t.program),e.deleteProgram(t.lineProgram)}function W(t){const{gl:e}=t;for(const i of t.meshes){e.deleteVertexArray(i.vao);for(const o of i.buffers)e.deleteBuffer(o)}t.meshes=[];for(const i of t.lineMeshes){e.deleteVertexArray(i.vao);for(const o of i.buffers)e.deleteBuffer(o)}t.lineMeshes=[]}function L(t,e,i){const{gl:o}=t;if(t.pointCloud){o.deleteVertexArray(t.pointCloud.vao);for(const a of t.pointCloud.buffers)o.deleteBuffer(a)}if(e.length===0){t.pointCloud=null;return}const r=o.createVertexArray();o.bindVertexArray(r);const n=o.createBuffer();o.bindBuffer(o.ARRAY_BUFFER,n),o.bufferData(o.ARRAY_BUFFER,e,o.STATIC_DRAW),o.enableVertexAttribArray(0),o.vertexAttribPointer(0,3,o.FLOAT,!1,0,0);const s=o.createBuffer();o.bindBuffer(o.ARRAY_BUFFER,s),o.bufferData(o.ARRAY_BUFFER,i,o.STATIC_DRAW),o.enableVertexAttribArray(1),o.vertexAttribPointer(1,3,o.FLOAT,!1,0,0),o.bindVertexArray(null),t.pointCloud={vao:r,buffers:[n,s],count:e.length/3}}function H(t,e,i){t.pointLogFloor=e,t.pointLogRange=i}function $(t,e){t.pointLogScale=e==="log"?1:0}function j(t,e,i){t.pointLogFloor=e,t.pointLogRange=i}function X(t,e){t.pointPhase=e}function E(t,e,i,o,r=0,n,s){const{gl:a}=t,h=a.createVertexArray();a.bindVertexArray(h);const l=a.createBuffer();a.bindBuffer(a.ARRAY_BUFFER,l),a.bufferData(a.ARRAY_BUFFER,e,a.STATIC_DRAW),a.enableVertexAttribArray(0),a.vertexAttribPointer(0,3,a.FLOAT,!1,0,0);const c=a.createBuffer();a.bindBuffer(a.ARRAY_BUFFER,c),a.bufferData(a.ARRAY_BUFFER,i,a.STATIC_DRAW),a.enableVertexAttribArray(1),a.vertexAttribPointer(1,3,a.FLOAT,!1,0,0);const f=[l,c];let d=!1;a.disableVertexAttribArray(2),a.vertexAttrib1f(2,0),a.bindVertexArray(null),t.meshes.push({vao:h,buffers:f,count:e.length/3,color:o,tag:r,visible:!0,depth_offset:n,has_scalar:d})}function K(t,e,i){t.bbox.min=e,t.bbox.max=i}function J(t,e,i,o,r=1){const{gl:n}=t;n.viewport(0,0,i,o),n.clear(n.COLOR_BUFFER_BIT|n.DEPTH_BUFFER_BIT);const s=i/o||1,a=t.bbox.max[0]-t.bbox.min[0],h=t.bbox.max[1]-t.bbox.min[1],l=t.bbox.max[2]-t.bbox.min[2],c=.5*Math.sqrt(a*a+h*h+l*l),f=Math.max(e.distance*.001,c*.001,1e-9),d=(e.distance+c)*8,v=z(Math.PI/6,s,f,d),p=q(e),m=k(p,e.target,[0,0,1]),b=O(v,m),A=Y(m),x=p[0]-e.target[0],u=p[1]-e.target[1],P=p[2]-e.target[2]+e.distance*.3,M=Math.sqrt(x*x+u*u+P*P),F=[x/M,u/M,P/M];n.useProgram(t.program),n.uniformMatrix4fv(t.uMVP,!1,b),n.uniformMatrix3fv(t.uNormalMat,!1,A),n.uniform3f(t.uLightDir,F[0],F[1],F[2]),n.uniform1f(t.uAmbient,.8),n.uniform1f(t.uZFlip,r),n.uniform4f(t.uClipPlane,t.clip_plane[0],t.clip_plane[1],t.clip_plane[2],t.clip_plane[3]),n.uniform1f(t.uClipEnable,t.clip_enable?1:0);let y=!1;for(const g of t.meshes)g.visible&&(g.depth_offset?(y||(n.enable(n.POLYGON_OFFSET_FILL),y=!0),n.polygonOffset(g.depth_offset[0],g.depth_offset[1])):y&&(n.disable(n.POLYGON_OFFSET_FILL),y=!1),n.uniform3f(t.uColor,g.color[0],g.color[1],g.color[2]),n.uniform1f(t.uColormap,g.has_scalar?1:0),n.bindVertexArray(g.vao),n.drawArrays(n.TRIANGLES,0,g.count));if(y&&n.disable(n.POLYGON_OFFSET_FILL),t.lineMeshes.length>0){n.useProgram(t.lineProgram),n.uniformMatrix4fv(t.uLineMVP,!1,b);for(const g of t.lineMeshes)g.visible&&(n.uniform3f(t.uLineColor,g.color[0],g.color[1],g.color[2]),n.bindVertexArray(g.vao),n.drawArrays(n.LINES,0,g.count))}if(t.pointCloud&&t.pointCloud.count>0){n.useProgram(t.pointProgram),n.uniformMatrix4fv(t.uPointMVP,!1,b),n.uniform1f(t.uPointZFlip,r);const g=t.bbox.max[0]-t.bbox.min[0],lt=t.bbox.max[1]-t.bbox.min[1],ct=Math.max(g,lt,1e-9);n.uniform1f(t.uPointScale,ct*.4),n.uniform1f(t.uPointPhase,t.pointPhase),n.uniform1f(t.uPointLogFloor,t.pointLogFloor),n.uniform1f(t.uPointLogRange,t.pointLogRange),n.uniform1f(t.uPointLogScale,t.pointLogScale),n.disable(n.DEPTH_TEST),n.depthMask(!1),n.enable(n.BLEND),n.blendFunc(n.ONE,n.ONE),n.bindVertexArray(t.pointCloud.vao),n.drawArrays(n.POINTS,0,t.pointCloud.count),n.disable(n.BLEND),n.depthMask(!0),n.enable(n.DEPTH_TEST)}n.bindVertexArray(null)}function Q(t,e){const i=(t[0]+e[0])/2,o=(t[1]+e[1])/2,r=(t[2]+e[2])/2,n=e[0]-t[0],s=e[1]-t[1],a=e[2]-t[2],h=Math.max(Math.sqrt(n*n+s*s+a*a),1e-9),l=Math.PI/12,c=h*.6/Math.tan(l)*1.05;return{theta:Math.PI/4,phi:Math.PI/4,distance:c,target:[i,o,r]}}const tt=1380339014;function et(t){for(let e=t.cells.length-1;e>=0;e--)for(const i of t.cells[e].display_events)if(i.kind==="mesh"&&i.payload)return i.payload;return null}function ot(t){for(let e=t.cells.length-1;e>=0;e--)for(const i of t.cells[e].display_events)if(i.kind==="result"&&i.payload)return i.payload;return null}async function nt(t,e){const i=new URL(t.url,e).href,o=await fetch(i);if(!o.ok)throw new Error(`bin fetch ${o.status}`);const r=await o.arrayBuffer(),n=new DataView(r);if(n.getUint32(0,!0)!==tt)throw new Error("field bin: bad magic");const s=n.getUint32(8,!0),a=n.getUint32(12,!0),h=n.getUint32(16,!0),l=20,c=new Uint8Array(r,l,s*a),f=l+(c.byteLength+3&-4),d=new Float32Array(r,f),v=[];let p=0;for(let m=0;m<s;m++){const b=[];for(let A=0;A<a;A++)c[m*a+A]===0?b.push(null):(b.push(Array.from(d.subarray(p,p+h))),p+=h);v.push(b)}return v}function it(t){const e=t.tets,i=t.tet_phys,o=i.length,r=(a,h,l)=>{const c=[a,h,l].sort((f,d)=>f-d);return(BigInt(c[0])*0x100000000n+BigInt(c[1]))*0x100000000n+BigInt(c[2])},n=new Map;for(let a=0;a<o;a++){const h=i[a];if(!h)continue;let l=n.get(h);l||(l=[],n.set(h,l)),l.push(a)}const s=[];for(const[,a]of n.entries()){const h=new Map;for(const l of a){const c=e[l*4],f=e[l*4+1],d=e[l*4+2],v=e[l*4+3],p=[[c,f,d],[c,f,v],[c,d,v],[f,d,v]];for(const m of p){const b=r(m[0],m[1],m[2]),A=h.get(b);A?A.count++:h.set(b,{count:1,tri:m})}}for(const l of h.values())l.count===1&&s.push(l.tri[0],l.tri[1],l.tri[2])}return s}function C(t,e){const i=e.length/3,o=new Float32Array(i*9),r=new Float32Array(i*9),n=[0,0,0],s=[0,0,0],a=[0,0,0],h=[0,0,0],l=[0,0,0];for(let c=0;c<i;c++){const f=e[c*3],d=e[c*3+1],v=e[c*3+2];n[0]=t[f*3],n[1]=t[f*3+1],n[2]=t[f*3+2],s[0]=t[d*3],s[1]=t[d*3+1],s[2]=t[d*3+2],a[0]=t[v*3],a[1]=t[v*3+1],a[2]=t[v*3+2],h[0]=s[0]-n[0],h[1]=s[1]-n[1],h[2]=s[2]-n[2],l[0]=a[0]-n[0],l[1]=a[1]-n[1],l[2]=a[2]-n[2];const p=h[1]*l[2]-h[2]*l[1],m=h[2]*l[0]-h[0]*l[2],b=h[0]*l[1]-h[1]*l[0],A=1/Math.max(Math.hypot(p,m,b),1e-20),x=p*A,u=m*A,P=b*A;o.set(n,c*9),o.set(s,c*9+3),o.set(a,c*9+6),r[c*9]=x,r[c*9+1]=u,r[c*9+2]=P,r[c*9+3]=x,r[c*9+4]=u,r[c*9+5]=P,r[c*9+6]=x,r[c*9+7]=u,r[c*9+8]=P}return{positions:o,normals:r}}function rt(t,e){const i=t.nodes,o=t.tets,r=o.length/4,n=new Float32Array(r*3),s=new Float32Array(r*3);for(let a=0;a<r;a++){const h=o[a*4],l=o[a*4+1],c=o[a*4+2],f=o[a*4+3];n[a*3]=.25*(i[h*3]+i[l*3]+i[c*3]+i[f*3]),n[a*3+1]=.25*(i[h*3+1]+i[l*3+1]+i[c*3+1]+i[f*3+1]),n[a*3+2]=.25*(i[h*3+2]+i[l*3+2]+i[c*3+2]+i[f*3+2]),s[a*3]=.25*(e[h*3]+e[l*3]+e[c*3]+e[f*3]),s[a*3+1]=.25*(e[h*3+1]+e[l*3+1]+e[c*3+1]+e[f*3+1]),s[a*3+2]=.25*(e[h*3+2]+e[l*3+2]+e[c*3+2]+e[f*3+2])}return{positions:n,abc:s}}function at(t,e){let i=0;for(let r=0;r<t.length;r+=3){const n=Math.max(t[r],t[r+1]);n>i&&(i=n)}const o=Math.sqrt(Math.max(i,1e-30));return e==="log"?{floor:Math.log10(o)-4,range:4}:{floor:0,range:o}}class st extends HTMLElement{canvas=null;wrapper=null;loadingEl=null;glState=null;camera=R();animId=0;mounted=!1;mesh=null;fields=null;needsRender=!1;isDragging=!1;isRightDrag=!1;lastMouse={x:0,y:0};static get observedAttributes(){return["src","width","height","rotate","interactive","transparent","speed","theta","phi","show-geometry","show-mesh","show-field","field-mode","field-freq","field-port","animate-field"]}connectedCallback(){this.mounted=!0;const e=this.attachShadow({mode:"open"}),i=this.hasAttribute("transparent");this.wrapper=document.createElement("div"),this.wrapper.style.cssText=`position:relative;width:${this.getAttribute("width")||"100%"};height:${this.getAttribute("height")||"400px"};background:${i?"transparent":"#131316"};overflow:hidden;border-radius:inherit;`,this.canvas=document.createElement("canvas"),this.canvas.style.cssText=`display:block;width:100%;height:100%;cursor:${this.hasAttribute("interactive")?"grab":"default"};`,this.wrapper.appendChild(this.canvas),this.loadingEl=document.createElement("div"),this.loadingEl.style.cssText="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font:500 11px/1 monospace;color:#55535a;",this.loadingEl.textContent="",this.wrapper.appendChild(this.loadingEl);const o=document.createElement("a");if(o.href="https://fem.rapidpassives.org",o.target="_blank",o.rel="noopener",o.textContent="RapidFEM",o.style.cssText="position:absolute;bottom:6px;right:8px;font:500 9px/1 monospace;color:#55535a;text-decoration:none;opacity:0.7;transition:opacity 0.15s;",o.onmouseenter=()=>o.style.opacity="1",o.onmouseleave=()=>o.style.opacity="0.7",this.wrapper.appendChild(o),e.appendChild(this.wrapper),this.glState=Z(this.canvas),!this.glState)return;this.hasAttribute("interactive")&&(this.canvas.addEventListener("pointerdown",s=>this.onPointerDown(s)),this.canvas.addEventListener("pointermove",s=>this.onPointerMove(s)),this.canvas.addEventListener("pointerup",()=>this.onPointerUp()),this.canvas.addEventListener("wheel",s=>this.onWheel(s),{passive:!1}),this.canvas.addEventListener("contextmenu",s=>s.preventDefault()),this.canvas.addEventListener("dblclick",()=>this.fitView())),new ResizeObserver(()=>{this.needsRender=!0}).observe(this.wrapper);const n=this.getAttribute("src");n&&this.load(n)}disconnectedCallback(){this.mounted=!1,this.animId++,this.glState&&G(this.glState)}attributeChangedCallback(e,i,o){this.mounted&&(e==="src"&&o?this.load(o):(e==="field-mode"||e==="field-freq"||e==="field-port")&&(this.applyField(),this.needsRender=!0))}onPointerDown(e){this.isDragging=!0,this.isRightDrag=e.button===2,this.lastMouse={x:e.clientX,y:e.clientY},this.canvas?.setPointerCapture(e.pointerId),this.canvas&&(this.canvas.style.cursor="grabbing")}onPointerMove(e){if(!this.isDragging)return;const i=e.clientX-this.lastMouse.x,o=e.clientY-this.lastMouse.y;if(this.lastMouse={x:e.clientX,y:e.clientY},this.isRightDrag){const r=this.camera.distance*7e-4,n=Math.cos(this.camera.theta),s=Math.sin(this.camera.theta);this.camera={...this.camera,target:[this.camera.target[0]+(i*n-o*s*Math.sin(this.camera.phi))*r,this.camera.target[1]-(i*s+o*n*Math.sin(this.camera.phi))*r,this.camera.target[2]+o*Math.cos(this.camera.phi)*r]}}else this.camera={...this.camera,theta:this.camera.theta+i*.005,phi:Math.max(.05,Math.min(Math.PI/2-.05,this.camera.phi+o*.005))};this.needsRender=!0}onPointerUp(){this.isDragging=!1,this.isRightDrag=!1,this.canvas&&(this.canvas.style.cursor="grab")}onWheel(e){e.preventDefault(),this.camera={...this.camera,distance:this.camera.distance*(e.deltaY>0?1.1:1/1.1)},this.needsRender=!0}fitView(){if(!this.glState||!this.mesh)return;this.camera=Q(this.mesh.bbox.min,this.mesh.bbox.max);const e=parseFloat(this.getAttribute("theta")||"45")*Math.PI/180,i=parseFloat(this.getAttribute("phi")||"45")*Math.PI/180;this.camera={...this.camera,theta:e,phi:i},this.needsRender=!0}async load(e){this.loadingEl&&(this.loadingEl.textContent="Loading...",this.loadingEl.style.display="flex");try{const i=new URL(e,location.href).href,o=await fetch(i);if(!o.ok)throw new Error(`HTTP ${o.status}`);const r=await o.json();if(this.mesh=et(r),!this.mesh)throw new Error("no mesh payload in bake");const n=ot(r);n&&n.fields&&n.fields.$bin?(this.loadingEl&&(this.loadingEl.textContent="Decoding field..."),this.fields=await nt(n.fields,i)):this.fields=null,this.rebuildScene(),this.fitView(),this.loadingEl&&(this.loadingEl.style.display="none"),this.startAnimation()}catch(i){console.error("[fem-viewer] load failed",i),this.loadingEl&&(this.loadingEl.textContent=`Error: ${i.message}`)}}rebuildScene(){if(!this.glState||!this.mesh)return;if(W(this.glState),K(this.glState,this.mesh.bbox.min,this.mesh.bbox.max),this.attrBool("show-geometry",!0)){const i=it(this.mesh);if(i.length){const{positions:o,normals:r}=C(this.mesh.nodes,i);E(this.glState,o,r,[.16,.16,.2],1)}if(this.mesh.tris.length){const{positions:o,normals:r}=C(this.mesh.nodes,this.mesh.tris);E(this.glState,o,r,[.36,.3,.34],2)}}this.applyField()}applyField(){if(!this.glState||!this.mesh)return;if(!this.attrBool("show-field",this.fields!==null)||!this.fields){L(this.glState,new Float32Array(0),new Float32Array(0));return}const i=Math.min(parseInt(this.getAttribute("field-freq")||"-1",10),this.fields.length-1),o=i>=0?i:this.fields.length-1,r=parseInt(this.getAttribute("field-port")||"0",10),n=this.fields[o],s=n&&n[r];if(!s){L(this.glState,new Float32Array(0),new Float32Array(0));return}const{positions:a,abc:h}=rt(this.mesh,s);L(this.glState,a,h);const l=(this.getAttribute("field-mode")||"lin")==="log"?"log":"lin",c=at(h,l);$(this.glState,l),l==="log"?H(this.glState,c.floor,c.range):j(this.glState,c.floor,c.range)}attrBool(e,i){if(!this.hasAttribute(e))return i;const o=this.getAttribute(e);return o===null||o===""||o==="true"||o==="1"}syncCanvas(){if(!this.canvas)return{w:0,h:0};const e=this.canvas.getBoundingClientRect(),i=Math.round(e.width),o=Math.round(e.height);if(i<=0||o<=0)return{w:i,h:o};const r=window.devicePixelRatio||1,n=Math.round(i*r),s=Math.round(o*r);return(this.canvas.width!==n||this.canvas.height!==s)&&(this.canvas.width=n,this.canvas.height=s),{w:i,h:o}}renderFrame(e=0){if(!this.glState||!this.canvas||!this.mounted)return;const{w:i,h:o}=this.syncCanvas();if(i<=0||o<=0)return;const r=this.hasAttribute("rotate"),n=this.hasAttribute("animate-field");this.hasAttribute("transparent");const s=parseFloat(this.getAttribute("speed")||"1");if(r&&!this.isDragging&&(this.camera={...this.camera,theta:this.camera.theta+.003*s}),n){const a=e*.001*s%(2*Math.PI);X(this.glState,a)}J(this.glState,this.camera,i,o,0),this.needsRender=!1}startAnimation(){const e=++this.animId,i=this.hasAttribute("rotate")||this.hasAttribute("animate-field");if(!i&&!this.hasAttribute("interactive")){this.renderFrame(0);return}const o=r=>{!this.mounted||e!==this.animId||((i||this.needsRender)&&this.renderFrame(r),requestAnimationFrame(o))};requestAnimationFrame(o)}}typeof customElements<"u"&&!customElements.get("fem-viewer")&&customElements.define("fem-viewer",st)})();
