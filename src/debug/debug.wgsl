// 调试渲染：颗粒团画成球体 impostor（按速度着色），沙粒画成点，玻璃画成轮廓线。

struct Cam {
  view: mat4x4f,
  proj: mat4x4f,
  model: mat4x4f,
  params: vec4f, // rc, mc, 沙粒半径, 单位深度上一个像素的世界尺寸
}

@group(0) @binding(0) var<uniform> C: Cam;
@group(0) @binding(1) var<storage, read> pts: array<vec4f>;
@group(0) @binding(2) var<storage, read> vels: array<vec4f>;

const CORNERS = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
);

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) vpos: vec3f,
  @location(1) @interpolate(flat) center: vec3f,
  @location(2) @interpolate(flat) radius: f32,
  @location(3) @interpolate(flat) col: vec3f,
}

@vertex
fn vsClump(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let p = pts[ii];
  if (p.w <= 0.0) {
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  let r = C.params.x * pow(p.w / C.params.y, 1.0 / 3.0);
  let cv = (C.view * C.model * vec4f(p.xyz, 1.0)).xyz;
  let vp = cv + vec3f(CORNERS[vi] * r * 1.4, 0.0);
  o.pos = C.proj * vec4f(vp, 1.0);
  o.vpos = vp;
  o.center = cv;
  o.radius = r;
  let sp = length(vels[ii].xyz);
  let fresh = p.w < C.params.y * 0.999;
  var base = vec3f(0.86, 0.79, 0.66);
  if (fresh) { base = vec3f(0.55, 0.75, 0.95); }
  o.col = mix(base, vec3f(1.0, 0.3, 0.08), clamp(sp * 8.0, 0.0, 1.0));
  return o;
}

struct FOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fsClump(i: VOut) -> FOut {
  let rd = normalize(i.vpos);
  let oc = -i.center;
  let b = dot(oc, rd);
  let c = dot(oc, oc) - i.radius * i.radius;
  let h = b * b - c;
  if (h < 0.0) { discard; }
  let t = -b - sqrt(h);
  let hp = rd * t;
  let n = normalize(hp - i.center);
  let clip = C.proj * vec4f(hp, 1.0);
  let l = normalize(vec3f(0.4, 0.8, 0.5));
  let diff = max(dot(n, l), 0.0) * 0.75 + 0.3;
  var o: FOut;
  o.color = vec4f(i.col * diff, 1.0);
  o.depth = clip.z / clip.w;
  return o;
}

struct GOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) alpha: f32,
}

@vertex
fn vsGrain(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GOut {
  var o: GOut;
  let p = pts[ii];
  if (p.w <= 0.0) {
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  let cv = (C.view * C.model * vec4f(p.xyz, 1.0)).xyz;
  let pixel = C.params.w * (-cv.z);
  let rg = C.params.z;
  let r = max(rg, pixel * 0.75);
  let vp = cv + vec3f(CORNERS[vi] * r, 0.0);
  o.pos = C.proj * vec4f(vp, 1.0);
  o.uv = CORNERS[vi];
  o.alpha = clamp(rg * rg / (r * r) * 3.0, 0.25, 1.0);
  return o;
}

@fragment
fn fsGrain(i: GOut) -> @location(0) vec4f {
  if (dot(i.uv, i.uv) > 1.0) { discard; }
  return vec4f(vec3f(0.62, 0.48, 0.30) * i.alpha, i.alpha);
}

struct LOut {
  @builtin(position) pos: vec4f,
}

@vertex
fn vsLine(@location(0) p: vec3f) -> LOut {
  var o: LOut;
  o.pos = C.proj * C.view * C.model * vec4f(p, 1.0);
  return o;
}

@fragment
fn fsLine() -> @location(0) vec4f {
  return vec4f(0.45, 0.5, 0.56, 1.0);
}
