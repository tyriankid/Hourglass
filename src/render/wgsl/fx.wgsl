// 凝聚/消散时的金色微光：加色混合的柔光点，透过玻璃一起被折射
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<uniform> P: Profile;
@group(0) @binding(2) var<storage, read> fxPos: array<vec4f>;
@group(0) @binding(3) var<storage, read> fxVel: array<vec4f>;

const CORNERS = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
);

struct FOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) col: vec3f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FOut {
  var o: FOut;
  let p = fxPos[ii];
  if (p.w <= 0.0) {
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  let life = clamp(p.w / max(fxVel[ii].w, 1e-3), 0.0, 1.0);
  let cv = (V.view * V.model * vec4f(p.xyz, 1.0)).xyz;
  let pixel = V.sim.w * (-cv.z);
  let r = max(0.00035, pixel * 1.5);
  o.pos = V.proj * vec4f(cv + vec3f(CORNERS[vi] * r, 0.0), 1.0);
  o.uv = CORNERS[vi];
  // 先亮起再褪去，带一点闪烁
  let tw = 0.7 + 0.3 * sin(V.misc.x * 23.0 + f32(ii) * 1.7);
  let env = sin(life * PI) * tw;
  o.col = vec3f(1.5, 0.95, 0.32) * env * min(1.0, (0.00035 * 0.00035) / (r * r) * 4.0 + 0.25);
  return o;
}

@fragment
fn fs(i: FOut) -> @location(0) vec4f {
  let d = dot(i.uv, i.uv);
  let g = exp(-d * 4.0);
  return vec4f(i.col * g, 0.0);
}
