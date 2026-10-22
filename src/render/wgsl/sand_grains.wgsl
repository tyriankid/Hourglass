// 沙粒：按真实尺寸投影，亚像素时用覆盖率混合；下落时翻滚，偶尔闪一下
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<uniform> P: Profile;
@group(0) @binding(2) var<storage, read> grains: array<vec4f>;
@group(0) @binding(3) var<storage, read> gvel: array<vec4f>;

const CORNERS = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
);

struct GOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) col: vec4f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GOut {
  var o: GOut;
  let g = grains[ii];
  if (g.w <= 0.0) {
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  let cv = (V.view * V.model * vec4f(g.xyz, 1.0)).xyz;
  let pixel = V.sim.w * (-cv.z);
  let rg = V.sim.z;
  // 至少一个像素宽，透明度按真实投影面积换算（方形四边形面积 4r² 对应圆面积 πrg²）
  let r = max(rg, pixel * 0.75);
  o.pos = V.proj * vec4f(cv + vec3f(CORNERS[vi] * r, 0.0), 1.0);
  o.uv = CORNERS[vi];
  let cover = clamp(0.785 * (rg * rg) / (r * r), 0.02, 1.0);
  let h = pcg(ii * 747796405u + u32(V.misc.y) / 3u);
  var tone = vec3f(0.68, 0.52, 0.31);
  let r0 = u01(pcg(ii + 11u));
  if (r0 < 0.07) { tone = vec3f(0.20, 0.15, 0.10); } else if (r0 < 0.19) { tone = vec3f(0.80, 0.78, 0.72); }
  var col = tone * (0.55 + 0.5 * V.light.w);
  if (u01(h) < 0.03) { col += vec3f(4.0, 3.9, 3.6); }
  o.col = vec4f(col * cover, cover);
  return o;
}

@fragment
fn fs(i: GOut) -> @location(0) vec4f {
  return i.col;
}
