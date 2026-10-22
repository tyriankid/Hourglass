// 玻璃外表面的正面和背面：写出视空间法线和距离，供合成时做折射
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<uniform> P: Profile;

struct GOut {
  @builtin(position) pos: vec4f,
  @location(0) nV: vec3f,
  @location(1) vpos: vec3f,
}

@vertex
fn vs(@location(0) p: vec3f, @location(1) n: vec3f) -> GOut {
  var o: GOut;
  let v = V.view * V.model * vec4f(p, 1.0);
  o.pos = V.proj * v;
  o.vpos = v.xyz;
  o.nV = (V.view * V.model * vec4f(n, 0.0)).xyz;
  return o;
}

@fragment
fn fs(i: GOut) -> @location(0) vec4f {
  return vec4f(normalize(i.nV), -i.vpos.z);
}
