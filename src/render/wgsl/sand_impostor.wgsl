// 沙面第一步：颗粒团画成球体 impostor，写出视空间距离和颗粒团 id
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<uniform> P: Profile;
@group(0) @binding(2) var<storage, read> pos: array<vec4f>;

const CORNERS = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
);

struct IOut {
  @builtin(position) pos: vec4f,
  @location(0) vpos: vec3f,
  @location(1) @interpolate(flat) center: vec3f,
  @location(2) @interpolate(flat) radius: f32,
  @location(3) @interpolate(flat) id: u32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> IOut {
  var o: IOut;
  let p = pos[ii];
  if (p.w <= 0.0) {
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    return o;
  }
  // 略放大，填平球与球之间的缝
  let r = V.sim.x * pow(p.w / V.sim.y, 1.0 / 3.0) * 1.12;
  let cv = (V.view * V.model * vec4f(p.xyz, 1.0)).xyz;
  let vp = cv + vec3f(CORNERS[vi] * r * 1.35, 0.0);
  o.pos = V.proj * vec4f(vp, 1.0);
  o.vpos = vp;
  o.center = cv;
  o.radius = r;
  o.id = ii;
  return o;
}

struct IFrag {
  @location(0) depth: f32,
  @location(1) id: u32,
  @builtin(frag_depth) z: f32,
}

@fragment
fn fs(i: IOut) -> IFrag {
  let rd = normalize(i.vpos);
  let b = dot(-i.center, rd);
  let c = dot(i.center, i.center) - i.radius * i.radius;
  let h = b * b - c;
  if (h < 0.0) { discard; }
  let t = -b - sqrt(h);
  let hp = rd * t;
  let clip = V.proj * vec4f(hp, 1.0);
  var o: IFrag;
  o.depth = -hp.z;
  o.id = i.id;
  o.z = clip.z / clip.w;
  return o;
}
