// 共用函数：只引用 U（SimU）和 P（Profile），所有内核都声明这两个绑定。

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rnd(s: ptr<function, u32>) -> f32 {
  *s = pcg(*s);
  return f32(*s) * (1.0 / 4294967296.0);
}

fn nCells() -> u32 { return U.gdim.x * U.gdim.y * U.gdim.z; }

fn cellOf(p: vec3f) -> vec3i { return vec3i(floor((p - U.gmin_cs.xyz) / U.gmin_cs.w)); }

fn cellIdx(c: vec3i) -> i32 {
  let d = vec3i(U.gdim.xyz);
  if (any(c < vec3i(0)) || any(c >= d)) { return -1; }
  return c.x + d.x * (c.y + d.y * c.z);
}

fn bulbH() -> f32 { return P.s[129].x; }
fn neckR() -> f32 { return P.s[129].y; }

fn profileAt(u: f32) -> vec4f {
  let x = clamp(u, 0.0, 1.0) * 128.0;
  let i = min(u32(x), 127u);
  return mix(P.s[i], P.s[i + 1u], x - f32(i));
}

// 玻璃内腔的有符号距离（腔内为正）和指向腔内的法线：vec4(d, n)
fn interiorSdf(p: vec3f) -> vec4f {
  let H = bulbH();
  let ay = abs(p.y);
  let rho = length(p.xz);
  let pr = profileAt(ay / H);
  let s = pr.y;
  let sy = select(-1.0, 1.0, p.y >= 0.0);
  var radial = vec3f(0.0);
  if (rho > 1e-9) { radial = vec3f(p.x, 0.0, p.z) / rho; }
  // f = r(|y|) - ρ，∇f = (-x/ρ, r'·sign(y), -z/ρ)
  let grad = -radial + vec3f(0.0, s * sy, 0.0);
  let gl = length(grad);
  let dLat = (pr.x - rho) / sqrt(1.0 + s * s);
  let dCap = H - ay;
  if (dCap < dLat) { return vec4f(dCap, 0.0, -sy, 0.0); }
  return vec4f(dLat, grad / max(gl, 1e-9));
}

// 非惯性系里的惯性力（单位质量）：欧拉力、离心力、科里奥利力。平移部分已并进 U.g_dt.xyz。
fn fict(x: vec3f, v: vec3f) -> vec3f {
  let w = U.omega_rc.xyz;
  let a = U.alpha_mc.xyz;
  return -cross(a, x) - cross(w, cross(w, x)) - 2.0 * cross(w, v);
}

fn clumpRadius(m: f32) -> f32 {
  return U.omega_rc.w * pow(max(m, 1e-12) / U.alpha_mc.w, 1.0 / 3.0);
}

fn gravityDir() -> vec3f {
  let g = U.g_dt.xyz;
  let l = length(g);
  if (l < 1e-6) { return vec3f(0.0, -1.0, 0.0); }
  return g / l;
}
