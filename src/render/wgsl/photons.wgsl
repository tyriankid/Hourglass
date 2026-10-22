// 桌面光照贴图：从主光方向射出光子，穿过玻璃（解析折射 + 菲涅尔）和沙（查物理网格），打到桌面上累加。
// 与物理共用 types.wgsl / fns.wgsl（SimU、Profile、网格函数）。

struct PhotonU {
  model: mat4x4f,
  modelInv: mat4x4f,
  L: vec4f,      // 指向主光的世界方向
  pu: vec4f,     // 光子平面基向量（世界）
  pv: vec4f,
  pc: vec4f,     // 光子平面中心（世界），w = 半边长
  cfg: vec4u,    // 每边光子数 N, 本批起点, 本批数量, 贴图分辨率
  lm: vec4f,     // 贴图覆盖的半边长（m），端盖厚度，端盖半径，端盖中心离瓶颈的距离
}

@group(0) @binding(0) var<uniform> Q: PhotonU;
@group(0) @binding(1) var<uniform> P: Profile;
@group(0) @binding(2) var<uniform> U: SimU;
@group(0) @binding(3) var<storage, read> grid: array<u32>;
@group(0) @binding(4) var<storage, read_write> acc: array<atomic<u32>>;

const ENERGY: f32 = 256.0;

fn surfF(p: vec3f, outer: bool) -> f32 {
  let ay = abs(p.y);
  if (ay > bulbH()) { return -1.0; }
  let pr = profileAt(ay / bulbH());
  return select(pr.x, pr.z, outer) - length(p.xz);
}

fn latheNormal(p: vec3f, outer: bool) -> vec3f {
  let pr = profileAt(abs(p.y) / bulbH());
  let s = select(pr.y, pr.w, outer);
  let sy = select(-1.0, 1.0, p.y >= 0.0);
  let rho = length(p.xz);
  var radial = vec3f(1.0, 0.0, 0.0);
  if (rho > 1e-9) { radial = vec3f(p.x, 0.0, p.z) / rho; }
  return normalize(radial - vec3f(0.0, s * sy, 0.0));
}

// 沿 [t0, t1] 找 surfF 第一次变号的位置，找不到返回 -1
fn latheHit(o: vec3f, d: vec3f, outer: bool, t0: f32, t1: f32, steps: i32) -> f32 {
  var tp = t0;
  var fp = surfF(o + d * t0, outer);
  for (var i = 1; i <= steps; i++) {
    let t = t0 + (t1 - t0) * f32(i) / f32(steps);
    let f = surfF(o + d * t, outer);
    if ((f > 0.0) != (fp > 0.0)) {
      var a = tp;
      var b = t;
      for (var k = 0; k < 10; k++) {
        let m = 0.5 * (a + b);
        if ((surfF(o + d * m, outer) > 0.0) == (fp > 0.0)) { a = m; } else { b = m; }
      }
      return 0.5 * (a + b);
    }
    tp = t;
    fp = f;
  }
  return -1.0;
}

// 光线进入包围圆柱（ρ ≤ R、|y| ≤ yMax）的区间
fn cylRange(o: vec3f, d: vec3f, R: f32, yMax: f32) -> vec2f {
  var t0 = -1e9;
  var t1 = 1e9;
  let a = dot(d.xz, d.xz);
  let b = dot(o.xz, d.xz);
  let c = dot(o.xz, o.xz) - R * R;
  if (a < 1e-12) {
    if (c > 0.0) { return vec2f(1.0, -1.0); }
  } else {
    let disc = b * b - a * c;
    if (disc < 0.0) { return vec2f(1.0, -1.0); }
    let s = sqrt(disc);
    t0 = (-b - s) / a;
    t1 = (-b + s) / a;
  }
  if (abs(d.y) < 1e-12) {
    if (abs(o.y) > yMax) { return vec2f(1.0, -1.0); }
  } else {
    let ta = (-yMax - o.y) / d.y;
    let tb = (yMax - o.y) / d.y;
    t0 = max(t0, min(ta, tb));
    t1 = min(t1, max(ta, tb));
  }
  return vec2f(t0, t1);
}

// 光线最先碰到端盖的距离（两个实心圆饼）
fn capHit(o: vec3f, d: vec3f) -> f32 {
  let H = bulbH();
  let T = Q.lm.y;
  var best = -1.0;
  for (var k = 0; k < 2; k++) {
    let sy = select(-1.0, 1.0, k == 0);
    let oc = o - vec3f(0.0, sy * (H + 0.5 * T), 0.0);
    let r = cylRange(oc, d, Q.lm.z, 0.5 * T);
    if (r.x <= r.y && r.y > 0.0) {
      let t = max(r.x, 0.0);
      if (best < 0.0 || t < best) { best = t; }
    }
  }
  return best;
}

fn fresnelT(cosT: f32) -> f32 {
  let f0 = 0.04;
  return 1.0 - (f0 + (1.0 - f0) * pow(1.0 - clamp(cosT, 0.0, 1.0), 5.0));
}

// 泡内这一段有没有被沙挡住：按 1 mm 步长查物理网格
fn blockedBySand(a: vec3f, b: vec3f) -> bool {
  let len = length(b - a);
  let n = i32(clamp(len / 0.001, 1.0, 180.0));
  for (var i = 0; i <= n; i++) {
    let p = mix(a, b, f32(i) / f32(n));
    let c = cellIdx(cellOf(p));
    if (c >= 0 && grid[u32(c)] > 0u) { return true; }
  }
  return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= Q.cfg.z) { return; }
  let id = Q.cfg.y + gid.x;
  let N = Q.cfg.x;
  if (id >= N * N) { return; }
  let S = Q.pc.w;
  // 光子平面上的位置（带抖动，减少规则网格的摩尔纹）
  let h = pcg(id * 2654435761u + 7u);
  let jx = f32(h & 0xffffu) / 65536.0;
  let jy = f32(h >> 16u) / 65536.0;
  let uu = ((f32(id % N) + jx) / f32(N) * 2.0 - 1.0) * S;
  let vv = ((f32(id / N) + jy) / f32(N) * 2.0 - 1.0) * S;
  let Lw = normalize(Q.L.xyz);
  let oW = Q.pc.xyz + Q.pu.xyz * uu + Q.pv.xyz * vv + Lw * 0.4;
  var o = (Q.modelInv * vec4f(oW, 1.0)).xyz;
  var d = normalize((Q.modelInv * vec4f(-Lw, 0.0)).xyz);
  var energy = 1.0;
  let H = bulbH();
  let Rb = P.s[129].z + 0.004;
  let eps = 2e-5;

  for (var bounce = 0; bounce < 2; bounce++) {
    let br = cylRange(o, d, Rb, H);
    let tc = capHit(o, d);
    var tOut = -1.0;
    if (br.x <= br.y && br.y > 0.0) { tOut = latheHit(o, d, true, max(br.x, 0.0), br.y, 72); }
    if (tc >= 0.0 && (tOut < 0.0 || tc < tOut)) { return; }
    if (tOut < 0.0) { break; }
    // 外表面进入玻璃
    var p = o + d * tOut;
    var n = latheNormal(p, true);
    energy *= fresnelT(-dot(d, n));
    d = refract(d, n, 1.0 / 1.5);
    let q = p + d * eps;
    let tIn = latheHit(q, d, false, 0.0, 0.012, 24);
    let tEx = latheHit(q, d, true, 0.0, 0.012, 24);
    if (tIn > 0.0 && (tEx < 0.0 || tIn < tEx)) {
      // 内表面进入泡内空气
      let p2 = q + d * tIn;
      let n2 = latheNormal(p2, false);
      energy *= fresnelT(-dot(d, n2));
      d = refract(d, n2, 1.5);
      if (dot(d, d) < 0.5) { return; }
      let q2 = p2 + d * eps;
      let tFar = latheHit(q2, d, false, 0.0, 0.2, 96);
      if (tFar < 0.0) { return; }
      let p3 = q2 + d * tFar;
      if (blockedBySand(q2, p3)) { return; }
      // 远侧内表面进入玻璃，再从外表面出去
      let n3 = latheNormal(p3, false);
      energy *= fresnelT(dot(d, n3));
      d = refract(d, -n3, 1.0 / 1.5);
      let q3 = p3 + d * eps;
      let tEx2 = latheHit(q3, d, true, 0.0, 0.012, 24);
      if (tEx2 < 0.0) { return; }
      let p4 = q3 + d * tEx2;
      let n4 = latheNormal(p4, true);
      energy *= fresnelT(dot(d, n4));
      d = refract(d, -n4, 1.5);
      if (dot(d, d) < 0.5) { return; }
      o = p4 + d * eps;
    } else if (tEx > 0.0) {
      // 掠过玻璃边缘：只在玻璃里走一段就出去
      let p4 = q + d * tEx;
      let n4 = latheNormal(p4, true);
      energy *= fresnelT(dot(d, n4));
      d = refract(d, -n4, 1.5);
      if (dot(d, d) < 0.5) { return; }
      o = p4 + d * eps;
    } else {
      return;
    }
  }
  // 打到桌面（世界 y = 0）
  let ow = (Q.model * vec4f(o, 1.0)).xyz;
  let dw = (Q.model * vec4f(d, 0.0)).xyz;
  if (dw.y >= -1e-6) { return; }
  let t = -ow.y / dw.y;
  let hit = ow + dw * t;
  let res = Q.cfg.w;
  let uv = hit.xz / (2.0 * Q.lm.x) + 0.5;
  if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) { return; }
  let ij = vec2u(uv * f32(res));
  atomicAdd(&acc[ij.y * res + ij.x], u32(energy * ENERGY));
}
