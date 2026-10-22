// PBD 颗粒材料求解（Jacobi，一次迭代）：颗粒团之间的接触 + 静/动摩擦，颗粒团与玻璃的接触 + 摩擦。
// shock propagation：沿重力方向越低的颗粒团有效质量越大，高沙柱更稳。
// 休眠：静止的颗粒团被锁定（相当于静摩擦锁住），对邻居而言质量无穷大；
// 每个子步的第一轮迭代检查它是否失去支撑或被明显压入，是则唤醒。
override FIRST: bool = false;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.gdim.w) { return; }
  let p = predIn[i];
  if (p.w <= 0.0) { predOut[i] = p; return; }
  let x0 = pos[i].xyz;
  let ri = clumpRadius(p.w);
  let up = -gravityDir();
  let k = U.misc.x;
  let cc = cellOf(p.xyz);
  let nc = nCells();
  var asleep = rest[i].w < 0.0;
  if (asleep) {
    if (!FIRST) { predOut[i] = p; return; }
    // 唤醒检查：下方有没有紧贴的支撑（颗粒团或玻璃），有没有被醒着的邻居明显压入
    var supported = false;
    var pushed = false;
    for (var dz = -1; dz <= 1; dz++) {
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          let c = cellIdx(cc + vec3i(dx, dy, dz));
          if (c < 0) { continue; }
          let cnt = min(grid[u32(c)], SLOTS);
          for (var s = 0u; s < cnt; s++) {
            let j = grid[nc + u32(c) * SLOTS + s];
            if (j == i) { continue; }
            let q = predIn[j];
            if (q.w <= 0.0) { continue; }
            let d = p.xyz - q.xyz;
            let dist = length(d);
            let rr = ri + clumpRadius(q.w);
            if (dist < 1e-9) { continue; }
            if (dist - rr < 0.1 * ri && dot(d / dist, up) > 0.25) { supported = true; }
            if (rest[j].w >= 0.0 && rr - dist > 0.05 * rr) { pushed = true; }
          }
        }
      }
    }
    let swk = interiorSdf(p.xyz);
    if (swk.x - ri < 0.1 * ri && dot(swk.yzw, up) > 0.25) { supported = true; }
    if (supported && !pushed) { predOut[i] = p; return; }
    rest[i] = vec4f(p.xyz, 0.0);
    asleep = false;
  }
  var dsum = vec3f(0.0);
  var n = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let c = cellIdx(cc + vec3i(dx, dy, dz));
        if (c < 0) { continue; }
        let cnt = min(grid[u32(c)], SLOTS);
        for (var s = 0u; s < cnt; s++) {
          let j = grid[nc + u32(c) * SLOTS + s];
          if (j == i) { continue; }
          let q = predIn[j];
          if (q.w <= 0.0) { continue; }
          let d = p.xyz - q.xyz;
          let d2 = dot(d, d);
          let rr = ri + clumpRadius(q.w);
          if (d2 >= rr * rr || d2 < 1e-20) { continue; }
          let dist = sqrt(d2);
          let nrm = d / dist;
          let pen = rr - dist;
          // shock propagation 的质量比只取决于两者的高度差：w = mj/(mi+mj) = 1/(1 + (m_i/m_j)·e^{-k(h_i-h_j)})，
          // 指数截断防溢出；休眠的邻居不可移动，修正全部由自己承担
          let ex = clamp(-k * dot(p.xyz - q.xyz, up), -30.0, 30.0);
          let w = select(1.0 / (1.0 + (p.w / q.w) * exp(ex)), 1.0, rest[j].w < 0.0);
          dsum += w * pen * nrm;
          let dp = (p.xyz - x0) - (q.xyz - pos[j].xyz);
          let dt = dp - dot(dp, nrm) * nrm;
          let lt = length(dt);
          if (lt > 1e-12) {
            if (lt < U.fric.x * pen) { dsum -= w * dt; } else { dsum -= w * dt * min(U.fric.y * pen / lt, 1.0); }
          }
          n += 1.0;
        }
      }
    }
  }
  let sw = interiorSdf(p.xyz);
  let penW = ri - sw.x;
  if (penW > 0.0) {
    let nw = sw.yzw;
    dsum += penW * nw;
    let dp = p.xyz - x0;
    let dt = dp - dot(dp, nw) * nw;
    let lt = length(dt);
    if (lt > 1e-12) { dsum -= dt * min(U.fric.z * penW / lt, 1.0); }
    n += 1.0;
  }
  // ints.y = 1：按接触数平均（Macklin）；0：直接累加再乘松弛系数
  var np = p.xyz + U.misc.w * dsum / select(1.0, max(n, 1.0), U.ints.y == 1u);
  let s2 = interiorSdf(np);
  if (s2.x < ri) { np += (ri - s2.x) * s2.yzw; }
  if (p.y * np.y < 0.0) { np.y = p.y; }
  predOut[i] = vec4f(np, p.w);
}
