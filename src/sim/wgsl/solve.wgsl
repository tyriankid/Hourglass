// PBD 颗粒材料求解（Jacobi，一次迭代）：颗粒团之间的接触 + 静/动摩擦，颗粒团与玻璃的接触 + 摩擦。
// shock propagation：沿重力方向越低的颗粒团有效质量越大，高沙柱更稳。
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
  let mi = p.w * exp(-k * dot(p.xyz, up));
  let cc = cellOf(p.xyz);
  let nc = nCells();
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
          let mj = q.w * exp(-k * dot(q.xyz, up));
          let w = mj / (mi + mj);
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
