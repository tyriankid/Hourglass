// 唤醒检查（每个子步一次，在求解之前）：休眠的颗粒团若失去下方支撑，或被醒着的邻居明显压入，就醒来
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.gdim.w) { return; }
  let p = pred[i];
  if (p.w <= 0.0 || rest[i].w >= 0.0) { return; }
  let ri = clumpRadius(p.w);
  let up = -gravityDir();
  let cc = cellOf(p.xyz);
  let nc = nCells();
    // 唤醒检查：静摩擦能不能托住它。
    // 单个接触：法线与竖直方向的夹角要在静摩擦角（约 29°）以内；
    // 多个接触：它们的水平方向要分布在两侧把颗粒"托住"（贴玻璃壁 + 另一侧有邻居，也算托住）。
    var pushed = false;
    var maxDot = 0.0;
    var nBelow = 0.0;
    var hsum = vec3f(0.0);
    for (var dz = -1; dz <= 1; dz++) {
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          let c = cellIdx(cc + vec3i(dx, dy, dz));
          if (c < 0) { continue; }
          let cnt = min(grid[u32(c)], SLOTS);
          for (var s = 0u; s < cnt; s++) {
            let j = grid[nc + u32(c) * SLOTS + s];
            if (j == i) { continue; }
            let q = pred[j];
            if (q.w <= 0.0) { continue; }
            let d = p.xyz - q.xyz;
            let dist = length(d);
            let rr = ri + clumpRadius(q.w);
            if (dist < 1e-9) { continue; }
            let n = d / dist;
            let du = dot(n, up);
            let touching = dist - rr < 0.1 * ri;
            if (touching && du > 0.2) {
              maxDot = max(maxDot, du);
              nBelow += 1.0;
              let h = n - du * up;
              if (length(h) > 1e-6) { hsum += normalize(h); }
              // 身下的支撑正在往下走（漏斗流的中心通道在流）：跟着塌落
              if (rest[j].w >= 0.0 && dot(vel[j].xyz, up) < -0.01) { pushed = true; }
            }
            // 被"正在运动"的邻居压过来才算：用邻居上一子步求解后的实际速度判断
            // （不能用预测位移，重力一项 g·dt² 就恰好等于 2 cm/s × dt）
            if (rest[j].w >= 0.0 && rr - dist > 0.02 * rr && length(vel[j].xyz) > 0.02) { pushed = true; }
          }
        }
      }
    }
    // 玻璃壁：作为一个接触参与判断（与沙的摩擦角约 17°，单独托住要求接近水平）
    let swk = interiorSdf(p.xyz);
    if (swk.x - ri < 0.1 * ri) {
      let du = dot(swk.yzw, up);
      if (du > 0.95) { maxDot = 1.0; }
      if (du > -0.2) {
        nBelow += 1.0;
        let h = swk.yzw - du * up;
        if (length(h) > 1e-6) { hsum += normalize(h); }
      }
    }
    let supported = maxDot > 0.87 || (nBelow >= 2.0 && length(hsum) < 0.7 * nBelow);
    if (!(supported && !pushed)) { rest[i] = vec4f(p.xyz, 0.0); }
}
