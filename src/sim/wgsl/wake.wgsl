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
            let q = pred[j];
            if (q.w <= 0.0) { continue; }
            let d = p.xyz - q.xyz;
            let dist = length(d);
            let rr = ri + clumpRadius(q.w);
            if (dist < 1e-9) { continue; }
            let below = dist - rr < 0.1 * ri && dot(d / dist, up) > 0.25;
            if (below) { supported = true; }
            // 身下的支撑正在往下走（漏斗流的中心通道在流）：跟着塌落
            if (below && rest[j].w >= 0.0 && dot(vel[j].xyz, up) < -0.01) { pushed = true; }
            // 被"正在运动"的邻居压过来才算：用邻居上一子步求解后的实际速度判断
            // （不能用预测位移，重力一项 g·dt² 就恰好等于 2 cm/s × dt）
            if (rest[j].w >= 0.0 && rr - dist > 0.02 * rr && length(vel[j].xyz) > 0.02) { pushed = true; }
          }
        }
      }
    }
    // 玻璃壁只有接近水平时才托得住（沙与玻璃的摩擦角约 17°）；陡峭的锥壁上沙会滑下去
    let swk = interiorSdf(p.xyz);
    if (swk.x - ri < 0.1 * ri && dot(swk.yzw, up) > 0.95) { supported = true; }
    if (!(supported && !pushed)) { rest[i] = vec4f(p.xyz, 0.0); }
}
