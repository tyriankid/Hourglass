// 合并（一）：每个未满的颗粒团挑一个接触着的、比自己大的未满颗粒团作为接收者
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.gdim.w) { return; }
  recv[i] = NONE;
  let p = pos[i];
  let mc = U.alpha_mc.w;
  if (p.w <= 0.0 || p.w >= mc) { return; }
  // 孔口附近的颗粒团正被抽走，不参与合并
  if (length(p.xyz) < U.drain.w * 1.5) { return; }
  let ri = clumpRadius(p.w);
  let cc = cellOf(p.xyz);
  let nc = nCells();
  var best = NONE;
  var bestM = p.w;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let c = cellIdx(cc + vec3i(dx, dy, dz));
        if (c < 0) { continue; }
        let cnt = min(grid[u32(c)], SLOTS);
        for (var s = 0u; s < cnt; s++) {
          let j = grid[nc + u32(c) * SLOTS + s];
          if (j == i) { continue; }
          let q = pos[j];
          if (q.w <= 0.0 || q.w >= mc || q.y * p.y < 0.0) { continue; }
          let rr = ri + clumpRadius(q.w);
          let d = q.xyz - p.xyz;
          if (dot(d, d) > rr * rr * 1.1) { continue; }
          if (q.w > bestM || (q.w == bestM && j > i)) { bestM = q.w; best = j; }
        }
      }
    }
  }
  recv[i] = best;
}
