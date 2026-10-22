// 预稳定（Macklin 2014 §5.1）：每个子步开始前，只在"当前位置"上把已有的重叠推开，不改变速度。
// 否则休眠时被锁住的残余重叠在醒来时会被一步推开，PBD 会把这一步位移当成速度——颗粒被"弹飞"。
// 与 solve_gs 一样按网格 8 色并行；只处理穿透，不做摩擦；休眠的邻居不可移动。
override COLOR: u32 = 0u;

fn stabOne(i: u32) {
  let p0 = pos[i];
  if (p0.w <= 0.0 || rest[i].w < 0.0) { return; }
  let ri = clumpRadius(p0.w);
  let up = -gravityDir();
  let k = U.misc.x;
  let cc = cellOf(p0.xyz);
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
          let q = pos[j];
          if (q.w <= 0.0) { continue; }
          let d = p0.xyz - q.xyz;
          let d2 = dot(d, d);
          let rr = ri + clumpRadius(q.w);
          if (d2 >= rr * rr || d2 < 1e-20) { continue; }
          let dist = sqrt(d2);
          let ex = clamp(-k * dot(d, up), -30.0, 30.0);
          let w = select(1.0 / (1.0 + (p0.w / q.w) * exp(ex)), 1.0, rest[j].w < 0.0);
          dsum += w * (rr - dist) * (d / dist);
          n += 1.0;
        }
      }
    }
  }
  var np = p0.xyz + dsum / max(n, 1.0);
  let s2 = interiorSdf(np);
  if (s2.x < ri) { np += (ri - s2.x) * s2.yzw; }
  if (p0.y * np.y < 0.0) { np.y = p0.y; }
  pos[i] = vec4f(np, p0.w);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let nx = U.gdim.x;
  let ny = U.gdim.y;
  let nz = U.gdim.z;
  let hx = (nx + 1u) / 2u;
  let hy = (ny + 1u) / 2u;
  let hz = (nz + 1u) / 2u;
  let kk = gid.x;
  if (kk >= hx * hy * hz) { return; }
  let cx = (kk % hx) * 2u + (COLOR & 1u);
  let cy = ((kk / hx) % hy) * 2u + ((COLOR >> 1u) & 1u);
  let cz = (kk / (hx * hy)) * 2u + ((COLOR >> 2u) & 1u);
  if (cx >= nx || cy >= ny || cz >= nz) { return; }
  let c = cx + nx * (cy + ny * cz);
  let cnt = min(grid[c], SLOTS);
  for (var s = 0u; s < cnt; s++) {
    stabOne(grid[nCells() + c * SLOTS + s]);
  }
}
