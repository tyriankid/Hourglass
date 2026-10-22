// 凝聚：在泡里指定高度区间的空腔中，按点阵生成颗粒团（每个线程一个互不相同的格点，同批之间不重叠），
// 生成前查网格，避开已有的颗粒团；可带"刚凝聚"的发光标记，同时放出微光粒子。
// O.a = (y0, y1, 泡, 数量)；O.b = (随机种子, 时间, 是否发光, 质量)；O.r = (点阵起点, 微光容量, 点阵格数, 步长)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= u32(O.a.w)) { return; }
  let rc = U.omega_rc.w;
  let sp = 2.3 * rc;
  let ylo = min(O.a.x, O.a.y);
  let yhi = max(O.a.x, O.a.y);
  let rmax = P.s[129].z;
  let nxl = u32(ceil(2.0 * rmax / sp));
  let nyl = max(1u, u32(ceil((yhi - ylo) / sp)));
  let total = u32(O.r.z);
  let cell = (u32(O.r.x) + k * u32(O.r.w)) % total;
  let ix = cell % nxl;
  let iz = (cell / nxl) % nxl;
  let iy = cell / (nxl * nxl);
  if (iy >= nyl) { return; }
  var seed = pcg(cell * 7919u + u32(O.b.x));
  let jit = (vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) - 0.5) * 0.1 * rc;
  let p = vec3f(-rmax + (f32(ix) + 0.5) * sp, ylo + (f32(iy) + 0.5) * sp, -rmax + (f32(iz) + 0.5) * sp) + jit;
  let pr = profileAt(abs(p.y) / bulbH());
  if (length(p.xz) > pr.x - rc * 1.3 || abs(p.y) > bulbH() - rc * 1.2) { return; }
  // 避开已有的颗粒团
  let cc = cellOf(p);
  let nc = nCells();
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let c = cellIdx(cc + vec3i(dx, dy, dz));
        if (c < 0) { continue; }
        let cnt = min(grid[u32(c)], SLOTS);
        for (var s = 0u; s < cnt; s++) {
          let q = pos[grid[nc + u32(c) * SLOTS + s]];
          if (q.w > 0.0 && length(q.xyz - p) < (rc + clumpRadius(q.w)) * 1.15) { return; }
        }
      }
    }
  }
  let t = atomicSub(&ctl[C_FREE_TOP], 1u);
  if (t == 0u || t > U.gdim.w) {
    atomicAdd(&ctl[C_FREE_TOP], 1u);
    atomicAdd(&ctl[C_SPAWN_FAIL], 1u);
    return;
  }
  let j = freeStack[t - 1u];
  pos[j] = vec4f(p, O.b.w);
  rest[j] = vec4f(p, 0.0);
  vel[j] = vec4f(0.0, 0.0, 0.0, select(0.0, O.b.y, O.b.z > 0.0));
  atomicAdd(&ctl[C_CONDENSED], u32(round(O.b.w * 1000.0)));
  if (O.b.z > 0.0) {
    let dir = normalize(vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) - 0.5);
    emitFx(p + dir * rc, dir * 0.008 + O.up.xyz * 0.01, 0.5 + 0.5 * rnd(&seed));
  }
}
