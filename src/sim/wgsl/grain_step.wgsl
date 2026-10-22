// 沙粒运动：有效重力 + 惯性力下的自由飞行；与颗粒团做扫掠碰撞（不会穿过去），与玻璃做 SDF 碰撞。
// 低恢复系数反弹几次后停下，沉积进网格。
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.ints.x) { return; }
  let g = grains[i];
  if (g.w <= 0.0) { return; }
  var x = g.xyz;
  let gv = gvel[i];
  var v = gv.xyz;
  var bounces = g.w - 1.0;
  let dt = U.g_dt.w;
  let rg = U.misc.z;
  let cs = U.gmin_cs.w;
  let nc = nCells();
  let gdir = gravityDir();
  let nsub = u32(clamp(ceil((length(v) + length(U.g_dt.xyz) * dt) * dt / (0.5 * cs)), 1.0, 8.0));
  let h = dt / f32(nsub);
  var seed = pcg(i * 7919u + atomicLoad(&ctl[C_RNG]) * 104729u);
  var alive = true;
  for (var s = 0u; s < nsub; s++) {
    v += (U.g_dt.xyz + fict(x, v)) * h;
    let d = v * h;
    let dd = dot(d, d);
    var best = 2.0;
    var hitC = vec3f(0.0);
    var hitR = 0.0;
    var hitJ = NONE;
    var hitM = 0.0;
    var nearJ = NONE;
    var nearD = U.omega_rc.w * 2.5;
    let xe = x + d;
    let cc = cellOf(x + 0.5 * d);
    for (var dz = -1; dz <= 1; dz++) {
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          let c = cellIdx(cc + vec3i(dx, dy, dz));
          if (c < 0) { continue; }
          let cnt = min(grid[u32(c)], SLOTS);
          for (var k = 0u; k < cnt; k++) {
            let j = grid[nc + u32(c) * SLOTS + k];
            let q = pos[j];
            if (q.w <= 0.0) { continue; }
            // 只和同一个泡里的颗粒团作用：孔口上方的颗粒团球体会越过瓶颈，但它在另一个泡里
            if (q.y * x.y < 0.0) { continue; }
            let R = clumpRadius(q.w) + rg;
            if (q.w < U.alpha_mc.w) {
              let dn = length(xe - q.xyz) - R;
              if (dn < nearD) { nearD = dn; nearJ = j; }
            }
            let m = x - q.xyz;
            let c0 = dot(m, m) - R * R;
            var t = 0.0;
            if (c0 > 0.0) {
              let b = dot(m, d);
              if (b >= 0.0 || dd < 1e-20) { continue; }
              let disc = b * b - dd * c0;
              if (disc < 0.0) { continue; }
              t = (-b - sqrt(disc)) / dd;
              if (t > 1.0) { continue; }
            }
            if (t < best) { best = t; hitC = q.xyz; hitR = R; hitJ = j; hitM = q.w; }
          }
        }
      }
    }
    if (best <= 1.0) {
      var xc = x + d * best;
      var nrm = xc - hitC;
      let ln = length(nrm);
      if (ln > 1e-9) { nrm /= ln; } else { nrm = -gdir; }
      xc = hitC + nrm * (hitR + 1e-6);
      let speed = length(v);
      let vn = dot(v, nrm);
      if (vn >= 0.0) {
        // 本来就嵌在球里、正在离开：只推到表面，不算一次撞击
        x = xc;
        continue;
      }
      atomicAdd(&ctl[C_IMPACT_SAND], 1u);
      atomicAdd(&ctl[C_IMPACT_SPEED], u32(speed * 1000.0));
      bounces += 1.0;
      if (bounces > 2.0 || speed < U.lim.y) {
        // 并入撞到的未满颗粒团，其次是附近最近的未满颗粒团，都没有才攒在网格里
        var tj = NONE;
        // 撞到的颗粒团不超过标称质量的 1.12 倍（半径 +3.8%）就直接并进去；否则找附近未满的
        if (hitM < U.alpha_mc.w * 1.12) { tj = hitJ; } else if (nearJ != NONE) { tj = nearJ; }
        if (tj != NONE) {
          atomicAdd(&pend[tj], 1u);
          atomicAdd(&ctl[C_DEPOSITED], 1u);
        } else {
          deposit(xc);
        }
        alive = false;
        x = xc;
        break;
      }
      let e = 0.12 + 0.25 * rnd(&seed);
      let vt = (v - vn * nrm) * 0.55;
      v = vt - e * vn * nrm;
      let r3 = vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) - 0.5;
      v += r3 * speed * 0.2;
      x = xc;
      continue;
    }
    let xn = x + d;
    let sw = interiorSdf(xn);
    if (sw.x < rg) {
      let nw = sw.yzw;
      let xp = xn + (rg - sw.x) * nw;
      let vn = dot(v, nw);
      if (vn < 0.0) {
        let speed = length(v);
        atomicAdd(&ctl[C_IMPACT_GLASS], 1u);
        bounces += 1.0;
        let support = dot(nw, -gdir);
        if ((speed < U.lim.y || bounces > 3.0) && support > 0.3) {
          if (nearJ != NONE) {
            atomicAdd(&pend[nearJ], 1u);
            atomicAdd(&ctl[C_DEPOSITED], 1u);
          } else {
            deposit(xp);
          }
          alive = false;
          x = xp;
          break;
        }
        let vt = (v - vn * nw) * 0.75;
        v = vt - 0.35 * vn * nw;
      }
      x = xp;
    } else {
      x = xn;
    }
  }
  if (alive) {
    grains[i] = vec4f(x, bounces + 1.0);
    gvel[i] = vec4f(v, gv.w);
  } else {
    grains[i] = vec4f(x, 0.0);
  }
}
