// Gauss-Seidel 求解（按网格着色）：空间网格按 (x, y, z) 奇偶分成 8 色，同色的格子相隔至少一格、
// 里面的颗粒不可能互相接触，可以并行；每个线程顺序处理一个格子里的颗粒，并立刻写回位置。
// 8 色依次各跑一遍 = 一轮完整的 Gauss-Seidel 迭代。休眠的颗粒团对邻居而言不可移动。
override COLOR: u32 = 0u;

fn solveOne(i: u32) {
  let p0 = pred[i];
  if (p0.w <= 0.0) { return; }
  if (rest[i].w < 0.0) { return; }
  let x0 = pos[i].xyz;
  let ri = clumpRadius(p0.w);
  let up = -gravityDir();
  let k = U.misc.x;
  let mi = p0.w * exp(-k * dot(p0.xyz, up));
  let cc = cellOf(p0.xyz);
  let nc = nCells();
  // 瓶颈附近的流动区：真实细沙在这里处于剪切、松散的流动态，有效摩擦远低于静止沙堆（μ(I) 流变）。
  // 颗粒团比真实沙粒粗得多，若沿用静摩擦会在锥壁之间成拱堵死，所以这里降低摩擦。
  let inFlow = U.drain.y != 0.0 && p0.y * U.drain.y > 0.0 && length(p0.xyz) < U.drain.w * 2.5;
  let fm = select(1.0, 0.15, inFlow);
  // 对这个颗粒团的各个接触取平均后整步修正（逐个接触完整修正在密实沙堆里会层层过冲，实测爆炸）
  let p = p0.xyz;
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
          let q = pred[j];
          if (q.w <= 0.0) { continue; }
          let d = p - q.xyz;
          let d2 = dot(d, d);
          let rr = ri + clumpRadius(q.w);
          if (d2 >= rr * rr || d2 < 1e-20) { continue; }
          let dist = sqrt(d2);
          let nrm = d / dist;
          let pen = rr - dist;
          let mj = q.w * exp(-k * dot(q.xyz, up));
          // 休眠的邻居不可移动：修正全部由自己承担
          let w = select(mj / (mi + mj), 1.0, rest[j].w < 0.0);
          dsum += w * pen * nrm;
          let dp = (p - x0) - (q.xyz - pos[j].xyz);
          let dtan = dp - dot(dp, nrm) * nrm;
          let lt = length(dtan);
          if (lt > 1e-12) {
            if (lt < U.fric.x * fm * pen) { dsum -= w * dtan; } else { dsum -= w * dtan * min(U.fric.y * fm * pen / lt, 1.0); }
          }
          n += 1.0;
        }
      }
    }
  }
  var np = p + U.misc.w * dsum / max(n, 1.0);
  // 玻璃：接触 + 摩擦，最后再硬投影一次保证不穿出
  let sw = interiorSdf(np);
  let penW = ri - sw.x;
  if (penW > 0.0) {
    let nw = sw.yzw;
    np += penW * nw;
    let dpw = np - x0;
    let dtw = dpw - dot(dpw, nw) * nw;
    let ltw = length(dtw);
    if (ltw > 1e-12) { np -= dtw * min(U.fric.z * fm * penW / ltw, 1.0); }
  }
  let s2 = interiorSdf(np);
  if (s2.x < ri) { np += (ri - s2.x) * s2.yzw; }
  if (p0.y * np.y < 0.0) { np.y = p0.y; }
  pred[i] = vec4f(np, p0.w);
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
    solveOne(grid[nCells() + c * SLOTS + s]);
  }
}
