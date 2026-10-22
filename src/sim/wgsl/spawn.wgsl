// 沉积生长：本格累加的沙先并入本格里没长满的颗粒团，剩下的从空闲池取一个新颗粒团，
// 从极小开始长。每格只由一个线程处理，本格的颗粒团不会被别的线程同时修改。
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= nCells()) { return; }
  let b = c * 4u;
  let n = atomicLoad(&dep[b]);
  if (n == 0u) { return; }
  if (f32(n) * U.drain.z < 0.03 * U.alpha_mc.w && U.lim.z == 0.0) { return; }
  atomicStore(&dep[b], 0u);
  let sx = atomicExchange(&dep[b + 1u], 0u);
  let sy = atomicExchange(&dep[b + 2u], 0u);
  let sz = atomicExchange(&dep[b + 3u], 0u);
  let nx = U.gdim.x;
  let ny = U.gdim.y;
  let cc = vec3u(c % nx, (c / nx) % ny, c / (nx * ny));
  let frac = vec3f(f32(sx), f32(sy), f32(sz)) / (255.0 * f32(n));
  let centroid = U.gmin_cs.xyz + (vec3f(cc) + frac) * U.gmin_cs.w;
  let mc = U.alpha_mc.w;
  var rem = f32(n) * U.drain.z;
  let cnt = min(grid[c], SLOTS);
  for (var k = 0u; k < cnt; k++) {
    if (rem <= 0.0) { break; }
    let j = grid[nCells() + c * SLOTS + k];
    let q = pos[j];
    if (q.w > 0.0 && q.w < mc && cellIdx(cellOf(q.xyz)) == i32(c)) {
      let add = min(rem, mc - q.w);
      pos[j].w = q.w + add;
      rest[j] = vec4f(q.xyz, 0.0);
      rem -= add;
    }
  }
  for (var guard = 0; guard < 4 && rem > 0.0; guard++) {
    let t = atomicSub(&ctl[C_FREE_TOP], 1u);
    if (t == 0u || t > U.gdim.w) {
      atomicAdd(&ctl[C_FREE_TOP], 1u);
      atomicAdd(&ctl[C_SPAWN_FAIL], 1u);
      break;
    }
    let j = freeStack[t - 1u];
    let m = min(rem, mc);
    pos[j] = vec4f(centroid, m);
    vel[j] = vec4f(0.0);
    rest[j] = vec4f(centroid, 0.0);
    rem -= m;
  }
}
