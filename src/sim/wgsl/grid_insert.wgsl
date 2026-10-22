// 把颗粒团放进均匀网格（每格最多 8 个），顺便收集孔口捕获区里的颗粒团
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.gdim.w) { return; }
  let p = pred[i];
  if (p.w <= 0.0) { return; }
  let c = cellIdx(cellOf(p.xyz));
  if (c < 0) { return; }
  let slot = atomicAdd(&grid[u32(c)], 1u);
  if (slot < SLOTS) { atomicStore(&grid[nCells() + u32(c) * SLOTS + slot], i); } else { atomicAdd(&ctl[C_GRID_OVERFLOW], 1u); }
  let up = U.drain.y;
  if (up != 0.0 && p.y * up > 0.0) {
    let d = length(p.xyz);
    if (d < U.drain.w) {
      let k = atomicAdd(&ctl[C_CAP_COUNT], 1u);
      if (k < CAP_MAX) {
        atomicStore(&ctl[C_CAP_LIST + 2u * k], i);
        atomicStore(&ctl[C_CAP_LIST + 2u * k + 1u], bitcast<u32>(d));
      } else {
        atomicAdd(&ctl[C_CAP_OVERFLOW], 1u);
      }
    }
  }
}
