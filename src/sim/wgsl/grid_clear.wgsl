@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= nCells()) { return; }
  atomicStore(&grid[gid.x], 0u);
}
