// 合并（二）：接收者自己不往外送时，才把质量（连同还没并入的沉积）转给它，然后释放自己
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.gdim.w) { return; }
  let j = recv[i];
  if (j == NONE || recv[j] != NONE) { return; }
  let p = pos[i];
  if (p.w <= 0.0) { return; }
  let own = u32(round(p.w / U.drain.z));
  // 用接收者的 pend 做容量预留：并入后超过 1.05 倍标称质量就退回，不让颗粒团长得过大
  let room = u32(1.05 * U.alpha_mc.w / U.drain.z);
  let mj = u32(pos[j].w / U.drain.z);
  let before = atomicAdd(&pend[j], own);
  if (mj + before + own > room) {
    atomicSub(&pend[j], own);
    return;
  }
  atomicAdd(&pend[j], atomicExchange(&pend[i], 0u));
  pos[i] = vec4f(0.0, 1000.0, 0.0, 0.0);
  let k = atomicAdd(&ctl[C_FREE_TOP], 1u);
  freeStack[k] = i;
  atomicAdd(&ctl[C_MERGED], 1u);
}
