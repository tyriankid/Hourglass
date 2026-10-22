// 放置：把 CPU 算好的位置和质量（离开画面后补算的沙堆）写成颗粒团
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= u32(O.a.w)) { return; }
  let t = atomicSub(&ctl[C_FREE_TOP], 1u);
  if (t == 0u || t > U.gdim.w) {
    atomicAdd(&ctl[C_FREE_TOP], 1u);
    atomicAdd(&ctl[C_SPAWN_FAIL], 1u);
    return;
  }
  let j = freeStack[t - 1u];
  pos[j] = items[k];
  vel[j] = vec4f(0.0);
}
