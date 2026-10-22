// 收尾：由位置变化求速度；位移极小的颗粒团休眠（消除静止沙堆的抖动）；统计最大速度
var<workgroup> wMax: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) { atomicStore(&wMax, 0u); }
  workgroupBarrier();
  let i = gid.x;
  if (i < U.gdim.w) {
    let x0 = pos[i];
    if (x0.w > 0.0) {
      let dt = U.g_dt.w;
      var np = pred[i].xyz;
      if (x0.y * np.y < 0.0) { np.y = x0.y; }
      var d = np - x0.xyz;
      if (length(d) < U.fric.w * dt) { np = x0.xyz; d = vec3f(0.0); }
      var v = d / dt;
      let sp = length(v);
      if (sp > U.lim.x) { v *= U.lim.x / sp; }
      pos[i] = vec4f(np, x0.w);
      vel[i] = vec4f(v, vel[i].w);
      atomicMax(&wMax, u32(min(sp, 100.0) * 10000.0));
    }
  }
  workgroupBarrier();
  if (li == 0u) { atomicMax(&ctl[C_MAX_SPEED], atomicLoad(&wMax)); }
}
