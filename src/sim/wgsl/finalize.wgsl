// 收尾：由位置变化求速度；位移极小的颗粒团休眠（消除静止沙堆的抖动）；统计最大速度
var<workgroup> wMax: atomic<u32>;
var<workgroup> wMoving: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) {
    atomicStore(&wMax, 0u);
    atomicStore(&wMoving, 0u);
  }
  workgroupBarrier();
  let i = gid.x;
  if (i < U.gdim.w) {
    let x0 = pos[i];
    if (x0.w > 0.0 && rest[i].w < 0.0) {
      vel[i] = vec4f(0.0, 0.0, 0.0, vel[i].w);
    } else if (x0.w > 0.0) {
      let dt = U.g_dt.w;
      var np = pred[i].xyz;
      if (x0.y * np.y < 0.0) { np.y = x0.y; }
      var d = np - x0.xyz;
      if (length(d) < U.fric.w * dt) { np = x0.xyz; d = vec3f(0.0); }
      var v = d / dt * U.lim.w;
      // 沙是无弹性、强耗能的：接触修正只能让颗粒停下或改变方向，不能让它比求解前更快
      // （否则整堆沙挤进收窄的瓶颈时会被"挤飞"成射流）。位置修正保留，只是不变成动能。
      // 求解前的速度：上一子步的速度 + (有效重力 + 惯性力)·dt（与 predict 一致；不在 predict 里改写 vel，
      // 因为唤醒检查要读上一子步的实际速度）
      let vo = vel[i].xyz;
      var vp = vo + (U.g_dt.xyz + fict(x0.xyz, vo)) * dt;
      let spp = length(vp);
      if (spp > U.lim.x) { vp *= U.lim.x / spp; }
      // 竖直方向完全非弹性：修正不能让颗粒获得比求解前更大的"向上"（反重力）速度——没有反弹
      let upd = -gravityDir();
      let vu = dot(v, upd);
      let vuMax = max(dot(vp, upd), 0.0) + 0.02;
      if (vu > vuMax) { v -= (vu - vuMax) * upd; }
      let vmaxC = length(vp) + 0.02;
      var sp = length(v);
      if (sp > vmaxC) {
        v *= vmaxC / sp;
        sp = vmaxC;
      }
      if (sp > U.lim.x) { v *= U.lim.x / sp; }
      pos[i] = vec4f(np, x0.w);
      vel[i] = vec4f(v, vel[i].w);
      atomicMax(&wMax, u32(min(sp, 100.0) * 10000.0));
      if (sp > 0.02) { atomicAdd(&wMoving, 1u); }
    }
  }
  workgroupBarrier();
  if (li == 0u) {
    atomicMax(&ctl[C_MAX_SPEED], atomicLoad(&wMax));
    atomicAdd(&ctl[C_MOVING], atomicLoad(&wMoving));
  }
}
