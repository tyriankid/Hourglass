// 预测位置：v += (有效重力 + 惯性力)·dt，x* = x + v·dt
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.gdim.w) { return; }
  var p = pos[i];
  let add = atomicExchange(&pend[i], 0u);
  if (add > 0u && p.w > 0.0) {
    p.w += f32(add) * U.drain.z;
    pos[i].w = p.w;
  }
  if (p.w <= 0.0) { predOut[i] = p; return; }
  let dt = U.g_dt.w;
  var v = vel[i].xyz;
  v += (U.g_dt.xyz + fict(p.xyz, v)) * dt;
  let sp = length(v);
  if (sp > U.lim.x) { v *= U.lim.x / sp; }
  var np = p.xyz + v * dt;
  if (p.y * np.y < 0.0) { np.y = p.y; } // 颗粒团过不了 1 mm 的瓶颈，沙只能以沙粒形式穿过孔口
  predOut[i] = vec4f(np, p.w);
}
