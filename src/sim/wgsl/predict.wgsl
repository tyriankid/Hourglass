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
    rest[i] = vec4f(p.xyz, 0.0); // 质量变了：醒来
  }
  if (p.w <= 0.0) { predOut[i] = p; return; }
  if (U.ints.z == 1u && rest[i].w < 0.0) { rest[i] = vec4f(p.xyz, 0.0); } // 沙漏在动：全部醒来
  if (rest[i].w < 0.0) { predOut[i] = p; return; } // 休眠：锁定在原位
  let dt = U.g_dt.w;
  var v = vel[i].xyz;
  v += (U.g_dt.xyz + fict(p.xyz, v)) * dt;
  let sp = length(v);
  if (sp > U.lim.x) { v *= U.lim.x / sp; }
  var np = p.xyz + v * dt;
  if (p.y * np.y < 0.0) { np.y = p.y; } // 颗粒团过不了 1 mm 的瓶颈，沙只能以沙粒形式穿过孔口
  predOut[i] = vec4f(np, p.w);
  // 记下"求解前"的速度，收尾时用它限制接触修正带来的动能
  vel[i] = vec4f(v, vel[i].w);
}
