// 消散：去掉一个泡里高度超过阈值的颗粒团，可放出上升的微光粒子
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.gdim.w) { return; }
  let p = pos[i];
  if (p.w <= 0.0 || p.y * O.a.z <= 0.0) { return; }
  if (dot(p.xyz, O.up.xyz) <= O.a.x) { return; }
  atomicAdd(&ctl[C_DISSOLVED], u32(round(p.w * 1000.0)));
  pos[i] = vec4f(0.0, 1000.0, 0.0, 0.0);
  vel[i] = vec4f(0.0);
  let k = atomicAdd(&ctl[C_FREE_TOP], 1u);
  freeStack[k] = i;
  if (O.b.z > 0.0) {
    var seed = pcg(i * 31337u + u32(O.b.x));
    for (var f = 0; f < 2; f++) {
      let dir = normalize(vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) - 0.5);
      emitFx(p.xyz + dir * U.omega_rc.w, dir * 0.006 + O.up.xyz * 0.02, 0.6 + 0.6 * rnd(&seed));
    }
  }
}
