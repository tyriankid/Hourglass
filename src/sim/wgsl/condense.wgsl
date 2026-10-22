// 凝聚：在泡里指定高度区间的空腔中随机生成颗粒团（可带"刚凝聚"的发光标记），同时放出微光粒子
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= u32(O.a.w)) { return; }
  var seed = pcg(k * 7919u + u32(O.b.x));
  let rc = U.omega_rc.w;
  var p = vec3f(0.0);
  var ok = false;
  for (var t = 0; t < 8; t++) {
    let y = mix(O.a.x, O.a.y, rnd(&seed));
    let pr = profileAt(abs(y) / bulbH());
    let rmax = pr.x - rc * 1.4;
    if (rmax <= 0.0) { continue; }
    let ang = 6.2831853 * rnd(&seed);
    let rad = rmax * sqrt(rnd(&seed));
    p = vec3f(cos(ang) * rad, y, sin(ang) * rad);
    ok = true;
    break;
  }
  if (!ok) {
    atomicAdd(&ctl[C_PLACED_FAIL], 1u);
    return;
  }
  let t = atomicSub(&ctl[C_FREE_TOP], 1u);
  if (t == 0u || t > U.gdim.w) {
    atomicAdd(&ctl[C_FREE_TOP], 1u);
    atomicAdd(&ctl[C_SPAWN_FAIL], 1u);
    return;
  }
  let j = freeStack[t - 1u];
  pos[j] = vec4f(p, O.b.w);
  vel[j] = vec4f(0.0, 0.0, 0.0, select(0.0, O.b.y, O.b.z > 0.0));
  if (O.b.z > 0.0) {
    for (var f = 0; f < 2; f++) {
      let dir = normalize(vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) - 0.5);
      emitFx(p + dir * rc * 2.0 * rnd(&seed), dir * 0.01 + O.up.xyz * 0.012, 0.5 + 0.5 * rnd(&seed));
    }
  }
}
