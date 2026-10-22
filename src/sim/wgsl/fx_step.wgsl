// 微光粒子：缓慢漂移、减速、寿命递减
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(O.r.y)) { return; }
  var p = fxPos[i];
  if (p.w <= 0.0) { return; }
  var v = fxVel[i];
  let dt = O.b.y;
  v = vec4f(v.xyz * exp(-dt * 1.5), v.w);
  p = vec4f(p.xyz + v.xyz * dt, p.w - dt);
  fxPos[i] = p;
  fxVel[i] = v;
}
