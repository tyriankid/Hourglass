// 孔口下方发射沙粒：均匀分布在孔口圆盘内，出口速度约 √(gD)，带轻微发散和随机扰动
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= atomicLoad(&ctl[C_EMIT_COUNT])) { return; }
  let cap = U.ints.x;
  let slot = (atomicLoad(&ctl[C_EMIT_BASE]) + k) % cap;
  let old = grains[slot];
  if (old.w > 0.0) {
    deposit(old.xyz);
    atomicAdd(&ctl[C_FORCED], 1u);
  }
  var seed = pcg(slot * 9781u + atomicLoad(&ctl[C_RNG]) * 6271u + k);
  let dir = bitcast<f32>(atomicLoad(&ctl[C_EMIT_DIR]));
  let rg = U.misc.z;
  let R = max(neckR() - rg, rg);
  let ang = 6.2831853 * rnd(&seed);
  let rad = R * sqrt(rnd(&seed));
  let off = vec3f(cos(ang) * rad, 0.0, sin(ang) * rad);
  let v0 = U.misc.y * (0.85 + 0.3 * rnd(&seed));
  var v = vec3f(0.0, -dir * v0, 0.0);
  v += off / R * v0 * 0.12;
  v += (vec3f(rnd(&seed), rnd(&seed), rnd(&seed)) - 0.5) * v0 * 0.06;
  grains[slot] = vec4f(off.x, -dir * rg * 2.0, off.z, 1.0);
  gvel[slot] = vec4f(v, rnd(&seed));
}
