// 金色微光粒子（纯视觉，不参与物理）：pos.w = 剩余寿命，vel.w = 初始寿命
fn emitFx(p: vec3f, v: vec3f, life: f32) {
  let cap = u32(O.r.y);
  let k = atomicAdd(&ctl[C_FX_HEAD], 1u) % cap;
  fxPos[k] = vec4f(p, life);
  fxVel[k] = vec4f(v, life);
}
