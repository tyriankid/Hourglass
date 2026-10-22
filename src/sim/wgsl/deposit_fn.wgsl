// 沙粒停下后把自己沉积进所在网格的累加器（数量 + 格内位置和，用来求质心）
fn deposit(x: vec3f) {
  let cc = clamp(cellOf(x), vec3i(0), vec3i(U.gdim.xyz) - vec3i(1));
  let c = u32(cellIdx(cc));
  let o = (x - (U.gmin_cs.xyz + vec3f(cc) * U.gmin_cs.w)) / U.gmin_cs.w;
  let q = vec3u(clamp(o, vec3f(0.0), vec3f(1.0)) * 255.0);
  let b = c * 4u;
  atomicAdd(&dep[b], 1u);
  atomicAdd(&dep[b + 1u], q.x);
  atomicAdd(&dep[b + 2u], q.y);
  atomicAdd(&dep[b + 3u], q.z);
  atomicAdd(&ctl[C_DEPOSITED], 1u);
}
