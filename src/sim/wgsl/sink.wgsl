// 孔口边界条件（单线程）：按 Beverloo 流量从捕获区抽走沙，离孔口最近的颗粒团先缩小、先消失。
// 抽走量不超过捕获区里实际有的沙。抽走的沙全部变成下方的沙粒，质量严格守恒。
@compute @workgroup_size(1)
fn main() {
  let n = min(atomicLoad(&ctl[C_CAP_COUNT]), CAP_MAX);
  var idx: array<u32, 64>;
  var dst: array<f32, 64>;
  for (var k = 0u; k < n; k++) {
    idx[k] = atomicLoad(&ctl[C_CAP_LIST + 2u * k]);
    dst[k] = bitcast<f32>(atomicLoad(&ctl[C_CAP_LIST + 2u * k + 1u]));
  }
  for (var a = 1u; a < n; a++) {
    let ki = idx[a];
    let kd = dst[a];
    var b = a;
    while (b > 0u && dst[b - 1u] > kd) {
      idx[b] = idx[b - 1u];
      dst[b] = dst[b - 1u];
      b--;
    }
    idx[b] = ki;
    dst[b] = kd;
  }
  var avail = 0.0;
  for (var k = 0u; k < n; k++) { avail += max(pos[idx[k]].w, 0.0); }

  let goal = U.drain.x;
  let mc = U.alpha_mc.w;
  var drained = 0.0;
  var top = atomicLoad(&ctl[C_FREE_TOP]);
  for (var k = 0u; k < n; k++) {
    if (drained >= goal) { break; }
    let j = idx[k];
    var m = pos[j].w;
    if (m <= 0.0) { continue; }
    let take = min(m, goal - drained);
    m -= take;
    drained += take;
    if (m <= mc * 1e-4) { drained += m; m = 0.0; }
    if (m <= 0.0) {
      pos[j] = vec4f(0.0, 1000.0, 0.0, 0.0);
      pred[j] = vec4f(0.0, 1000.0, 0.0, 0.0);
      freeStack[top] = j;
      top++;
    } else {
      pos[j].w = m;
      pred[j].w = m;
    }
  }
  atomicStore(&ctl[C_FREE_TOP], top);

  let df = bitcast<f32>(atomicLoad(&ctl[C_DRAINED])) + drained;
  atomicStore(&ctl[C_DRAINED], bitcast<u32>(df));
  atomicStore(&ctl[C_AVAIL], bitcast<u32>(avail));
  if (drained > 0.0) { atomicStore(&ctl[C_EMIT_DIR], bitcast<u32>(U.drain.y)); }

  var budget = bitcast<f32>(atomicLoad(&ctl[C_EMIT_BUDGET])) + drained / U.drain.z;
  let cnt = u32(min(floor(budget), 1024.0));
  budget -= f32(cnt);
  atomicStore(&ctl[C_EMIT_BUDGET], bitcast<u32>(budget));
  let head = atomicLoad(&ctl[C_GRAIN_HEAD]);
  atomicStore(&ctl[C_EMIT_COUNT], cnt);
  atomicStore(&ctl[C_EMIT_BASE], head);
  atomicStore(&ctl[C_GRAIN_HEAD], (head + cnt) % U.ints.x);
  atomicAdd(&ctl[C_EMITTED], cnt);
  atomicStore(&ctl[C_CAP_COUNT], 0u);
  atomicAdd(&ctl[C_RNG], 1u);
}
