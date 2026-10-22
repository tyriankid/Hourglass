// 两个泡里颗粒团的总质量：工作组内浮点求和，再以 µg 定点原子累加（每组舍入误差 ≤ 0.5 µg）
var<workgroup> sPos: array<f32, 64>;
var<workgroup> sNeg: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  var mp = 0.0;
  var mn = 0.0;
  if (gid.x < U.gdim.w) {
    let p = pos[gid.x];
    if (p.w > 0.0) {
      if (p.y >= 0.0) { mp = p.w; } else { mn = p.w; }
    }
  }
  sPos[li] = mp;
  sNeg[li] = mn;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) {
      sPos[li] += sPos[li + s];
      sNeg[li] += sNeg[li + s];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    atomicAdd(&ctl[C_MASS_POS], u32(round(sPos[0] * 1000.0)));
    atomicAdd(&ctl[C_MASS_NEG], u32(round(sNeg[0] * 1000.0)));
  }
}
