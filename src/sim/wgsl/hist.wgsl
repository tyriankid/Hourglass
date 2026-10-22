// 一个泡里颗粒团沿"向上"方向的高度直方图（256 格，按 µg 累加），消散时据此决定从哪个高度往上削
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.gdim.w) { return; }
  let p = pos[i];
  if (p.w <= 0.0 || p.y * O.a.z <= 0.0) { return; }
  let h = dot(p.xyz, O.up.xyz);
  let t = clamp((h - O.up.w) / max(O.r.x - O.up.w, 1e-6), 0.0, 0.999999);
  atomicAdd(&hist[u32(t * 256.0)], u32(round(p.w * 1000.0)));
}
