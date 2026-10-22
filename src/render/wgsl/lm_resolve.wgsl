// 光照贴图收尾：光子计数 ÷ 无遮挡时的期望值 = 主光系数（< 1 阴影，> 1 焦散）；3×3 高斯去噪；端盖接触处的环境光遮蔽

struct PhotonU {
  model: mat4x4f,
  modelInv: mat4x4f,
  L: vec4f,
  pu: vec4f,
  pv: vec4f,
  pc: vec4f,
  cfg: vec4u,
  lm: vec4f,     // 贴图半边长，端盖厚度，端盖半径，端盖中心离瓶颈的距离
}

@group(0) @binding(0) var<uniform> Q: PhotonU;
@group(0) @binding(1) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(2) var lmOut: texture_storage_2d<rgba16float, write>;

override CLEAR: bool = false;

const ENERGY: f32 = 256.0;

fn cnt(x: i32, y: i32) -> f32 {
  let res = i32(Q.cfg.w);
  let cx = clamp(x, 0, res - 1);
  let cy = clamp(y, 0, res - 1);
  return f32(atomicLoad(&acc[u32(cy * res + cx)]));
}

// 点到实心圆饼（端盖）的距离，局部系
fn capDist(p: vec3f, cy: f32) -> f32 {
  let q = vec2f(length(p.xz) - Q.lm.z, abs(p.y - cy) - 0.5 * Q.lm.y);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let res = Q.cfg.w;
  if (gid.x >= res || gid.y >= res) { return; }
  if (CLEAR) {
    atomicStore(&acc[gid.y * res + gid.x], 0u);
    return;
  }
  let x = i32(gid.x);
  let y = i32(gid.y);
  var sum = 4.0 * cnt(x, y);
  sum += 2.0 * (cnt(x - 1, y) + cnt(x + 1, y) + cnt(x, y - 1) + cnt(x, y + 1));
  sum += cnt(x - 1, y - 1) + cnt(x + 1, y - 1) + cnt(x - 1, y + 1) + cnt(x + 1, y + 1);
  sum /= 16.0;
  let half = Q.lm.x;
  let texel = 2.0 * half / f32(res);
  let delta = 2.0 * Q.pc.w / f32(Q.cfg.x);
  let Lw = normalize(Q.L.xyz);
  let expected = texel * texel * abs(Lw.y) / (delta * delta) * ENERGY;
  var ratio = sum / expected;
  let pT = vec3f((f32(x) + 0.5) * texel - half, 0.0, (f32(y) + 0.5) * texel - half);
  // 光子平面没覆盖到的地方没有遮挡，主光系数为 1
  let t = dot(Q.pc.xyz - pT, Lw);
  let q = pT + Lw * t - Q.pc.xyz;
  if (max(abs(dot(q, Q.pu.xyz)), abs(dot(q, Q.pv.xyz))) > Q.pc.w * 0.98) { ratio = 1.0; }
  // 端盖贴近桌面的地方有接触阴影
  let pl = (Q.modelInv * vec4f(pT, 1.0)).xyz;
  let hb = Q.lm.w; // 端盖中心离瓶颈的距离
  let d = min(capDist(pl, hb), capDist(pl, -hb));
  let ao = 0.5 * exp(-max(d, 0.0) / 0.005);
  textureStore(lmOut, vec2i(x, y), vec4f(ratio, 0.0, ao, 0.0));
}
