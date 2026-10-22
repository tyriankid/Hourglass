// 沙面第二步：窄范围双边滤波（可分离，两遍），把一颗颗球面平滑成连续沙面
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<uniform> P: Profile;
@group(0) @binding(2) var src: texture_2d<f32>;

override DIR_X: f32 = 1.0;
override DIR_Y: f32 = 0.0;

@fragment
fn fs(i: FsOut) -> @location(0) f32 {
  let p = vec2i(i.pos.xy);
  let d0 = textureLoad(src, p, 0).r;
  if (d0 <= 0.0) { return 0.0; }
  let size = vec2i(textureDimensions(src));
  let rpx = V.sim.x / (V.sim.w * d0);
  let R = i32(clamp(rpx * 2.0, 1.0, 16.0));
  let sigma = f32(R) * 0.5 + 0.5;
  let range = V.sim.x * 2.0;
  let dir = vec2i(i32(DIR_X), i32(DIR_Y));
  var sum = 0.0;
  var wsum = 0.0;
  for (var k = -16; k <= 16; k++) {
    if (abs(k) > R) { continue; }
    let q = clamp(p + dir * k, vec2i(0), size - vec2i(1));
    let d = textureLoad(src, q, 0).r;
    if (d <= 0.0) { continue; }
    let w = exp(-f32(k * k) / (2.0 * sigma * sigma));
    sum += clamp(d, d0 - range, d0 + range) * w;
    wsum += w;
  }
  return sum / max(wsum, 1e-6);
}
