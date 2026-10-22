// 休眠判定（每帧一次）：连续 SLEEP_FRAMES 帧离锚点不超过 0.04 rc 就进入休眠；瓶颈捕获区里的不休眠
const SLEEP_FRAMES: f32 = 10.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.gdim.w) { return; }
  let p = pos[i];
  if (p.w <= 0.0) { return; }
  var r = rest[i];
  if (r.w < 0.0) { return; }
  // 瓶颈上方的流动通道（约 3 倍捕获半径）里的颗粒团不休眠
  if (length(p.xyz) < U.drain.w * 3.0) {
    rest[i] = vec4f(p.xyz, 0.0);
    return;
  }
  if (length(p.xyz - r.xyz) < 0.04 * U.omega_rc.w) {
    r.w += 1.0;
  } else {
    r = vec4f(p.xyz, 0.0);
  }
  if (r.w >= SLEEP_FRAMES) {
    r = vec4f(p.xyz, -1.0);
    vel[i] = vec4f(0.0, 0.0, 0.0, vel[i].w);
  }
  rest[i] = r;
}
