// 休眠判定（每帧一次）：连续 SLEEP_FRAMES 帧离锚点不超过 0.04 rc，就进入休眠（静摩擦锁住）。
// （试过放宽到 0.5 mm：大片带着残余重叠被锁住、再成片唤醒，反而激起整体起伏振荡）
// 例外：流动时上游泡里瓶颈正上方的漏斗通道（向上逐渐变宽）始终醒着——真实的漏斗流就是这一柱在流，
// 若让它休眠，上方的沙会被拱住、断供。
const SLEEP_FRAMES: f32 = 10.0;
const SLEEP_DIST_RC: f32 = 0.04;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= U.gdim.w) { return; }
  let p = pos[i];
  if (p.w <= 0.0) { return; }
  var r = rest[i];
  if (r.w < 0.0) { return; }
  let rc = U.omega_rc.w;
  let up = U.drain.y;
  if (up != 0.0 && p.y * up > 0.0) {
    let h = abs(p.y);
    if (length(p.xz) < 4.0 * rc + 0.25 * h || length(p.xyz) < U.drain.w * 3.0) {
      rest[i] = vec4f(p.xyz, 0.0);
      return;
    }
  }
  if (length(p.xyz - r.xyz) < SLEEP_DIST_RC * rc) {
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
