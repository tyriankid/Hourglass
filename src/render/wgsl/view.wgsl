// 渲染共用：视图常量、坐标变换、摄影棚环境光、玻璃轮廓查询。

struct View {
  view: mat4x4f,
  proj: mat4x4f,
  viewInv: mat4x4f,
  projInv: mat4x4f,
  model: mat4x4f,     // 沙漏局部系 → 世界
  modelInv: mat4x4f,
  camPos: vec4f,      // 世界坐标
  screen: vec4f,      // 宽, 高, 1/宽, 1/高
  sim: vec4f,         // rc, mc, 沙粒半径, 单位深度上一个像素的世界尺寸
  light: vec4f,       // 指向主光的世界方向, 强度
  misc: vec4f,        // 时间, 帧号, 曝光, 0
}

struct Profile { s: array<vec4f, 130> }

const PI: f32 = 3.14159265;
const IOR: f32 = 1.5;

fn profileAt(u: f32) -> vec4f {
  let x = clamp(u, 0.0, 1.0) * 128.0;
  let i = min(u32(x), 127u);
  return mix(P.s[i], P.s[i + 1u], x - f32(i));
}
fn bulbH() -> f32 { return P.s[129].x; }

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hash31(p: vec3i) -> u32 {
  return pcg((u32(p.x) * 73856093u) ^ pcg((u32(p.y) * 19349663u) ^ pcg(u32(p.z) * 83492791u)));
}
fn u01(h: u32) -> f32 { return f32(h) * (1.0 / 4294967296.0); }

// 像素坐标 + 视空间距离（沿 -z 的深度）→ 视空间位置
fn viewPosFromDepth(px: vec2f, depth: f32) -> vec3f {
  let ndc = vec2f(px.x * V.screen.z * 2.0 - 1.0, 1.0 - px.y * V.screen.w * 2.0);
  let p = V.projInv * vec4f(ndc, 0.5, 1.0);
  let dir = p.xyz / p.w;
  return dir * (depth / -dir.z);
}

// 视空间位置 → 像素坐标
fn project(p: vec3f) -> vec2f {
  let c = V.proj * vec4f(p, 1.0);
  let ndc = c.xy / c.w;
  return vec2f((ndc.x + 1.0) * 0.5 * V.screen.x, (1.0 - ndc.y) * 0.5 * V.screen.y);
}

fn viewToWorldDir(d: vec3f) -> vec3f { return (V.viewInv * vec4f(d, 0.0)).xyz; }
fn worldToViewDir(d: vec3f) -> vec3f { return (V.view * vec4f(d, 0.0)).xyz; }

fn rectLight(d: vec3f, center: vec3f, right: vec3f, up: vec3f, halfW: f32, halfH: f32, soft: f32) -> f32 {
  let c = dot(d, center);
  if (c <= 0.0) { return 0.0; }
  let x = dot(d, right) / c;
  let y = dot(d, up) / c;
  let fx = 1.0 - smoothstep(halfW - soft, halfW + soft, abs(x));
  let fy = 1.0 - smoothstep(halfH - soft, halfH + soft, abs(y));
  return fx * fy;
}

// 摄影棚环境：明亮的顶部大柔光箱、左右两条竖长条灯（玻璃上的竖条高光就来自它们）、暖灰无缝背景
fn env(d: vec3f) -> vec3f {
  let up = d.y;
  var c = mix(vec3f(0.30, 0.295, 0.29), vec3f(0.86, 0.85, 0.83), smoothstep(-0.25, 0.35, up));
  c = mix(c, vec3f(1.05, 1.04, 1.02), smoothstep(0.4, 0.95, up));
  // 顶部偏左前的大柔光箱
  let sb = normalize(vec3f(-0.35, 0.9, 0.25));
  c += vec3f(5.5, 5.4, 5.2) * rectLight(d, sb, normalize(vec3f(1.0, 0.39, 0.0)), normalize(cross(sb, vec3f(1.0, 0.39, 0.0))), 0.55, 0.42, 0.08);
  // 左右两条竖长条灯，略偏后
  let sl = normalize(vec3f(-0.93, 0.05, -0.36));
  let sr = normalize(vec3f(0.95, 0.05, -0.3));
  c += vec3f(3.6, 3.6, 3.7) * rectLight(d, sl, normalize(cross(vec3f(0.0, 1.0, 0.0), sl)), vec3f(0.0, 1.0, 0.0), 0.07, 0.9, 0.03);
  c += vec3f(2.4, 2.4, 2.5) * rectLight(d, sr, normalize(cross(vec3f(0.0, 1.0, 0.0), sr)), vec3f(0.0, 1.0, 0.0), 0.05, 0.9, 0.025);
  return c;
}

fn fresnel(cosT: f32) -> f32 {
  let f0 = 0.04;
  return f0 + (1.0 - f0) * pow(1.0 - clamp(cosT, 0.0, 1.0), 5.0);
}
