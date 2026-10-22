// 场景：无缝摄影棚背景 + 桌面（全屏解析求交），金属端盖（网格）。输出 HDR 颜色和深度。

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<uniform> P: Profile;
@group(0) @binding(2) var lightMap: texture_2d<f32>;
@group(0) @binding(3) var linSamp: sampler;

struct FsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> FsOut {
  let p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0))[vi];
  var o: FsOut;
  o.pos = vec4f(p, 0.0, 1.0);
  o.uv = vec2f((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
  return o;
}

struct SceneOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

const LIGHTMAP_HALF: f32 = 0.22; // 光照贴图覆盖桌面 ±22 cm：r 主光系数（< 1 阴影、> 1 焦散），b 接触阴影

const CAP_T: f32 = 0.007;

fn capDist(p: vec3f, cy: f32) -> f32 {
  let q = vec2f(length(p.xz) - P.s[129].w, abs(p.y - cy) - 0.5 * CAP_T);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

fn backdrop(d: vec3f) -> vec3f {
  let t = smoothstep(-0.1, 0.6, d.y);
  return mix(vec3f(0.80, 0.785, 0.765), vec3f(0.90, 0.89, 0.875), t);
}

@fragment
fn fsBackdrop(i: FsOut) -> SceneOut {
  let px = i.pos.xy;
  let vp = viewPosFromDepth(px, 1.0);
  let dirW = normalize(viewToWorldDir(vp));
  let o = V.camPos.xyz;
  var out: SceneOut;
  out.depth = 1.0;
  var col = backdrop(dirW);
  if (dirW.y < -1e-4) {
    let t = -o.y / dirW.y;
    let p = o + dirW * t;
    let dist = length(p.xz);
    // 桌面：环境光 + 主光，阴影和焦散来自光照贴图
    let lmUv = p.xz / (2.0 * LIGHTMAP_HALF) + 0.5;
    var lm = vec4f(1.0, 0.0, 0.0, 0.0);
    if (all(lmUv > vec2f(0.0)) && all(lmUv < vec2f(1.0))) { lm = textureSampleLevel(lightMap, linSamp, lmUv, 0.0); }
    let albedo = vec3f(0.80, 0.785, 0.765);
    let L = normalize(V.light.xyz);
    let direct = max(L.y, 0.0) * V.light.w * lm.x;
    // 端盖贴近桌面处的接触阴影：按当前姿态解析计算，不随光照贴图滞后
    let pl = (V.modelInv * vec4f(p, 1.0)).xyz;
    let hb = bulbH() + CAP_T * 0.5;
    let dc = min(capDist(pl, hb), capDist(pl, -hb));
    let ao = 0.5 * exp(-max(dc, 0.0) / 0.005);
    let ambient = 0.55 * (1.0 - ao);
    var tcol = albedo * (ambient + direct * 0.45);
    // 远处与背景融为一体（无缝摄影棚）
    col = mix(tcol, backdrop(dirW), smoothstep(0.35, 1.6, dist));
    let clip = V.proj * V.view * vec4f(p, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  }
  out.color = vec4f(col, 1.0);
  return out;
}

struct MeshOut {
  @builtin(position) pos: vec4f,
  @location(0) nW: vec3f,
  @location(1) pW: vec3f,
  @location(2) pL: vec3f,
}

@vertex
fn vsCap(@location(0) p: vec3f, @location(1) n: vec3f) -> MeshOut {
  var o: MeshOut;
  let w = V.model * vec4f(p, 1.0);
  o.pos = V.proj * V.view * w;
  o.pW = w.xyz;
  o.nW = (V.model * vec4f(n, 0.0)).xyz;
  o.pL = p;
  return o;
}

// 拉丝铝：沿圆周方向的细纹让高光拉长；边缘倒角提亮
@fragment
fn fsCap(i: MeshOut) -> @location(0) vec4f {
  var n = normalize(i.nW);
  let vdir = normalize(V.camPos.xyz - i.pW);
  let rho = length(i.pL.xz);
  let ang = atan2(i.pL.z, i.pL.x);
  let streak = u01(hash31(vec3i(i32(rho * 40000.0), 0, 7))) - 0.5;
  let tangentL = vec3f(-sin(ang), 0.0, cos(ang));
  let tangentW = normalize((V.model * vec4f(tangentL, 0.0)).xyz);
  n = normalize(n + tangentW * streak * 0.03);
  let r = reflect(-vdir, n);
  let bin = normalize(cross(n, tangentW));
  // 各向异性：沿切向多采几次环境
  var spec = vec3f(0.0);
  for (var k = -3; k <= 3; k++) {
    let rr = normalize(r + tangentW * f32(k) * 0.07 + bin * f32(k % 2) * 0.01);
    spec += env(rr);
  }
  spec /= 7.0;
  let f = 0.6 + 0.4 * pow(1.0 - max(dot(n, vdir), 0.0), 5.0);
  let base = vec3f(0.72, 0.73, 0.75);
  let L = normalize(V.light.xyz);
  let diff = max(dot(n, L), 0.0) * 0.15;
  return vec4f(base * spec * f + base * diff, 1.0);
}
