// 沙面第三步：由平滑深度重建法线并着色。颗粒级的色斑和闪光用"锚定在颗粒团上的材质坐标"生成，
// 所以沙流动时纹理跟着走，静止时纹理不动；视角变化时闪光会变，就像真沙。
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<uniform> P: Profile;
@group(0) @binding(2) var depthS: texture_2d<f32>;
@group(0) @binding(3) var idTex: texture_2d<u32>;
@group(0) @binding(4) var<storage, read> pos: array<vec4f>;
@group(0) @binding(5) var<storage, read> vel: array<vec4f>;

const GRAIN: f32 = 1.0e-4;

fn loadD(p: vec2i) -> f32 {
  let size = vec2i(textureDimensions(depthS));
  return textureLoad(depthS, clamp(p, vec2i(0), size - vec2i(1)), 0).r;
}

// 线性空间的反照率：米色为主，夹杂白色石英和深色矿物颗粒
const SAND_BASE: vec3f = vec3f(0.68, 0.52, 0.31);
fn sandTone(h: u32) -> vec3f {
  let r = u01(h);
  if (r < 0.07) { return vec3f(0.20, 0.15, 0.10); }
  if (r < 0.19) { return vec3f(0.80, 0.78, 0.72); }
  let k = u01(pcg(h)) * 0.16 - 0.08;
  return SAND_BASE * (1.0 + k);
}

@fragment
fn fs(i: FsOut) -> @location(0) vec4f {
  let p = vec2i(i.pos.xy);
  let d = loadD(p);
  if (d <= 0.0) { return vec4f(0.0); }
  let P0 = viewPosFromDepth(i.pos.xy, d);
  // 法线：两侧取深度差更小的一边，避免轮廓处被拉坏
  let dl = loadD(p - vec2i(1, 0));
  let dr = loadD(p + vec2i(1, 0));
  let du = loadD(p - vec2i(0, 1));
  let dd = loadD(p + vec2i(0, 1));
  var ex: vec3f;
  if (dr > 0.0 && (dl <= 0.0 || abs(dr - d) < abs(dl - d))) {
    ex = viewPosFromDepth(i.pos.xy + vec2f(1.0, 0.0), dr) - P0;
  } else {
    ex = P0 - viewPosFromDepth(i.pos.xy - vec2f(1.0, 0.0), select(d, dl, dl > 0.0));
  }
  var ey: vec3f;
  if (dd > 0.0 && (du <= 0.0 || abs(dd - d) < abs(du - d))) {
    ey = viewPosFromDepth(i.pos.xy + vec2f(0.0, 1.0), dd) - P0;
  } else {
    ey = P0 - viewPosFromDepth(i.pos.xy - vec2f(0.0, 1.0), select(d, du, du > 0.0));
  }
  var n = normalize(cross(ey, ex));
  if (dot(n, -P0) < 0.0) { n = -n; }

  // 材质坐标：局部系位置减去所属颗粒团的中心，再加每个颗粒团各自的随机偏移
  let id = textureLoad(idTex, p, 0).r;
  let wPos = (V.viewInv * vec4f(P0, 1.0)).xyz;
  let lPos = (V.modelInv * vec4f(wPos, 1.0)).xyz;
  let c = pos[id];
  let hid = pcg(id * 2654435761u + 17u);
  let mcoord = lPos - c.xyz + vec3f(u01(hid), u01(pcg(hid)), u01(pcg(hid + 1u))) * 0.05;

  // 像素足迹内能看到多少粒沙：足迹大于沙粒时，一个格子代表多粒沙，色斑对比按统计平均减弱
  let foot = V.sim.w * d;
  let L = max(GRAIN, foot);
  let cell = vec3i(floor(mcoord / L));
  let hc = hash31(cell);
  let nGrains = max(foot / GRAIN, 1.0);
  let tone = mix(SAND_BASE, sandTone(hc), clamp(1.4 / nGrains, 0.12, 1.0));

  // 光照：主光（包裹漫反射）+ 摄影棚环境 + 屏幕空间环境光遮蔽
  let Lv = normalize(worldToViewDir(V.light.xyz));
  let ndl = dot(n, Lv);
  let wrap = max((ndl + 0.35) / 1.35, 0.0);
  let rpx = V.sim.x / (V.sim.w * d);
  var occ = 0.0;
  for (var k = 0; k < 8; k++) {
    let a = f32(k) * 0.785398 + 0.3;
    let off = vec2i(vec2f(cos(a), sin(a)) * rpx * (1.5 + f32(k % 3)));
    let dn = loadD(p + off);
    if (dn > 0.0) { occ += clamp((d - dn) / (V.sim.x * 3.0), 0.0, 1.0); }
  }
  let ao = 1.0 - occ / 8.0 * 0.8;
  let nW = viewToWorldDir(n);
  let sky = 0.5 + 0.3 * nW.y;
  var col = tone * (sky * ao * 0.6 + wrap * V.light.w * 0.75);

  // 闪光：每个格子有一定概率含一粒朝向恰好把主光反射进相机的石英面
  let pGlint = 1.0 - pow(1.0 - 0.008, nGrains * nGrains);
  if (u01(pcg(hc ^ 0x9e3779b9u)) < pGlint) {
    let h1 = pcg(hc + 3u);
    let fn_ = normalize(vec3f(u01(h1), u01(pcg(h1)), u01(pcg(h1 + 7u))) * 2.0 - 1.0 + n * 1.3);
    let H = normalize(Lv + normalize(-P0));
    let sp = pow(max(dot(fn_, H), 0.0), 120.0);
    col += vec3f(1.0, 0.97, 0.9) * sp * 10.0 * V.light.w;
  }
  // 刚凝聚出来的颗粒团带一层渐渐褪去的金色辉光
  let born = vel[id].w;
  if (born > 0.0) {
    let age = V.misc.x - born;
    if (age >= 0.0 && age < 3.0) { col += vec3f(1.6, 1.1, 0.45) * exp(-age / 0.55); }
  }
  return vec4f(col, 1.0);
}
