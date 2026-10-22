// 合成：在玻璃覆盖的像素上做菲涅尔反射 + 薄壳折射，透过玻璃找沙面，再从背面折射看到背景；最后色调映射。
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<uniform> P: Profile;
@group(0) @binding(2) var sceneColor: texture_2d<f32>;
@group(0) @binding(3) var sceneDepth: texture_depth_2d;
@group(0) @binding(4) var sandColor: texture_2d<f32>;
@group(0) @binding(5) var sandDepth: texture_2d<f32>;
@group(0) @binding(6) var glassFront: texture_2d<f32>;
@group(0) @binding(7) var glassBack: texture_2d<f32>;
@group(0) @binding(8) var linSamp: sampler;

override MODE: u32 = 0u; // 调试：1 不画玻璃，2 玻璃不查沙面和背景，3 不查场景深度

fn clampPx(uv: vec2f) -> vec2i {
  return clamp(vec2i(uv), vec2i(0), vec2i(V.screen.xy) - vec2i(1));
}

fn sceneDist(ip: vec2i) -> f32 {
  if (MODE == 3u) { return 0.8; }
  let z = textureLoad(sceneDepth, ip, 0);
  let p = V.projInv * vec4f(0.0, 0.0, z, 1.0);
  return -p.z / p.w;
}

fn sampleScene(uv: vec2f) -> vec3f {
  return textureSampleLevel(sceneColor, linSamp, uv * V.screen.zw, 0.0).rgb;
}

// 沿内部光线找沙面：用屏幕空间的沙面深度做两三次不动点迭代
fn lookupSand(Q: vec3f, d: vec3f) -> vec4f {
  var uv = project(Q);
  var found = false;
  for (var k = 0; k < 3; k++) {
    let sd = textureLoad(sandDepth, clampPx(uv), 0).r;
    if (sd <= 0.0) { break; }
    found = true;
    let t = (-sd - Q.z) / min(d.z, -1e-4);
    uv = project(Q + d * max(t, 0.0));
  }
  let c = textureSampleLevel(sandColor, linSamp, uv * V.screen.zw, 0.0);
  if (found && textureLoad(sandDepth, clampPx(uv), 0).r > 0.0) { return vec4f(c.rgb, 1.0); }
  return c;
}

// 沿内部光线穿过背面看到的背景
fn lookupBackground(Q: vec3f, d: vec3f) -> vec3f {
  var X = Q + d * 0.15;
  var uv = project(X);
  for (var k = 0; k < 2; k++) {
    let sd = sceneDist(clampPx(uv));
    let t = (sd + Q.z) / max(-d.z, 1e-4);
    X = Q + d * clamp(t, 0.0, 3.0);
    uv = project(X);
  }
  return sampleScene(uv);
}

fn shadeGlass(px: vec2f, gf: vec4f) -> vec3f {
  let Pf = viewPosFromDepth(px, gf.w);
  let Vd = normalize(-Pf);
  var N = normalize(gf.xyz);
  if (dot(N, Vd) < 0.0) { N = -N; }
  let cosi = max(dot(N, Vd), 0.0);
  let F = fresnel(cosi);
  let refl = env(viewToWorldDir(reflect(-Vd, N)));

  let d1 = refract(-Vd, N, 1.0 / IOR);
  let toLocal = V.modelInv * V.viewInv;
  let toView = V.view * V.model;
  let O = (toLocal * vec4f(Pf, 1.0)).xyz;
  let D = normalize((toLocal * vec4f(d1, 0.0)).xyz);
  let H = bulbH();
  let ay = min(abs(O.y), H);
  let pr = profileAt(ay / H);
  let ri = pr.x;

  // 把内表面在这个高度近似成圆柱，求光线在玻璃里走到内表面的距离
  let a = dot(D.xz, D.xz);
  let b = dot(O.xz, D.xz);
  let c = dot(O.xz, O.xz) - ri * ri;
  let disc = b * b - a * c;
  var tIn = -1.0;
  if (disc > 0.0 && a > 1e-8) { tIn = (-b - sqrt(disc)) / a; }

  let bk = textureLoad(glassBack, clampPx(px), 0);
  var interior: vec3f;
  var path = 0.0;
  var innerRefl = vec3f(0.0);
  if (tIn > 0.0 && tIn < 0.02) {
    let Q = O + D * tIn;
    path = tIn;
    let rq = max(length(Q.xz), 1e-9);
    let radial = vec3f(Q.x, 0.0, Q.z) / rq;
    let sy = select(-1.0, 1.0, Q.y >= 0.0);
    let s = profileAt(abs(Q.y) / H).y;
    let Nin = normalize(radial - vec3f(0.0, s * sy, 0.0));
    var d2 = refract(D, Nin, IOR);
    let Fin = fresnel(abs(dot(D, Nin)));
    innerRefl = env(viewToWorldDir(normalize((toView * vec4f(reflect(D, Nin), 0.0)).xyz))) * Fin * 0.6;
    if (dot(d2, d2) < 0.5) { d2 = reflect(D, Nin); }
    let Qv = (toView * vec4f(Q, 1.0)).xyz;
    let d2v = normalize((toView * vec4f(d2, 0.0)).xyz);
    var sand = vec4f(0.0);
    var bg = vec3f(0.5);
    if (MODE != 2u) {
      sand = lookupSand(Qv, d2v);
      bg = lookupBackground(Qv, d2v);
    }
    // 背面玻璃：再一次菲涅尔损失，加上壁厚
    if (bk.w > 0.0) {
      let Nb = normalize(bk.xyz);
      bg *= 1.0 - fresnel(abs(dot(Nb, d2v)));
      path += (pr.z - pr.x) / max(abs(dot(Nb, d2v)), 0.2);
    }
    interior = sand.rgb + (1.0 - sand.a) * bg;
  } else {
    // 掠射的边缘或实心的瓶颈：光在玻璃壁里走，从背面出去
    path = (pr.z - pr.x) * 2.0 / max(cosi, 0.15);
    let Pv = Pf + d1 * 0.003;
    interior = lookupBackground(Pv, normalize(d1)) * 0.9;
  }
  let absorb = exp(-vec3f(10.0, 5.0, 7.0) * path);
  return F * refl + (1.0 - F) * (absorb * interior + innerRefl);
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn toSrgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

@fragment
fn fs(i: FsOut) -> @location(0) vec4f {
  let ip = vec2i(i.pos.xy);
  var col = textureLoad(sceneColor, ip, 0).rgb;
  let gf = textureLoad(glassFront, ip, 0);
  if (gf.w > 0.0 && MODE != 1u) { col = shadeGlass(i.pos.xy, gf); }
  let mapped = toSrgb(aces(col * V.misc.z));
  let dither = (u01(pcg(u32(ip.x) * 1973u + u32(ip.y) * 9277u + u32(V.misc.y) * 26699u)) - 0.5) / 255.0;
  return vec4f(mapped + dither, 1.0);
}
