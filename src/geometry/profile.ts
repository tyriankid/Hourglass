// 沙漏玻璃的轮廓：一个关于瓶颈（y = 0）上下对称的旋转体。
// 物理（碰撞 SDF）和渲染（光线追踪）共用这里生成的同一张采样表，保证看到的玻璃就是碰撞边界。

/** 单个玻璃泡内腔高度：瓶颈到端盖内表面 */
export const BULB_H = 0.09;
/** 内腔最大半径 */
export const BULB_R = 0.036;
/** 瓶颈孔半径（孔径 1.0 mm） */
export const NECK_R = 0.0005;
/** 端盖厚度和相对玻璃外缘多出的半径 */
export const CAP_T = 0.007;
export const CAP_OVERHANG = 0.0025;

/** 采样表：u = |y|/H 从 0 到 1 均匀取 PROFILE_N + 1 个点 */
export const PROFILE_N = 128;

// 内腔形状 B(u)：0 在瓶颈，1 在最宽处。下段近似圆锥（壁面陡于休止角，保证沙能流光），
// 上段是圆润的肩部，最后收到端盖处。
const CTRL: Array<[number, number]> = [
  [0.0, 0.0],
  [0.05, 0.085],
  [0.2, 0.34],
  [0.35, 0.6],
  [0.5, 0.83],
  [0.62, 0.95],
  [0.74, 1.0],
  [0.86, 0.975],
  [0.95, 0.925],
  [1.0, 0.89],
];

/** Fritsch–Carlson 单调三次插值的切线 */
function monotoneTangents(xs: number[], ys: number[]): number[] {
  const n = xs.length;
  const d: number[] = [];
  for (let i = 0; i < n - 1; i++) d.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  const m: number[] = new Array(n);
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) {
      m[i] = 0;
      m[i + 1] = 0;
      continue;
    }
    const a = m[i] / d[i];
    const b = m[i + 1] / d[i];
    const s = a * a + b * b;
    if (s > 9) {
      const t = 3 / Math.sqrt(s);
      m[i] = t * a * d[i];
      m[i + 1] = t * b * d[i];
    }
  }
  return m;
}

const XS = CTRL.map((c) => c[0]);
const YS = CTRL.map((c) => c[1]);
const MS = monotoneTangents(XS, YS);

/** B(u) 和 dB/du */
function shape(u: number): [number, number] {
  const x = Math.min(Math.max(u, 0), 1);
  let i = 0;
  while (i < XS.length - 2 && x > XS[i + 1]) i++;
  const h = XS[i + 1] - XS[i];
  const t = (x - XS[i]) / h;
  const t2 = t * t;
  const t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;
  const v = h00 * YS[i] + h10 * h * MS[i] + h01 * YS[i + 1] + h11 * h * MS[i + 1];
  const dh00 = 6 * t2 - 6 * t;
  const dh10 = 3 * t2 - 4 * t + 1;
  const dh01 = -6 * t2 + 6 * t;
  const dh11 = 3 * t2 - 2 * t;
  const dv = (dh00 * YS[i] + dh10 * h * MS[i] + dh01 * YS[i + 1] + dh11 * h * MS[i + 1]) / h;
  return [v, dv];
}

/** 玻璃壁厚：泡体 1.6 mm，越靠近瓶颈越厚（瓶颈是实心厚玻璃） */
function thickness(u: number): [number, number] {
  const a = 0.0016;
  const b = 0.0017;
  const w = 0.05;
  const e = Math.exp(-((u / w) ** 2));
  return [a + b * e, b * e * (-2 * u) / (w * w)];
}

/** 内表面半径与 dr/d|y| */
export function innerRadius(ay: number): [number, number] {
  const u = ay / BULB_H;
  const [b, db] = shape(u);
  return [NECK_R + (BULB_R - NECK_R) * b, ((BULB_R - NECK_R) * db) / BULB_H];
}

/** 外表面半径与 dr/d|y| */
export function outerRadius(ay: number): [number, number] {
  const u = ay / BULB_H;
  const [ri, dri] = innerRadius(ay);
  const [t, dt] = thickness(u);
  return [ri + t, dri + dt / BULB_H];
}

/** 端盖半径 */
export const CAP_R = outerRadius(BULB_H)[0] + CAP_OVERHANG;

/**
 * GPU 用的采样表：PROFILE_N + 1 个 vec4(rIn, drIn, rOut, drOut)，后面再跟一个
 * vec4(H, neckR, maxInnerR, capR)。共 (PROFILE_N + 2) * 4 个 f32。
 */
export function profileTable(): Float32Array {
  const out = new Float32Array((PROFILE_N + 2) * 4);
  let maxR = 0;
  for (let i = 0; i <= PROFILE_N; i++) {
    const ay = (i / PROFILE_N) * BULB_H;
    const [ri, dri] = innerRadius(ay);
    const [ro, dro] = outerRadius(ay);
    out.set([ri, dri, ro, dro], i * 4);
    maxR = Math.max(maxR, ri);
  }
  out.set([BULB_H, NECK_R, maxR, CAP_R], (PROFILE_N + 1) * 4);
  return out;
}

/** 单个玻璃泡的内腔容积（m³） */
export function bulbVolume(steps = 4000): number {
  let v = 0;
  const dy = BULB_H / steps;
  for (let i = 0; i < steps; i++) {
    const r = innerRadius((i + 0.5) * dy)[0];
    v += Math.PI * r * r * dy;
  }
  return v;
}

/** 从瓶颈到 |y| = h 这一段的内腔容积（m³） */
export function volumeBelow(h: number, steps = 2000): number {
  let v = 0;
  const dy = h / steps;
  for (let i = 0; i < steps; i++) {
    const r = innerRadius((i + 0.5) * dy)[0];
    v += Math.PI * r * r * dy;
  }
  return v;
}

/** 最宽处的 |y|：从瓶颈到这里，壁面必须足够陡，沙才能流光 */
export function widestY(steps = 2000): number {
  let best = 0;
  let bestR = 0;
  for (let i = 0; i <= steps; i++) {
    const ay = (i / steps) * BULB_H;
    const r = innerRadius(ay)[0];
    if (r > bestR) {
      bestR = r;
      best = ay;
    }
  }
  return best;
}
