// 旋转体网格：玻璃外表面和金属端盖。顶点格式：位置 xyz + 法线 xyz。

import { BULB_H, CAP_R, CAP_T, outerRadius } from '../geometry/profile';

export interface Mesh {
  vertices: Float32Array;
  indices: Uint32Array;
}

/**
 * 把 (ρ, y, nρ, ny) 的轮廓点列绕 y 轴旋转。相邻两点之间生成一圈四边形；
 * 三角形的绕向按顶点法线自动纠正为从外面看逆时针。
 */
function lathe(rings: Array<[number, number, number, number]>, segments: number, splitAt: number[] = []): Mesh {
  const v: number[] = [];
  const idx: number[] = [];
  const cols = segments + 1;
  for (const [r, y, nr, ny] of rings) {
    for (let j = 0; j <= segments; j++) {
      const a = (j / segments) * Math.PI * 2;
      const c = Math.cos(a);
      const s = Math.sin(a);
      v.push(r * c, y, r * s, nr * c, ny, nr * s);
    }
  }
  const pos = (k: number) => [v[k * 6], v[k * 6 + 1], v[k * 6 + 2]];
  const nrm = (k: number) => [v[k * 6 + 3], v[k * 6 + 4], v[k * 6 + 5]];
  const tri = (a: number, b: number, c: number) => {
    const pa = pos(a);
    const pb = pos(b);
    const pc = pos(c);
    const e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    const e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const na = nrm(a);
    const nb = nrm(b);
    const nc = nrm(c);
    const avg = [na[0] + nb[0] + nc[0], na[1] + nb[1] + nc[1], na[2] + nb[2] + nc[2]];
    if (n[0] * avg[0] + n[1] * avg[1] + n[2] * avg[2] >= 0) idx.push(a, b, c);
    else idx.push(a, c, b);
  };
  for (let i = 0; i < rings.length - 1; i++) {
    if (splitAt.includes(i)) continue;
    for (let j = 0; j < segments; j++) {
      const a = i * cols + j;
      const b = i * cols + j + 1;
      const c = (i + 1) * cols + j + 1;
      const d = (i + 1) * cols + j;
      tri(a, b, c);
      tri(a, c, d);
    }
  }
  return { vertices: new Float32Array(v), indices: new Uint32Array(idx) };
}

/** 玻璃外表面：从 y = -H 到 +H，瓶颈附近加密 */
export function glassMesh(rows = 150, segments = 112): Mesh {
  const rings: Array<[number, number, number, number]> = [];
  const ys: number[] = [];
  for (let i = rows; i >= 1; i--) ys.push(-BULB_H * (i / rows) ** 1.7);
  ys.push(0);
  for (let i = 1; i <= rows; i++) ys.push(BULB_H * (i / rows) ** 1.7);
  for (const y of ys) {
    const [r, dr] = outerRadius(Math.abs(y));
    const sy = y >= 0 ? 1 : -1;
    const l = Math.hypot(1, dr);
    rings.push([r, y, 1 / l, (-dr * sy) / l]);
  }
  return lathe(rings, segments);
}

/** 上下两个端盖：带 0.8 mm 倒角的圆饼 */
export function capsMesh(segments = 112): Mesh {
  const b = 0.0008;
  const parts: Mesh[] = [];
  for (const sy of [1, -1]) {
    const y0 = sy * BULB_H;
    const y1 = sy * (BULB_H + CAP_T);
    const lo = Math.min(y0, y1);
    const hi = Math.max(y0, y1);
    const s = Math.SQRT1_2;
    // 每段两端各一个点，法线取该段的面法线（段内平直，绕轴平滑）
    const prof: Array<[number, number, number, number]> = [
      [0, lo, 0, -1],
      [CAP_R - b, lo, 0, -1],
      [CAP_R - b, lo, s, -s],
      [CAP_R, lo + b, s, -s],
      [CAP_R, lo + b, 1, 0],
      [CAP_R, hi - b, 1, 0],
      [CAP_R, hi - b, s, s],
      [CAP_R - b, hi, s, s],
      [CAP_R - b, hi, 0, 1],
      [0, hi, 0, 1],
    ];
    // 只在同一段的两个点之间连面
    parts.push(lathe(prof, segments, [1, 3, 5, 7]));
  }
  return mergeMeshes(parts);
}

function mergeMeshes(list: Mesh[]): Mesh {
  let vn = 0;
  let inum = 0;
  for (const m of list) {
    vn += m.vertices.length;
    inum += m.indices.length;
  }
  const vertices = new Float32Array(vn);
  const indices = new Uint32Array(inum);
  let vo = 0;
  let io = 0;
  for (const m of list) {
    vertices.set(m.vertices, vo);
    for (let k = 0; k < m.indices.length; k++) indices[io + k] = m.indices[k] + vo / 6;
    vo += m.vertices.length;
    io += m.indices.length;
  }
  return { vertices, indices };
}
