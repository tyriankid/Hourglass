// 沙漏的姿态与运动：翻转、横放、立起的动画，拖拽/陀螺仪带来的倾斜（弹簧阻尼，松手会回摆），
// 始终让沙漏最低点贴着桌面。每帧给出模型矩阵，以及物理需要的局部系有效重力、角速度、角加速度。

import { BULB_H, CAP_R, CAP_T } from '../geometry/profile';
import { G0 } from '../physics';
import type { Mat4, Vec3 } from '../math/mat4';

type Mat3 = Float32Array; // 列主序

function mat3Mul(a: Mat3, b: Mat3): Mat3 {
  const o = new Float32Array(9);
  for (let c = 0; c < 3; c++)
    for (let r = 0; r < 3; r++) o[c * 3 + r] = a[r] * b[c * 3] + a[3 + r] * b[c * 3 + 1] + a[6 + r] * b[c * 3 + 2];
  return o;
}

function axisAngle(axis: Vec3, ang: number): Mat3 {
  const l = Math.hypot(axis[0], axis[1], axis[2]);
  if (l < 1e-9 || Math.abs(ang) < 1e-12) return new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const [x, y, z] = [axis[0] / l, axis[1] / l, axis[2] / l];
  const c = Math.cos(ang);
  const s = Math.sin(ang);
  const t = 1 - c;
  return new Float32Array([t * x * x + c, t * x * y + s * z, t * x * z - s * y, t * x * y - s * z, t * y * y + c, t * y * z + s * x, t * x * z + s * y, t * y * z - s * x, t * z * z + c]);
}

function mulVec(m: Mat3, v: Vec3): Vec3 {
  return [m[0] * v[0] + m[3] * v[1] + m[6] * v[2], m[1] * v[0] + m[4] * v[1] + m[7] * v[2], m[2] * v[0] + m[5] * v[1] + m[8] * v[2]];
}

function mulTVec(m: Mat3, v: Vec3): Vec3 {
  return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
}

/** 相对旋转 B·Aᵀ 对应的旋转向量（轴 × 角度） */
function rotVec(a: Mat3, b: Mat3): Vec3 {
  // R = B Aᵀ
  const r = new Float32Array(9);
  for (let c = 0; c < 3; c++) for (let rr = 0; rr < 3; rr++) r[c * 3 + rr] = b[rr] * a[c] + b[3 + rr] * a[3 + c] + b[6 + rr] * a[6 + c];
  const tr = r[0] + r[4] + r[8];
  const ang = Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2)));
  if (ang < 1e-9) return [0, 0, 0];
  const k = ang / (2 * Math.sin(ang));
  return [(r[5] - r[7]) * k, (r[6] - r[2]) * k, (r[1] - r[3]) * k];
}

const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

interface Anim {
  from: number;
  to: number;
  t: number;
  dur: number;
  lift: number;
  done?: () => void;
}

export interface Kinematics {
  /** 局部系有效重力（含平移惯性力） */
  gLocal: Vec3;
  omega: Vec3;
  alpha: Vec3;
}

export class HourglassMotion {
  /** 绕世界 z 轴的静止角：0 立着，π 倒过来，±π/2 横躺 */
  angle = 0;
  private anim: Anim | null = null;
  /** 倾斜（世界系里的旋转向量的 x、z 分量）及其速度、目标 */
  tilt: [number, number] = [0, 0];
  private tiltV: [number, number] = [0, 0];
  tiltTarget: [number, number] = [0, 0];
  /** 弹簧：自然频率（rad/s）和阻尼比；阻尼比小于 1 会回摆，让沙子晃动 */
  springW = 9;
  springZeta = 0.32;

  private R: Mat3 = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  private p: Vec3 = [0, BULB_H + CAP_T, 0];
  private hist: Array<{ R: Mat3; p: Vec3; dt: number }> = [];
  private omegaW: Vec3 = [0, 0, 0];
  private alphaW: Vec3 = [0, 0, 0];
  private velW: Vec3 = [0, 0, 0];
  private accW: Vec3 = [0, 0, 0];

  get animating(): boolean {
    return this.anim !== null;
  }

  /** 端到端翻转 180°：提起、转、放下 */
  flip(done?: () => void) {
    this.start(this.angle + Math.PI, 1.35, 0.035, done);
  }

  /** 横放：向一侧倒下 90° */
  layDown(done?: () => void) {
    const to = Math.round(this.angle / Math.PI) * Math.PI + Math.PI / 2;
    this.start(to, 1.1, 0.004, done);
  }

  /** 从横放立起来，回到最近的竖直角度（保持原来的上下朝向） */
  rise(uprightAngle: number, done?: () => void) {
    this.start(uprightAngle, 1.0, 0.004, done);
  }

  private start(to: number, dur: number, lift: number, done?: () => void) {
    this.anim = { from: this.angle, to, t: 0, dur, lift, done };
  }

  /** 当前是否竖直（不在动画中、角度接近 0 或 π） */
  get upright(): boolean {
    const a = ((this.angle % Math.PI) + Math.PI) % Math.PI;
    return !this.anim && (a < 0.05 || a > Math.PI - 0.05);
  }

  /** 推进一帧 */
  update(dt: number) {
    let lift = 0;
    if (this.anim) {
      const a = this.anim;
      a.t = Math.min(a.t + dt / a.dur, 1);
      const e = easeInOut(a.t);
      this.angle = a.from + (a.to - a.from) * e;
      lift = a.lift * Math.sin(Math.PI * a.t);
      if (a.t >= 1) {
        this.angle = a.to;
        this.anim = null;
        a.done?.();
      }
    }
    // 倾斜弹簧（半隐式欧拉，按子步积分保持稳定）
    const n = Math.max(1, Math.ceil(dt / 0.004));
    const h = dt / n;
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < 2; i++) {
        const acc = -this.springW * this.springW * (this.tilt[i] - this.tiltTarget[i]) - 2 * this.springZeta * this.springW * this.tiltV[i];
        this.tiltV[i] += acc * h;
        this.tilt[i] += this.tiltV[i] * h;
      }
    }
    const base = axisAngle([0, 0, 1], this.angle);
    const tiltAng = Math.hypot(this.tilt[0], this.tilt[1]);
    const tiltR = axisAngle([this.tilt[0], 0, this.tilt[1]], tiltAng);
    const R = mat3Mul(tiltR, base);
    // 贴地：两个端盖最低点中更低的那个落在桌面上
    const axis = mulVec(R, [0, 1, 0]);
    const ay = Math.abs(axis[1]);
    const side = Math.sqrt(Math.max(0, 1 - axis[1] * axis[1]));
    let low = Infinity;
    for (const s of [1, -1]) {
      const cy = s * (BULB_H + CAP_T / 2) * axis[1];
      low = Math.min(low, cy - (CAP_T / 2) * ay - CAP_R * side);
    }
    const p: Vec3 = [0, -low + lift, 0];
    this.pushHistory(R, p, dt);
  }

  private pushHistory(R: Mat3, p: Vec3, dt: number) {
    const prev = this.hist[this.hist.length - 1];
    this.R = R;
    this.p = p;
    this.hist.push({ R, p, dt });
    if (this.hist.length > 3) this.hist.shift();
    if (!prev || dt <= 0) return;
    // 有限差分求速度、加速度、角速度、角加速度，再做一点低通，避免帧间抖动
    const v: Vec3 = [(p[0] - prev.p[0]) / dt, (p[1] - prev.p[1]) / dt, (p[2] - prev.p[2]) / dt];
    const w = rotVec(prev.R, R).map((x) => x / dt) as Vec3;
    const k = 1 - Math.exp(-dt / 0.03);
    const a: Vec3 = [(v[0] - this.velW[0]) / dt, (v[1] - this.velW[1]) / dt, (v[2] - this.velW[2]) / dt];
    const al: Vec3 = [(w[0] - this.omegaW[0]) / dt, (w[1] - this.omegaW[1]) / dt, (w[2] - this.omegaW[2]) / dt];
    for (let i = 0; i < 3; i++) {
      this.accW[i] += (a[i] - this.accW[i]) * k;
      this.alphaW[i] += (al[i] - this.alphaW[i]) * k;
      this.velW[i] = v[i];
      this.omegaW[i] = w[i];
    }
  }

  kinematics(): Kinematics {
    const g: Vec3 = [-this.accW[0], -G0 - this.accW[1], -this.accW[2]];
    return { gLocal: mulTVec(this.R, g), omega: mulTVec(this.R, this.omegaW), alpha: mulTVec(this.R, this.alphaW) };
  }

  model(): Mat4 {
    const m = new Float32Array(16);
    for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) m[c * 4 + r] = this.R[c * 3 + r];
    m[12] = this.p[0];
    m[13] = this.p[1];
    m[14] = this.p[2];
    m[15] = 1;
    return m;
  }

  /** 世界系里沙漏中心（瓶颈）的位置 */
  center(): Vec3 {
    return [...this.p];
  }

  /** 局部系里"向上"的方向（与有效重力相反） */
  localUp(): Vec3 {
    const g = mulTVec(this.R, [0, -1, 0]);
    return [-g[0], -g[1], -g[2]];
  }
}
