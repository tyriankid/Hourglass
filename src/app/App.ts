// 应用编排：状态机、沙量控制器（凝聚/消散）、沙子时钟、离开画面后的补算、每帧调度。

import { SandClock, formatClock } from '../clock/SandClock';
import { BULB_H, innerRadius } from '../geometry/profile';
import { HourglassMotion } from '../interaction/Motion';
import { Input } from '../interaction/Input';
import { RHO_BULK, W0, beverlooRate, clumpMass } from '../physics';
import { Renderer } from '../render/Renderer';
import { HIST_BINS, SandSim, defaultParams, type FrameStats } from '../sim/SandSim';
import { Overlay, type Settings } from '../ui/Overlay';
import type { Vec3 } from '../math/mat4';

export type State = 'idle' | 'flipping' | 'running' | 'laying' | 'paused' | 'rising' | 'finished';

const H = BULB_H;
const MAX_SAND = W0 * 3600;

/** 从端盖一侧往瓶颈方向，内腔容积达到 v 时离瓶颈的距离（沙面平铺时的高度） */
function flatLevelFromNeck(v: number): number {
  let acc = 0;
  const steps = 900;
  const dy = H / steps;
  for (let i = 0; i < steps; i++) {
    const ay = H - (i + 0.5) * dy;
    const r = innerRadius(ay)[0];
    acc += Math.PI * r * r * dy;
    if (acc >= v) return ay;
  }
  return 0;
}

export class App {
  readonly sim: SandSim;
  readonly renderer: Renderer;
  readonly motion = new HourglassMotion();
  readonly input: Input;
  readonly ui: Overlay;
  state: State = 'idle';
  clock: SandClock | null = null;
  /** 应用时间（秒），单调递增，物理和渲染共用；从 1 开始（0 表示"没有凝聚时间"） */
  time = 1;
  private mc: number;
  private massPos = 0;
  private massNeg = 0;
  private maxSpeed = 0;
  private statsSeen = 0;
  private cooldown = 30;
  private histBusy = false;
  private flipQueued = false;
  private uprightAngle = 0;
  private finishedAt = 0;
  private hiddenAt = 0;
  private busy = false;
  private prev = performance.now();
  private inFlight = 0;
  private lastStats: FrameStats | null = null;
  onFrame: ((app: App, s: FrameStats | null) => void) | null = null;

  constructor(private device: GPUDevice, canvas: HTMLCanvasElement, readonly rc: number, readonly settings: Settings) {
    this.sim = new SandSim(device, defaultParams(rc, MAX_SAND));
    this.mc = clumpMass(rc);
    this.renderer = new Renderer(device, canvas, this.sim);
    this.renderer.camera.distance = 0.78;
    this.input = new Input(canvas, this.renderer, this.motion);
    this.ui = new Overlay(settings);
    this.input.onTapHourglass = () => this.tap();
    this.ui.onReset = () => this.reset();
    this.ui.onMinutes = () => {
      this.cooldown = Math.min(this.cooldown, 2);
    };
    this.ui.onSettings = (s) => {
      this.input.enableGyro(s.gyro);
    };
    if (settings.gyro) this.input.enableGyro(true);
    this.sim.onStats = (s) => this.onStats(s);
    document.addEventListener('visibilitychange', () => this.onVisibility());
  }

  start() {
    const loop = (now: number) => {
      requestAnimationFrame(loop);
      this.frame(now);
    };
    requestAnimationFrame(loop);
  }

  // ───────── 状态机 ─────────

  tap() {
    if (this.busy) return;
    switch (this.state) {
      case 'idle':
        if (this.ready()) this.flip();
        else this.flipQueued = true;
        break;
      case 'running':
        this.state = 'laying';
        this.motion.layDown(() => (this.state = 'paused'));
        break;
      case 'paused':
        this.state = 'rising';
        this.motion.rise(this.uprightAngle, () => (this.state = 'running'));
        break;
      case 'finished':
        this.state = 'idle';
        break;
    }
  }

  private flip() {
    const down = this.downBulb();
    const total = down > 0 ? this.massPos : this.massNeg;
    this.clock = new SandClock(total);
    this.flipQueued = false;
    this.state = 'flipping';
    this.uprightAngle = Math.round((this.motion.angle + Math.PI) / Math.PI) * Math.PI;
    this.motion.flip(() => (this.state = 'running'));
  }

  private reset() {
    if (this.state === 'idle' || this.busy) return;
    const wasLaid = this.state === 'paused' || this.state === 'laying';
    const finishReset = () => {
      const up = this.motion.localUp();
      const enc = this.device.createCommandEncoder();
      // 上泡剩下的沙化光消散；下泡由沙量控制器补齐到刻度尺上的时间
      this.sim.dissolve(enc, (up[1] > 0 ? 1 : -1) as 1 | -1, up, -1, true, this.time);
      this.device.queue.submit([enc.finish()]);
      this.clock = null;
      this.state = 'idle';
      this.cooldown = 12;
    };
    if (wasLaid) {
      this.state = 'rising';
      this.motion.rise(this.uprightAngle, finishReset);
    } else {
      finishReset();
    }
  }

  /** 当前朝下的那个泡（局部 y 的符号） */
  private downBulb(): 1 | -1 {
    return this.motion.localUp()[1] > 0 ? -1 : 1;
  }

  private ready(): boolean {
    const target = W0 * this.ui.minutes * 60;
    const m = this.downBulb() > 0 ? this.massPos : this.massNeg;
    return Math.abs(m - target) <= 1 && this.cooldown === 0 && !this.histBusy && this.maxSpeed < 0.08 && this.statsSeen > 10;
  }

  private onStats(s: FrameStats) {
    this.lastStats = s;
    this.statsSeen++;
    this.massPos = s.massPos;
    this.massNeg = s.massNeg;
    this.maxSpeed = s.maxSpeed;
    if (this.clock && this.state !== 'idle') this.clock.add(s.drained);
  }

  // ───────── 沙量控制器（调时间时） ─────────

  private controller(enc: GPUCommandEncoder) {
    if (this.cooldown > 0) {
      this.cooldown--;
      return;
    }
    if (this.histBusy || this.statsSeen < 3) return;
    const target = W0 * this.ui.minutes * 60;
    const s = this.downBulb();
    const measured = s > 0 ? this.massPos : this.massNeg;
    const diff = target - measured;
    const up = this.motion.localUp();
    if (diff > 1) {
      // 在下泡空腔里凝聚：区域避开沙堆（平铺高度再往瓶颈方向让出锥顶的高度）
      const flat = flatLevelFromNeck(measured / RHO_BULK);
      const far = Math.min(0.6 * H, flat - 0.025);
      const near = 0.07 * H;
      const lo = far > near + 0.008 ? near : 0.04 * H;
      const hi = far > near + 0.008 ? far : 0.14 * H;
      const ra = innerRadius(lo)[0];
      const rb = innerRadius(hi)[0];
      const vol = (Math.PI / 3) * (hi - lo) * (ra * ra + ra * rb + rb * rb);
      const vClump = (4 / 3) * Math.PI * this.rc ** 3;
      const cap = Math.max(1, Math.floor((vol * 0.07) / vClump));
      let count: number;
      let massEach: number;
      if (diff < this.mc) {
        count = 1;
        massEach = diff;
      } else {
        count = Math.min(cap, Math.floor(diff / this.mc));
        massEach = this.mc;
      }
      this.sim.condense(enc, { y0: s * lo, y1: s * hi, bulb: s, count, massEach, glow: true, time: this.time, up });
      this.cooldown = 3;
    } else if (diff < -1) {
      // 多了：读高度直方图，从顶层往下消散刚好够的量
      this.histBusy = true;
      const lo = -H - 0.01;
      const hi = H + 0.01;
      this.sim.histogram(s, up, lo, hi).then((hist) => {
        const excess = -diff * 1000;
        let acc = 0;
        let bin = HIST_BINS - 1;
        for (; bin >= 0; bin--) {
          acc += hist[bin];
          if (acc >= excess) break;
        }
        const thr = lo + ((hi - lo) * Math.max(bin, 0)) / HIST_BINS;
        const e = this.device.createCommandEncoder();
        this.sim.dissolve(e, s, up, thr, true, this.time);
        this.device.queue.submit([e.finish()]);
        this.histBusy = false;
        this.cooldown = 4;
      });
    }
  }

  // ───────── 离开画面后的补算 ─────────

  private onVisibility() {
    if (document.hidden) {
      this.hiddenAt = performance.now();
      return;
    }
    const gap = (performance.now() - this.hiddenAt) / 1000;
    this.prev = performance.now();
    if (gap > 0.5) this.catchUp(gap);
  }

  /** 离开期间应该流过的沙：从上泡顶层移走，在下泡沙堆上方摆好，再用真实物理沉降 */
  private async catchUp(gap: number) {
    if (!this.clock || this.busy) return;
    // 只有竖立计时中才在流；横放暂停时不流
    if (this.state !== 'running' && this.state !== 'flipping' && this.state !== 'rising') return;
    const remaining = this.clock.totalMg - this.clock.drained;
    const kin = this.motion.kinematics();
    const rate = beverlooRate(Math.abs(kin.gLocal[1]));
    const want = Math.min(remaining, rate * gap);
    if (want < 1) return;
    this.busy = true;
    this.ui.veilOn(true);
    await this.sim.flush();
    const up = this.motion.localUp();
    const upBulb: 1 | -1 = up[1] > 0 ? 1 : -1;
    const lo = -H - 0.01;
    const hi = H + 0.01;
    // 1. 上泡：从顶层往下移走
    const histUp = await this.sim.histogram(upBulb, up, lo, hi);
    let acc = 0;
    let bin = HIST_BINS - 1;
    for (; bin >= 0; bin--) {
      acc += histUp[bin];
      if (acc >= want * 1000) break;
    }
    const thr = lo + ((hi - lo) * Math.max(bin, 0)) / HIST_BINS;
    const removed = await this.sim.dissolveMeasured(upBulb, up, thr);
    // 2. 下泡：在沙堆顶上方按点阵摆放同样多的沙
    const histDown = await this.sim.histogram(-upBulb as 1 | -1, up, lo, hi);
    let top = lo;
    for (let b = HIST_BINS - 1; b >= 0; b--)
      if (histDown[b] > 0) {
        top = lo + ((hi - lo) * (b + 1)) / HIST_BINS;
        break;
      }
    const items = this.latticeAbove(-upBulb as 1 | -1, up, top, removed);
    const enc = this.device.createCommandEncoder();
    this.sim.place(enc, items);
    this.device.queue.submit([enc.finish()]);
    this.clock.add(removed);
    // 3. 不渲染，只跑物理让沙沉降（孔口关闭，免得这段计算时间里又流出沙）
    for (let i = 0; i < 72; i++) {
      const e = this.device.createCommandEncoder();
      this.sim.encodeFrame(e, { wallDt: 1 / 60, substeps: 8, gLocal: kin.gLocal, omega: [0, 0, 0], alpha: [0, 0, 0], flow: false });
      this.device.queue.submit([e.finish()]);
      this.sim.afterSubmit();
      if (i % 8 === 7) await this.sim.flush();
    }
    await this.sim.flush();
    this.prev = performance.now();
    this.busy = false;
    this.ui.veilOn(false);
    // 沉降这段时间本身也算离开，再补一次（很快收敛）
    const extra = (performance.now() - this.prev) / 1000;
    if (extra > 0.5) this.catchUp(extra);
  }

  /** 在 bulb 泡里、沿 up 方向高于 top 的空腔按点阵摆放总质量为 mass 的颗粒团，从低往高 */
  private latticeAbove(bulb: 1 | -1, up: Vec3, top: number, mass: number): Float32Array {
    const rc = this.rc;
    const s = 2 * rc * 1.02;
    const out: number[] = [];
    let left = mass;
    const pts: number[][] = [];
    for (let ay = rc; ay < H; ay += s) {
      const y = bulb * ay;
      const r = innerRadius(ay)[0] - rc * 1.05;
      if (r <= 0) continue;
      for (let x = -r; x <= r; x += s)
        for (let z = -r; z <= r; z += s) {
          if (x * x + z * z > r * r) continue;
          const h = x * up[0] + y * up[1] + z * up[2];
          if (h > top + rc * 1.1) pts.push([x, y, z, h]);
        }
    }
    pts.sort((a, b) => a[3] - b[3]);
    for (const p of pts) {
      if (left <= 1e-6) break;
      const m = Math.min(this.mc, left);
      out.push(p[0], p[1], p[2], m);
      left -= m;
    }
    return new Float32Array(out);
  }

  // ───────── 每帧 ─────────

  /** 测试用：按固定步长同步推进一帧，等 GPU 做完 */
  async step(dt: number) {
    this.prev = performance.now() - dt * 1000;
    this.inFlight = 0;
    this.frame(performance.now(), dt);
    await this.device.queue.onSubmittedWorkDone();
    await this.sim.flush();
  }

  get debugState() {
    return { state: this.state, massPos: this.massPos, massNeg: this.massNeg, maxSpeed: this.maxSpeed, remaining: this.clock?.remainingSeconds ?? null };
  }

  private frame(now: number, fixedDt?: number) {
    if (this.busy || this.inFlight >= 2) return;
    const wallDt = fixedDt ?? Math.max(0, (now - this.prev) / 1000);
    this.prev = now;
    if (wallDt > 0.5 && this.clock && this.state === 'running') {
      this.catchUp(wallDt);
      return;
    }
    const dt = Math.min(wallDt, 0.05);
    this.time += dt;
    this.ui.update(dt);
    this.input.update(dt);
    this.motion.update(dt);
    const kin = this.motion.kinematics();
    const byStability = Math.ceil(dt / 0.0021);
    const byMotion = Math.ceil((this.maxSpeed * dt) / (0.5 * this.rc));
    const substeps = Math.max(4, Math.min(16, Math.max(byStability, byMotion)));

    const enc = this.device.createCommandEncoder();
    if (this.state === 'idle') {
      this.controller(enc);
      if (this.flipQueued && this.ready()) this.flip();
    }
    this.sim.encodeFrame(enc, {
      wallDt: dt,
      drainDt: wallDt,
      substeps,
      gLocal: kin.gLocal,
      omega: kin.omega,
      alpha: kin.alpha,
      flow: true,
      flushDeposits: this.state === 'idle',
    });
    this.renderer.model = this.motion.model();
    // 对准点比瓶颈略低，沙漏在画面里偏上，给底部的时间和刻度尺留位置
    this.renderer.camera.target = [0, Math.max(0.05, this.motion.center()[1] - 0.02), 0];
    this.renderer.encode(enc, this.time);
    this.device.queue.submit([enc.finish()]);
    this.sim.afterSubmit();
    this.inFlight++;
    this.device.queue.onSubmittedWorkDone().then(() => this.inFlight--);
    this.updateUi();
    this.onFrame?.(this, this.lastStats);
  }

  private updateUi() {
    const ui = this.ui;
    switch (this.state) {
      case 'idle': {
        ui.setTime(formatClock(ui.minutes * 60));
        ui.setHint(this.flipQueued ? '沙子正在落下…' : '轻点沙漏开始');
        ui.showRuler(true);
        ui.showReset(false);
        break;
      }
      case 'flipping':
      case 'running':
      case 'rising': {
        const c = this.clock!;
        ui.setTime(formatClock(c.remainingSeconds));
        ui.setHint(this.state === 'running' ? '轻点沙漏暂停' : '');
        ui.showRuler(false);
        ui.showReset(true);
        const upMass = this.downBulb() > 0 ? this.massNeg : this.massPos;
        if (this.state === 'running' && (c.done || (c.remainingSeconds < 1 && upMass < 1))) {
          this.state = 'finished';
          this.finishedAt = this.time;
        }
        break;
      }
      case 'laying':
      case 'paused':
        ui.setTime(formatClock(this.clock!.remainingSeconds));
        ui.setHint(this.state === 'paused' ? '已暂停 · 轻点沙漏继续' : '');
        ui.showRuler(false);
        ui.showReset(true);
        break;
      case 'finished':
        ui.setTime('00:00');
        ui.setHint('完成');
        ui.showRuler(false);
        ui.showReset(false);
        if (this.time - this.finishedAt > 4) {
          this.state = 'idle';
          this.clock = null;
        }
        break;
    }
  }
}
