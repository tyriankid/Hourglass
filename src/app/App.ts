// 应用编排：状态机、沙量控制器（凝聚/消散）、沙子时钟、离开画面后的补算、每帧调度。

import { SandClock, formatClock } from '../clock/SandClock';
import { BULB_H, innerRadius } from '../geometry/profile';
import { HourglassMotion } from '../interaction/Motion';
import { Input } from '../interaction/Input';
import { RHO_BULK, W0, beverlooRate, clumpMass } from '../physics';
import { Renderer } from '../render/Renderer';
import { HIST_BINS, SandSim, defaultParams, seedBulb, type FrameStats, type SimParams } from '../sim/SandSim';
import { Overlay, type Settings } from '../ui/Overlay';
import { AudioEngine } from '../audio/AudioEngine';
import type { Vec3 } from '../math/mat4';

export type State = 'idle' | 'flipping' | 'running' | 'laying' | 'paused' | 'rising' | 'finished';

const H = BULB_H;
const MAX_SAND = W0 * 3600;
const SESSION_KEY = 'hourglass.session.v1';

/** 计时会话的存档：页面被刷新或被系统回收后据此恢复 */
interface Session {
  total: number;
  drained: number;
  savedAt: number;
  paused: boolean;
  angle: number;
}

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
  readonly audio: AudioEngine;
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
  private prevG: number[] | null = null;
  lastWake = false;
  lastKin: unknown = null;
  /** 帧时间的指数平均（ms），用于自适应分辨率和性能面板 */
  frameMs = 16.7;
  substepsUsed = 8;
  private scaleTimer = 0;
  autoScale = true;
  onFrame: ((app: App, s: FrameStats | null) => void) | null = null;

  constructor(
    private device: GPUDevice,
    canvas: HTMLCanvasElement,
    readonly rc: number,
    readonly settings: Settings,
    tier: { maxDpr: number; photons: number } = { maxDpr: 2, photons: 768 },
    simOverrides: Partial<SimParams> = {},
  ) {
    this.sim = new SandSim(device, { ...defaultParams(rc, MAX_SAND), ...simOverrides });
    this.mc = clumpMass(rc);
    this.renderer = new Renderer(device, canvas, this.sim, tier.photons);
    this.renderer.maxDpr = tier.maxDpr;
    this.renderer.camera.distance = 0.78;
    this.input = new Input(canvas, this.renderer, this.motion);
    this.ui = new Overlay(settings);
    this.audio = new AudioEngine(settings);
    this.input.onTapHourglass = () => this.tap();
    this.ui.onMinutes = () => {
      this.cooldown = Math.min(this.cooldown, 2);
      this.audio.tick();
    };
    let quality = settings.quality;
    this.ui.onSettings = (s) => {
      this.input.enableGyro(s.gyro);
      this.audio.applySettings(s);
      // 画质决定颗粒团大小，需要重建物理：保存后刷新
      if (s.quality !== quality) {
        quality = s.quality;
        setTimeout(() => location.reload(), 150);
      }
    };
    this.input.onTapBackground = () => this.ui.closePanel();
    // iOS 只能在用户手势里申请陀螺仪权限：上次开着的话，第一次触摸时再启用
    if (settings.gyro) window.addEventListener('pointerdown', () => this.input.enableGyro(this.settings.gyro), { once: true });
    this.sim.onStats = (s) => this.onStats(s);
    document.addEventListener('visibilitychange', () => this.onVisibility());
  }

  /** 恢复上次没走完的计时：按离开时长分配两个泡里的沙，摆好后用真实物理沉降 */
  async restore(): Promise<boolean> {
    let sess: Session | null = null;
    try {
      sess = JSON.parse(localStorage.getItem(SESSION_KEY) ?? 'null');
    } catch {
      sess = null;
    }
    if (!sess || !(sess.total > 0)) return false;
    this.ui.veilOn(true);
    const away = sess.paused ? 0 : Math.max(0, (Date.now() - sess.savedAt) / 1000);
    const drained = Math.min(sess.total, sess.drained + W0 * away);
    const finished = drained >= sess.total - 1;
    this.uprightAngle = sess.angle;
    this.motion.angle = finished || !sess.paused ? sess.angle : sess.angle + Math.PI / 2;
    this.motion.update(0);
    // 按立着时的朝向分配沙：上泡从瓶颈往上堆，下泡从端盖往上堆；横放的话之后在横放姿态下沉降
    const topBulb: 1 | -1 = Math.cos(sess.angle) > 0 ? 1 : -1;
    const parts: Float32Array[] = [];
    if (sess.total - drained > 0.01) parts.push(seedBulb(this.rc, sess.total - drained, topBulb, topBulb, 3));
    if (drained > 0.01) parts.push(seedBulb(this.rc, drained, (-topBulb) as 1 | -1, topBulb, 4, 0.6));
    const all = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) {
      all.set(p, o);
      o += p.length;
    }
    this.sim.seed(all);
    const kin = this.motion.kinematics();
    for (let i = 0; i < 60; i++) {
      const e = this.device.createCommandEncoder();
      this.sim.encodeFrame(e, { wallDt: 1 / 60, substeps: 8, gLocal: kin.gLocal, omega: [0, 0, 0], alpha: [0, 0, 0], flow: false });
      this.device.queue.submit([e.finish()]);
      this.sim.afterSubmit();
      if (i % 10 === 9) await this.sim.flush();
    }
    await this.sim.flush();
    if (finished) {
      this.clearSession();
      this.state = 'idle';
    } else {
      this.clock = new SandClock(sess.total);
      this.clock.drained = drained;
      this.state = sess.paused ? 'paused' : 'running';
    }
    this.prev = performance.now();
    this.ui.veilOn(false);
    return true;
  }

  private sessionTimer = 0;

  private saveSession() {
    if (!this.clock) return;
    const sess: Session = {
      total: this.clock.totalMg,
      drained: this.clock.drained,
      savedAt: Date.now(),
      paused: this.state === 'paused' || this.state === 'laying',
      angle: this.uprightAngle,
    };
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify(sess));
    } catch {
      /* 隐私模式下忽略 */
    }
  }

  private clearSession() {
    try {
      localStorage.removeItem(SESSION_KEY);
    } catch {
      /* 忽略 */
    }
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
        this.motion.layDown(() => this.transition('laying', 'paused'));
        break;
      case 'paused':
        this.state = 'rising';
        this.motion.rise(this.uprightAngle, () => this.transition('rising', 'running'));
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
    this.motion.flip(() => this.transition('flipping', 'running'));
  }

  /** 动画结束时的状态切换：只有当前状态仍是预期的那个才切（期间可能被重置打断） */
  private transition(from: State, to: State) {
    if (this.state === from) this.state = to;
  }

  private reset() {
    if (this.state === 'idle' || this.state === 'flipping' || this.state === 'finished' || this.busy) return;
    const wasLaid = this.state === 'paused' || this.state === 'laying';
    const finishReset = () => {
      const up = this.motion.localUp();
      const enc = this.device.createCommandEncoder();
      // 上泡剩下的沙化光消散；下泡由沙量控制器补齐到刻度尺上的时间
      this.sim.dissolve(enc, (up[1] > 0 ? 1 : -1) as 1 | -1, up, -1, true, this.time);
      this.device.queue.submit([enc.finish()]);
      this.clock = null;
      this.clearSession();
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
    // 只要求沙量正确：沙还在沉降时翻转，在物理上没有任何问题
    return Math.abs(m - target) <= 1 && this.cooldown === 0 && this.settled() && this.statsSeen > 10;
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

  /** 已请求凝聚、但还没反映到测量值里的沙量（按帧号记账，统计读回后扣掉） */
  private pendingSpawn: Array<{ frame: number; mg: number }> = [];

  private controller(enc: GPUCommandEncoder) {
    if (this.cooldown > 0) {
      this.cooldown--;
      return;
    }
    if (this.histBusy || this.statsSeen < 3) return;
    const seen = this.lastStats?.frame ?? 0;
    this.pendingSpawn = this.pendingSpawn.filter((p) => p.frame > seen);
    const inflight = this.pendingSpawn.reduce((a, p) => a + p.mg, 0);
    const target = W0 * this.ui.minutes * 60;
    const s = this.downBulb();
    const measured = s > 0 ? this.massPos : this.massNeg;
    const diff = target - measured - inflight;
    const up = this.motion.localUp();
    if (diff > 1) {
      // 在下泡空腔里凝聚：区域避开沙堆（平铺高度再往瓶颈方向让出锥顶的高度）
      const flat = flatLevelFromNeck((measured + inflight) / RHO_BULK);
      const far = Math.min(0.6 * H, flat - 0.025);
      const near = 0.07 * H;
      const lo = far > near + 0.008 ? near : 0.04 * H;
      const hi = far > near + 0.008 ? far : 0.14 * H;
      // 每帧一小批，像一阵柔和的沙雨：60 分钟约 6–8 秒、25 分钟约 3 秒。
      // 点阵里有不少格点落在区域外或已被占用会被跳过，所以请求量取实际需要的 3 倍
      const perFrame = Math.max(40, Math.ceil((MAX_SAND / this.mc / 400) * 3));
      let count: number;
      let massEach: number;
      if (diff < this.mc) {
        count = 1;
        massEach = diff;
      } else {
        count = Math.min(perFrame, Math.floor(diff / this.mc));
        massEach = this.mc;
      }
      this.sim.condense(enc, { y0: s * lo, y1: s * hi, bulb: s, count, massEach, glow: true, time: this.time, up });
      // 在途量：离目标还远时按请求量的 1/3 估计（有格点会被跳过）；接近目标时按全额记，绝不超量。统计读回后以实测为准
      const requested = count * massEach;
      this.pendingSpawn.push({ frame: this.sim.nextFrame, mg: diff > 3 * requested ? requested / 3 : requested });
    } else if (diff < -1 && inflight === 0) {
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

  /** 凝聚、消散都已结束：没有在途的生成，也没有在读的直方图 */
  private settled(): boolean {
    const seen = this.lastStats?.frame ?? 0;
    return this.pendingSpawn.every((p) => p.frame <= seen) && !this.histBusy;
  }

  // ───────── 离开画面后的补算 ─────────

  private onVisibility() {
    if (document.hidden) {
      this.hiddenAt = performance.now();
      if (this.clock && this.state !== 'idle' && this.state !== 'finished') this.saveSession();
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
          // 排序键带上离轴距离：正中先堆高，摆成居中的锥形
          if (h > top + rc * 1.1) pts.push([x, y, z, h + 0.6 * Math.hypot(x, z)]);
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
    // 慢帧也计入（最多按 1 s 算），性能面板和自适应分辨率才能如实反映掉帧
    if (wallDt > 0) this.frameMs += (Math.min(wallDt, 1) * 1000 - this.frameMs) * 0.1;
    this.adaptScale(dt);
    this.ui.update(dt);
    this.input.update(dt);
    this.motion.update(dt);
    const kin = this.motion.kinematics();
    const byStability = Math.ceil(dt / 0.0021);
    const byMotion = Math.ceil((this.maxSpeed * dt) / (0.5 * this.rc));
    const substeps = Math.max(4, Math.min(16, Math.max(byStability, byMotion)));
    this.substepsUsed = substeps;

    // 沙漏在动（转动、加速、重力方向变化）时唤醒所有休眠的颗粒团
    const w = Math.hypot(...kin.omega);
    const al = Math.hypot(...kin.alpha);
    const gl = Math.hypot(...kin.gLocal);
    const prevG = this.prevG;
    const gTurn = prevG ? Math.acos(Math.min(1, (prevG[0] * kin.gLocal[0] + prevG[1] * kin.gLocal[1] + prevG[2] * kin.gLocal[2]) / (gl * Math.hypot(...prevG)))) : 0;
    this.prevG = [...kin.gLocal];
    const wakeAll = w > 0.02 || al > 0.3 || Math.abs(gl - 9.81) > 0.08 || gTurn > 0.0015;
    this.lastWake = wakeAll;
    this.lastKin = { w: +w.toFixed(4), al: +al.toFixed(3), gl: +gl.toFixed(4), gTurn: +gTurn.toFixed(5), g: kin.gLocal.map((x) => +x.toFixed(3)) };
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
      wakeAll,
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
    this.sessionTimer += dt;
    if (this.sessionTimer > 1 && this.clock && this.state !== 'idle' && this.state !== 'finished') {
      this.sessionTimer = 0;
      this.saveSession();
    }
    this.audio.update(this.lastStats, dt);
    this.onFrame?.(this, this.lastStats);
  }

  /** 持续掉帧时降低内部渲染分辨率，余量充足时升回去 */
  private adaptScale(dt: number) {
    if (!this.autoScale) return;
    this.scaleTimer += dt;
    if (this.scaleTimer < 1.5) return;
    const r = this.renderer;
    if (this.frameMs > 21 && r.renderScale > 0.5) {
      r.renderScale = Math.max(0.5, r.renderScale - 0.1);
      this.scaleTimer = 0;
    } else if (this.frameMs < 14 && r.renderScale < 1) {
      r.renderScale = Math.min(1, r.renderScale + 0.05);
      this.scaleTimer = 0;
    }
  }

  private updateUi() {
    const ui = this.ui;
    const restart = { label: '重新开始', run: () => this.reset() };
    switch (this.state) {
      case 'idle':
        ui.setTime(formatClock(ui.minutes * 60));
        ui.setHint(this.flipQueued ? '正在准备…' : '轻点沙漏开始');
        ui.setActions([]);
        ui.showRuler(true);
        break;
      case 'flipping':
      case 'running':
      case 'rising': {
        const c = this.clock!;
        ui.setTime(formatClock(c.remainingSeconds));
        ui.setHint('');
        ui.setActions(this.state === 'running' ? [{ label: '暂停', run: () => this.tap() }, restart] : []);
        ui.showRuler(false);
        const upMass = this.downBulb() > 0 ? this.massNeg : this.massPos;
        if (this.state === 'running' && (c.done || (c.remainingSeconds < 1 && upMass < 1))) {
          this.state = 'finished';
          this.finishedAt = this.time;
          this.audio.chime();
          this.clearSession();
        }
        break;
      }
      case 'laying':
      case 'paused':
        ui.setTime(formatClock(this.clock!.remainingSeconds));
        ui.setHint(this.state === 'paused' ? '已暂停' : '');
        ui.setActions(this.state === 'paused' ? [{ label: '继续', run: () => this.tap() }, restart] : []);
        ui.showRuler(false);
        break;
      case 'finished':
        ui.setTime('00:00');
        ui.setHint('完成');
        ui.setActions([
          {
            label: '再来一次',
            run: () => {
              this.state = 'idle';
              this.clock = null;
              this.flipQueued = true;
            },
          },
        ]);
        ui.showRuler(false);
        if (this.time - this.finishedAt > 6) {
          this.state = 'idle';
          this.clock = null;
        }
        break;
    }
  }
}
