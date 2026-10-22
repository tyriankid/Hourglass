// GPU 沙子模拟：颗粒团（PBD）+ 孔口边界条件（Beverloo）+ 真实尺寸沙粒 + 沉积生长。
// 每帧按墙钟时长切成若干子步，孔口抽走量按墙钟时间计算，掉帧也不影响流过的沙量。

import { bindKernel, groups, makeKernel, timeKernelCompile, type BindSpec, type Kernel } from '../gpu/kernel';
import { BULB_H, PROFILE_N, innerRadius, profileTable } from '../geometry/profile';
import { GRAIN_D, GRAIN_MASS, ORIFICE_D, beverlooRate, clumpMass } from '../physics';
import typesSrc from './wgsl/types.wgsl?raw';
import fnsSrc from './wgsl/fns.wgsl?raw';
import depositSrc from './wgsl/deposit_fn.wgsl?raw';
import predictSrc from './wgsl/predict.wgsl?raw';
import gridClearSrc from './wgsl/grid_clear.wgsl?raw';
import gridInsertSrc from './wgsl/grid_insert.wgsl?raw';
import sinkSrc from './wgsl/sink.wgsl?raw';
import solveSrc from './wgsl/solve.wgsl?raw';
import finalizeSrc from './wgsl/finalize.wgsl?raw';
import grainEmitSrc from './wgsl/grain_emit.wgsl?raw';
import grainStepSrc from './wgsl/grain_step.wgsl?raw';
import spawnSrc from './wgsl/spawn.wgsl?raw';
import mergePickSrc from './wgsl/merge_pick.wgsl?raw';
import mergeApplySrc from './wgsl/merge_apply.wgsl?raw';
import massSumSrc from './wgsl/mass_sum.wgsl?raw';
import histSrc from './wgsl/hist.wgsl?raw';
import fxFnSrc from './wgsl/fx_fn.wgsl?raw';
import condenseSrc from './wgsl/condense.wgsl?raw';
import dissolveSrc from './wgsl/dissolve.wgsl?raw';
import placeSrc from './wgsl/place.wgsl?raw';
import fxStepSrc from './wgsl/fx_step.wgsl?raw';

export interface SimParams {
  /** 颗粒团标称半径（m） */
  rc: number;
  /** 颗粒团容量 */
  clumpCap: number;
  /** 沙粒环形缓冲容量 */
  grainCap: number;
  /** 每子步的 Jacobi 迭代次数 */
  iterations: number;
  muS: number;
  muK: number;
  muWall: number;
  /** 低于这个速度（m/s）的位移视为抖动，颗粒团休眠 */
  sleepV: number;
  /** shock propagation 系数（1/m） */
  shockK: number;
  sor: number;
  /** true：Jacobi 修正按接触数平均；false：直接累加再乘 sor（欠松弛） */
  average: boolean;
  vmax: number;
  /** 沙粒速度低于它（m/s）就停下沉积 */
  grainSettleV: number;
  /** 孔口捕获半径，以 rc 为单位 */
  captureR: number;
}

export function defaultParams(rc: number, maxSandMg: number): SimParams {
  const mc = clumpMass(rc);
  return {
    rc,
    clumpCap: Math.ceil((maxSandMg / mc) * 1.5) + 2048,
    grainCap: 32768,
    iterations: 3,
    muS: 0.55,
    muK: 0.45,
    muWall: 0.3,
    sleepV: 1.0e-3,
    shockK: 300,
    sor: 0.5,
    average: false,
    vmax: 2.0,
    grainSettleV: 0.04,
    captureR: 3.5,
  };
}

export interface FrameInput {
  /** 这一帧物理推进的时长（s） */
  wallDt: number;
  /** 孔口抽走量按这个墙钟时长计算（默认等于 wallDt）；设备太慢、物理步长被截断时沙量仍跟着真实时间走 */
  drainDt?: number;
  substeps: number;
  /** 局部系里的有效重力（含平移惯性力），m/s² */
  gLocal: [number, number, number];
  omega: [number, number, number];
  alpha: [number, number, number];
  /** false 时孔口关闭（调试和测试时的沉降阶段用） */
  flow: boolean;
  /** true 时网格里攒着的沙不等门槛，全部生成颗粒团（测试收尾、补算用） */
  flushDeposits?: boolean;
}

export interface FrameStats {
  frame: number;
  /** 本帧流过孔口的沙（mg） */
  drained: number;
  emitted: number;
  impactsSand: number;
  impactsGlass: number;
  impactSpeedSum: number;
  maxSpeed: number;
  spawnFail: number;
  deposited: number;
  available: number;
  capOverflow: number;
  forced: number;
  merged: number;
  clumpsAlive: number;
  /** 两个泡里颗粒团的总质量（mg）：y > 0 的泡、y < 0 的泡 */
  massPos: number;
  massNeg: number;
  dissolved: number;
  placedFail: number;
}

export const FX_CAP = 8192;
export const PLACE_CAP = 65536;
export const HIST_BINS = 256;

export interface CondenseOp {
  /** 局部 y 区间（带符号） */
  y0: number;
  y1: number;
  bulb: 1 | -1;
  count: number;
  massEach: number;
  glow: boolean;
  time: number;
  up: [number, number, number];
}

const U_WORDS = 40;
/** 每个子步一份 uniform，按 256 字节对齐（WebGPU 动态偏移的最小对齐） */
const U_STRIDE = 256;
const MAX_SUBSTEPS = 24;
const STATS_WORDS = 32;
const STAGING = 8;

export class SandSim {
  readonly mc: number;
  readonly cell: number;
  readonly dims: [number, number, number];
  readonly gridMin: [number, number, number];
  readonly nCells: number;

  readonly buf: Record<string, GPUBuffer> = {};
  private kernels: Record<string, Kernel> = {};
  private bg: Record<string, GPUBindGroup> = {};
  private uData = new ArrayBuffer(U_STRIDE * (MAX_SUBSTEPS + 1));
  private prevKin: { g: number[]; w: number[]; a: number[] } | null = null;
  private staging: Array<{ buf: GPUBuffer; busy: boolean }> = [];
  private resetStats = true;
  private frameNo = 0;

  onStats: ((s: FrameStats) => void) | null = null;
  private histStage: GPUBuffer;

  constructor(readonly device: GPUDevice, readonly p: SimParams) {
    this.mc = clumpMass(p.rc);
    this.cell = 2.2 * p.rc;
    const table = profileTable();
    const maxR = table[(PROFILE_N + 1) * 4 + 2];
    const half = maxR + p.rc;
    const hy = BULB_H + p.rc;
    this.dims = [Math.ceil((2 * half) / this.cell), Math.ceil((2 * hy) / this.cell), Math.ceil((2 * half) / this.cell)];
    this.gridMin = [-(this.dims[0] * this.cell) / 2, -(this.dims[1] * this.cell) / 2, -(this.dims[2] * this.cell) / 2];
    this.nCells = this.dims[0] * this.dims[1] * this.dims[2];

    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const mk = (name: string, size: number, usage = S) =>
      (this.buf[name] = device.createBuffer({ label: name, size: Math.max(16, size), usage }));
    mk('U', U_STRIDE * (MAX_SUBSTEPS + 1), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    mk('P', table.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(this.buf.P, 0, table);
    mk('pos', p.clumpCap * 16);
    mk('vel', p.clumpCap * 16);
    mk('predA', p.clumpCap * 16);
    mk('predB', p.clumpCap * 16);
    mk('grid', this.nCells * (1 + 8) * 4);
    mk('ctl', 256 * 4);
    mk('freeStack', p.clumpCap * 4);
    mk('pend', p.clumpCap * 4);
    mk('recv', p.clumpCap * 4);
    const UB = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    for (const o of ['opC', 'opD', 'opH', 'opP', 'opF']) mk(o, 64, UB);
    mk('hist', HIST_BINS * 4);
    mk('fxPos', FX_CAP * 16);
    mk('fxVel', FX_CAP * 16);
    mk('items', PLACE_CAP * 16);
    this.histStage = device.createBuffer({ size: HIST_BINS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    mk('grains', p.grainCap * 16);
    mk('gvel', p.grainCap * 16);
    mk('dep', this.nCells * 4 * 4);
    for (let i = 0; i < STAGING; i++) {
      this.staging.push({
        buf: device.createBuffer({ size: STATS_WORDS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        busy: false,
      });
    }
    this.buildKernels();
    this.seed(new Float32Array(0));
  }

  private buildKernels() {
    const d = this.device;
    const U: BindSpec = { name: 'U', kind: 'uniform', type: 'SimU', dynamic: true, size: U_WORDS * 4 };
    const P: BindSpec = { name: 'P', kind: 'uniform', type: 'Profile' };
    const v4 = (name: string, kind: 'read' | 'rw'): BindSpec => ({ name, kind, type: 'array<vec4f>' });
    const atom = (name: string): BindSpec => ({ name, kind: 'rw', type: 'array<atomic<u32>>' });
    const u32r = (name: string, kind: 'read' | 'rw' = 'read'): BindSpec => ({ name, kind, type: 'array<u32>' });
    const parts = (body: string, extra = '') => ({ types: typesSrc, fns: fnsSrc + '\n' + extra, body });
    const K = this.kernels;
    K.predict = makeKernel(d, 'predict', parts(predictSrc), [U, P, v4('pos', 'rw'), v4('vel', 'read'), v4('predOut', 'rw'), atom('pend')]);
    K.gridClear = makeKernel(d, 'gridClear', parts(gridClearSrc), [U, P, atom('grid')]);
    K.gridInsert = makeKernel(d, 'gridInsert', parts(gridInsertSrc), [U, P, v4('pred', 'read'), atom('grid'), atom('ctl')]);
    K.sink = makeKernel(d, 'sink', parts(sinkSrc), [U, P, v4('pos', 'rw'), v4('pred', 'rw'), atom('ctl'), u32r('freeStack', 'rw')]);
    K.solve = makeKernel(d, 'solve', parts(solveSrc), [U, P, v4('pos', 'read'), v4('predIn', 'read'), v4('predOut', 'rw'), u32r('grid')]);
    K.finalize = makeKernel(d, 'finalize', parts(finalizeSrc), [U, P, v4('pos', 'rw'), v4('vel', 'rw'), v4('pred', 'read'), atom('ctl')]);
    K.grainEmit = makeKernel(d, 'grainEmit', parts(grainEmitSrc, depositSrc), [U, P, v4('grains', 'rw'), v4('gvel', 'rw'), atom('ctl'), atom('dep')]);
    K.grainStep = makeKernel(
      d,
      'grainStep',
      parts(grainStepSrc, depositSrc),
      [U, P, v4('grains', 'rw'), v4('gvel', 'rw'), v4('pos', 'read'), u32r('grid'), atom('dep'), atom('ctl'), atom('pend')],
    );
    K.spawn = makeKernel(d, 'spawn', parts(spawnSrc), [U, P, atom('dep'), u32r('grid'), v4('pos', 'rw'), v4('vel', 'rw'), u32r('freeStack'), atom('ctl')]);
    const O: BindSpec = { name: 'O', kind: 'uniform', type: 'OpU' };
    K.massSum = makeKernel(d, 'massSum', parts(massSumSrc), [U, P, v4('pos', 'read'), atom('ctl')]);
    K.hist = makeKernel(d, 'hist', parts(histSrc), [U, P, O, v4('pos', 'read'), atom('hist')]);
    const fxSpecs = [v4('fxPos', 'rw'), v4('fxVel', 'rw')];
    K.condense = makeKernel(d, 'condense', parts(condenseSrc, fxFnSrc), [U, P, O, v4('pos', 'rw'), v4('vel', 'rw'), u32r('freeStack'), atom('ctl'), ...fxSpecs]);
    K.dissolve = makeKernel(d, 'dissolve', parts(dissolveSrc, fxFnSrc), [U, P, O, v4('pos', 'rw'), v4('vel', 'rw'), u32r('freeStack', 'rw'), atom('ctl'), ...fxSpecs]);
    K.place = makeKernel(d, 'place', parts(placeSrc), [U, P, O, v4('items', 'read'), v4('pos', 'rw'), v4('vel', 'rw'), u32r('freeStack'), atom('ctl')]);
    K.fxStep = makeKernel(d, 'fxStep', parts(fxStepSrc), [U, P, O, ...fxSpecs]);
    K.mergePick = makeKernel(d, 'mergePick', parts(mergePickSrc), [U, P, v4('pos', 'read'), u32r('grid'), u32r('recv', 'rw')]);
    K.mergeApply = makeKernel(d, 'mergeApply', parts(mergeApplySrc), [U, P, v4('pos', 'rw'), u32r('recv'), atom('pend'), u32r('freeStack', 'rw'), atom('ctl')]);

    const B = this.buf;
    const base = { U: B.U, P: B.P };
    const bg = this.bg;
    bg.predict = bindKernel(d, K.predict, { ...base, pos: B.pos, vel: B.vel, predOut: B.predA, pend: B.pend });
    bg.gridClear = bindKernel(d, K.gridClear, { ...base, grid: B.grid });
    bg.gridInsert = bindKernel(d, K.gridInsert, { ...base, pred: B.predA, grid: B.grid, ctl: B.ctl });
    bg.sink = bindKernel(d, K.sink, { ...base, pos: B.pos, pred: B.predA, ctl: B.ctl, freeStack: B.freeStack });
    bg.solveAB = bindKernel(d, K.solve, { ...base, pos: B.pos, predIn: B.predA, predOut: B.predB, grid: B.grid });
    bg.solveBA = bindKernel(d, K.solve, { ...base, pos: B.pos, predIn: B.predB, predOut: B.predA, grid: B.grid });
    bg.finalizeA = bindKernel(d, K.finalize, { ...base, pos: B.pos, vel: B.vel, pred: B.predA, ctl: B.ctl });
    bg.finalizeB = bindKernel(d, K.finalize, { ...base, pos: B.pos, vel: B.vel, pred: B.predB, ctl: B.ctl });
    bg.grainEmit = bindKernel(d, K.grainEmit, { ...base, grains: B.grains, gvel: B.gvel, ctl: B.ctl, dep: B.dep });
    bg.grainStep = bindKernel(d, K.grainStep, {
      ...base,
      grains: B.grains,
      gvel: B.gvel,
      pos: B.pos,
      grid: B.grid,
      dep: B.dep,
      ctl: B.ctl,
      pend: B.pend,
    });
    bg.massSum = bindKernel(d, K.massSum, { ...base, pos: B.pos, ctl: B.ctl });
    bg.hist = bindKernel(d, K.hist, { ...base, O: B.opH, pos: B.pos, hist: B.hist });
    const fx = { fxPos: B.fxPos, fxVel: B.fxVel };
    bg.condense = bindKernel(d, K.condense, { ...base, O: B.opC, pos: B.pos, vel: B.vel, freeStack: B.freeStack, ctl: B.ctl, ...fx });
    bg.dissolve = bindKernel(d, K.dissolve, { ...base, O: B.opD, pos: B.pos, vel: B.vel, freeStack: B.freeStack, ctl: B.ctl, ...fx });
    bg.place = bindKernel(d, K.place, { ...base, O: B.opP, items: B.items, pos: B.pos, vel: B.vel, freeStack: B.freeStack, ctl: B.ctl });
    bg.fxStep = bindKernel(d, K.fxStep, { ...base, O: B.opF, ...fx });
    bg.mergePick = bindKernel(d, K.mergePick, { ...base, pos: B.pos, grid: B.grid, recv: B.recv });
    bg.mergeApply = bindKernel(d, K.mergeApply, { ...base, pos: B.pos, recv: B.recv, pend: B.pend, freeStack: B.freeStack, ctl: B.ctl });
    bg.spawn = bindKernel(d, K.spawn, {
      ...base,
      dep: B.dep,
      grid: B.grid,
      pos: B.pos,
      vel: B.vel,
      freeStack: B.freeStack,
      ctl: B.ctl,
    });
  }

  /** 重置全部状态，放入给定的颗粒团（每个 4 个 f32：x, y, z, 质量 mg） */
  seed(clumps: Float32Array) {
    const n = clumps.length / 4;
    const cap = this.p.clumpCap;
    if (n > cap) throw new Error(`颗粒团 ${n} 超过容量 ${cap}`);
    const q = this.device.queue;
    const pos = new Float32Array(cap * 4);
    pos.set(clumps);
    for (let i = n; i < cap; i++) pos.set([0, 1000, 0, 0], i * 4);
    q.writeBuffer(this.buf.pos, 0, pos);
    q.writeBuffer(this.buf.predA, 0, pos);
    q.writeBuffer(this.buf.predB, 0, pos);
    q.writeBuffer(this.buf.vel, 0, new Float32Array(cap * 4));
    // 空闲栈：栈顶弹出的是最小的空闲下标
    const free = new Uint32Array(cap);
    const nFree = cap - n;
    for (let k = 0; k < nFree; k++) free[k] = cap - 1 - k;
    q.writeBuffer(this.buf.freeStack, 0, free);
    const ctl = new Uint32Array(256);
    ctl[0] = nFree;
    new Float32Array(ctl.buffer)[20] = 1;
    q.writeBuffer(this.buf.ctl, 0, ctl);
    const zeroGrains = new Float32Array(this.p.grainCap * 4);
    q.writeBuffer(this.buf.grains, 0, zeroGrains);
    q.writeBuffer(this.buf.gvel, 0, zeroGrains);
    q.writeBuffer(this.buf.dep, 0, new Uint32Array(this.nCells * 4));
    q.writeBuffer(this.buf.pend, 0, new Uint32Array(cap));
    q.writeBuffer(this.buf.fxPos, 0, new Float32Array(FX_CAP * 4));
    this.prevKin = null;
    this.resetStats = true;
  }

  /** 由有效重力求孔口状态：上游方向和 Beverloo 流量（mg/s） */
  static orifice(gLocal: [number, number, number]): { upstream: number; rate: number; gAxial: number } {
    const gAx = -gLocal[1];
    const a = Math.abs(gAx);
    if (a < 1e-3) return { upstream: 0, rate: 0, gAxial: 0 };
    return { upstream: Math.sign(gAx), rate: beverlooRate(a), gAxial: a };
  }

  private writeUniforms(f: FrameInput) {
    const p = this.p;
    const n = Math.min(f.substeps, MAX_SUBSTEPS);
    const prev = this.prevKin ?? { g: f.gLocal, w: f.omega, a: f.alpha };
    const lerp = (a: number[], b: number[], t: number) => a.map((x, k) => x + (b[k] - x) * t) as [number, number, number];
    // 槽位 0..n-1 给各子步，槽位 n 给每帧一次的内核（合并），取帧末的值
    for (let s = 0; s <= n; s++) {
      const t = Math.min((s + 1) / n, 1);
      const g = lerp(prev.g, f.gLocal, t);
      const w = lerp(prev.w, f.omega, t);
      const a = lerp(prev.a, f.alpha, t);
      const fv = new Float32Array(this.uData, s * U_STRIDE, U_WORDS);
      const uv = new Uint32Array(this.uData, s * U_STRIDE, U_WORDS);
      const dt = f.wallDt / n;
      const o = SandSim.orifice(g);
      const flowing = f.flow && o.upstream !== 0;
      fv.set([...g, dt], 0);
      fv.set([...w, p.rc], 4);
      fv.set([...a, this.mc], 8);
      fv.set([...this.gridMin, this.cell], 12);
      uv.set([...this.dims, p.clumpCap], 16);
      const drainDt = (f.drainDt ?? f.wallDt) / n;
      fv.set([flowing ? o.rate * drainDt : 0, flowing ? o.upstream : 0, GRAIN_MASS, p.captureR * p.rc], 20);
      fv.set([p.muS, p.muK, p.muWall, p.sleepV], 24);
      fv.set([p.shockK, Math.sqrt(Math.max(o.gAxial, 0.5) * ORIFICE_D), GRAIN_D / 2, p.sor], 28);
      uv.set([p.grainCap, p.average ? 1 : 0, 0, 0], 32);
      fv.set([p.vmax, p.grainSettleV, f.flushDeposits ? 1 : 0, 0], 36);
    }
    this.prevKin = { g: [...f.gLocal], w: [...f.omega], a: [...f.alpha] };
    this.device.queue.writeBuffer(this.buf.U, 0, this.uData, 0, U_STRIDE * (n + 1));
  }

  /** 编码一帧（若干子步）。调用者负责 submit，然后调用 afterSubmit()。 */
  encodeFrame(encoder: GPUCommandEncoder, f: FrameInput) {
    this.writeUniforms(f);
    if (this.resetStats) {
      this.device.queue.writeBuffer(this.buf.ctl, 8 * 4, new Uint32Array(12));
      this.device.queue.writeBuffer(this.buf.ctl, 24 * 4, new Uint32Array(4));
      this.resetStats = false;
    }
    const cap = this.p.clumpCap;
    const pass = encoder.beginComputePass({ label: 'sand' });
    let slot = 0;
    const run = (k: string, bg: string, n: number) => {
      pass.setPipeline(this.kernels[k].pipeline);
      pass.setBindGroup(0, this.bg[bg], [slot * U_STRIDE]);
      pass.dispatchWorkgroups(groups(n));
    };
    const nSub = Math.min(f.substeps, MAX_SUBSTEPS);
    for (let s = 0; s < nSub; s++) {
      slot = s;
      run('predict', 'predict', cap);
      run('gridClear', 'gridClear', this.nCells);
      run('gridInsert', 'gridInsert', cap);
      pass.setPipeline(this.kernels.sink.pipeline);
      pass.setBindGroup(0, this.bg.sink, [slot * U_STRIDE]);
      pass.dispatchWorkgroups(1);
      let inA = true;
      for (let it = 0; it < this.p.iterations; it++) {
        run('solve', inA ? 'solveAB' : 'solveBA', cap);
        inA = !inA;
      }
      run('finalize', inA ? 'finalizeA' : 'finalizeB', cap);
      run('grainEmit', 'grainEmit', 1024);
      run('grainStep', 'grainStep', this.p.grainCap);
      run('spawn', 'spawn', this.nCells);
    }
    // 每帧一次：合并互相接触的未满颗粒团
    slot = nSub;
    run('mergePick', 'mergePick', cap);
    run('mergeApply', 'mergeApply', cap);
    run('massSum', 'massSum', cap);
    this.writeOp('opF', [0, 0, 0, 0], [0, f.wallDt, 0, 0], [0, 1, 0, 0], [0, FX_CAP, 0, 0]);
    run('fxStep', 'fxStep', FX_CAP);
    pass.end();
    const stage = this.staging.find((s) => !s.busy);
    this.frameNo++;
    if (stage) {
      encoder.copyBufferToBuffer(this.buf.ctl, 0, stage.buf, 0, STATS_WORDS * 4);
      stage.busy = true;
      this.resetStats = true;
      this.toMap.push({ slot: stage, frame: this.frameNo });
    }
  }

  /** 诊断：一个子步里的每个调度单独提交并计时 */
  async timedSubstep(f: FrameInput): Promise<Array<[string, number]>> {
    this.writeUniforms({ ...f, substeps: 1 });
    this.prevKin = null;
    const cap = this.p.clumpCap;
    const order: Array<[string, string, number]> = [
      ['predict', 'predict', cap],
      ['gridClear', 'gridClear', this.nCells],
      ['gridInsert', 'gridInsert', cap],
      ['sink', 'sink', 1],
      ['solve', 'solveAB', cap],
      ['solve', 'solveBA', cap],
      ['finalize', 'finalizeA', cap],
      ['grainEmit', 'grainEmit', 1024],
      ['grainStep', 'grainStep', this.p.grainCap],
      ['spawn', 'spawn', this.nCells],
    ];
    const out: Array<[string, number]> = [];
    for (const [k, bg, n] of order) {
      const t0 = performance.now();
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.kernels[k].pipeline);
      pass.setBindGroup(0, this.bg[bg], [0]);
      pass.dispatchWorkgroups(k === 'sink' ? 1 : groups(n));
      pass.end();
      this.device.queue.submit([enc.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      out.push([bg, performance.now() - t0]);
      console.log(`[bench] ${bg} ${(performance.now() - t0).toFixed(1)} ms`);
    }
    return out;
  }

  private writeOp(name: string, a: number[], b: number[], up: number[], r: number[]) {
    this.device.queue.writeBuffer(this.buf[name], 0, new Float32Array([...a, ...b, ...up, ...r]));
  }

  private dispatchOp(encoder: GPUCommandEncoder, k: string, n: number) {
    const pass = encoder.beginComputePass({ label: k });
    pass.setPipeline(this.kernels[k].pipeline);
    pass.setBindGroup(0, this.bg[k], [0]);
    pass.dispatchWorkgroups(groups(n));
    pass.end();
  }

  /** 凝聚：在 [y0, y1] 的空腔里随机生成 count 个颗粒团 */
  condense(encoder: GPUCommandEncoder, op: CondenseOp) {
    if (op.count <= 0) return;
    this.writeOp('opC', [op.y0, op.y1, op.bulb, op.count], [Math.random() * 1e9, op.time, op.glow ? 1 : 0, op.massEach], [...op.up, 0], [0, FX_CAP, 0, 0]);
    this.dispatchOp(encoder, 'condense', op.count);
  }

  /** 消散：去掉 bulb 泡里沿 up 方向高度超过 threshold 的颗粒团 */
  dissolve(encoder: GPUCommandEncoder, bulb: 1 | -1, up: [number, number, number], threshold: number, fx: boolean, time: number) {
    this.writeOp('opD', [threshold, 0, bulb, 0], [Math.random() * 1e9, time, fx ? 1 : 0, 0], [...up, 0], [0, FX_CAP, 0, 0]);
    this.dispatchOp(encoder, 'dissolve', this.p.clumpCap);
  }

  /** 消散并精确读回实际去掉的质量（mg），离开画面补算时用 */
  async dissolveMeasured(bulb: 1 | -1, up: [number, number, number], threshold: number): Promise<number> {
    const d = this.device;
    d.queue.writeBuffer(this.buf.ctl, 26 * 4, new Uint32Array([0]));
    const e = d.createCommandEncoder();
    this.dissolve(e, bulb, up, threshold, false, 0);
    const st = d.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    e.copyBufferToBuffer(this.buf.ctl, 26 * 4, st, 0, 4);
    d.queue.submit([e.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const mg = new Uint32Array(st.getMappedRange())[0] / 1000;
    st.unmap();
    st.destroy();
    return mg;
  }

  /** 按给定的位置和质量放置颗粒团（每个 4 个 f32） */
  place(encoder: GPUCommandEncoder, items: Float32Array) {
    const n = Math.min(items.length / 4, PLACE_CAP);
    if (n <= 0) return;
    this.device.queue.writeBuffer(this.buf.items, 0, items, 0, n * 4);
    this.writeOp('opP', [0, 0, 0, n], [0, 0, 0, 0], [0, 1, 0, 0], [0, FX_CAP, 0, 0]);
    this.dispatchOp(encoder, 'place', n);
  }

  /** 读回一个泡里颗粒团沿 up 方向的高度直方图（µg/格），区间 [hmin, hmax] */
  async histogram(bulb: 1 | -1, up: [number, number, number], hmin: number, hmax: number): Promise<Uint32Array> {
    const d = this.device;
    d.queue.writeBuffer(this.buf.hist, 0, new Uint32Array(HIST_BINS));
    this.writeOp('opH', [0, 0, bulb, 0], [0, 0, 0, 0], [...up, hmin], [hmax, FX_CAP, 0, 0]);
    const e = d.createCommandEncoder();
    this.dispatchOp(e, 'hist', this.p.clumpCap);
    e.copyBufferToBuffer(this.buf.hist, 0, this.histStage, 0, HIST_BINS * 4);
    d.queue.submit([e.finish()]);
    await this.histStage.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(this.histStage.getMappedRange().slice(0));
    this.histStage.unmap();
    return out;
  }

  private toMap: Array<{ slot: { buf: GPUBuffer; busy: boolean }; frame: number }> = [];
  private chain: Promise<void> = Promise.resolve();

  /** submit 之后调用：异步读回本帧统计，按提交顺序回调 onStats */
  afterSubmit() {
    const jobs = this.toMap;
    this.toMap = [];
    for (const job of jobs) {
      const mapped = job.slot.buf.mapAsync(GPUMapMode.READ);
      this.chain = this.chain.then(async () => {
        await mapped;
        const u = new Uint32Array(job.slot.buf.getMappedRange().slice(0));
        job.slot.buf.unmap();
        job.slot.busy = false;
        const f = new Float32Array(u.buffer);
        this.onStats?.({
          frame: job.frame,
          drained: f[8],
          emitted: u[9],
          impactsSand: u[10],
          impactsGlass: u[11],
          impactSpeedSum: u[12] / 1000,
          maxSpeed: u[13] / 10000,
          spawnFail: u[14],
          deposited: u[15],
          available: f[16],
          capOverflow: u[18],
          forced: u[19],
          merged: u[17],
          massPos: u[24] / 1000,
          massNeg: u[25] / 1000,
          dissolved: u[26] / 1000,
          placedFail: u[27],
          clumpsAlive: this.p.clumpCap - u[0],
        });
      });
    }
  }

  /** 等所有已提交帧的统计都读回 */
  async flush() {
    await this.chain;
  }

  /** 读回全部颗粒团和沙粒（测试用） */
  async readState(): Promise<{ pos: Float32Array; grains: Float32Array; dep: Uint32Array; pend: Uint32Array }> {
    const d = this.device;
    const read = async (src: GPUBuffer) => {
      const b = d.createBuffer({ size: src.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = d.createCommandEncoder();
      e.copyBufferToBuffer(src, 0, b, 0, src.size);
      d.queue.submit([e.finish()]);
      await b.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(b.getMappedRange().slice(0));
      b.unmap();
      b.destroy();
      return out;
    };
    return {
      pos: await read(this.buf.pos),
      grains: await read(this.buf.grains),
      dep: new Uint32Array((await read(this.buf.dep)).buffer),
      pend: new Uint32Array((await read(this.buf.pend)).buffer),
    };
  }
}

/** 诊断：逐个编译所有物理内核并打印耗时 */
export async function diagnoseCompile(device: GPUDevice) {
  const U: BindSpec = { name: 'U', kind: 'uniform', type: 'SimU' };
  const P: BindSpec = { name: 'P', kind: 'uniform', type: 'Profile' };
  const v4 = (name: string, kind: 'read' | 'rw'): BindSpec => ({ name, kind, type: 'array<vec4f>' });
  const atom = (name: string): BindSpec => ({ name, kind: 'rw', type: 'array<atomic<u32>>' });
  const u32r = (name: string, kind: 'read' | 'rw' = 'read'): BindSpec => ({ name, kind, type: 'array<u32>' });
  const parts = (body: string, extra = '') => ({ types: typesSrc, fns: fnsSrc + '\n' + extra, body });
  const list: Array<[string, ReturnType<typeof parts>, BindSpec[]]> = [
    ['gridClear', parts(gridClearSrc), [U, P, atom('grid')]],
    ['predict', parts(predictSrc), [U, P, v4('pos', 'read'), v4('vel', 'read'), v4('predOut', 'rw')]],
    ['gridInsert', parts(gridInsertSrc), [U, P, v4('pred', 'read'), atom('grid'), atom('ctl')]],
    ['finalize', parts(finalizeSrc), [U, P, v4('pos', 'rw'), v4('vel', 'rw'), v4('pred', 'read'), atom('ctl')]],
    ['spawn', parts(spawnSrc), [U, P, atom('dep'), u32r('grid'), v4('pos', 'rw'), v4('vel', 'rw'), u32r('freeStack'), atom('ctl')]],
    ['grainEmit', parts(grainEmitSrc, depositSrc), [U, P, v4('grains', 'rw'), v4('gvel', 'rw'), atom('ctl'), atom('dep')]],
    ['sink', parts(sinkSrc), [U, P, v4('pos', 'rw'), v4('pred', 'rw'), atom('ctl'), u32r('freeStack', 'rw')]],
    ['solve', parts(solveSrc), [U, P, v4('pos', 'read'), v4('predIn', 'read'), v4('predOut', 'rw'), u32r('grid')]],
    ['grainStep', parts(grainStepSrc, depositSrc), [U, P, v4('grains', 'rw'), v4('gvel', 'rw'), v4('pos', 'read'), u32r('grid'), atom('dep'), atom('ctl')]],
  ];
  for (const [name, pr, specs] of list) {
    const ms = await timeKernelCompile(device, name, pr, specs);
    console.log(`[compile] ${name} ${ms.toFixed(0)} ms`);
  }
}

/** CPU 版内腔 SDF（与 fns.wgsl 的 interiorSdf 一致），摆放初始沙量用 */
export function interiorSdfCpu(x: number, y: number, z: number): number {
  const ay = Math.abs(y);
  if (ay > BULB_H) return BULB_H - ay;
  const [r, s] = innerRadius(ay);
  const rho = Math.hypot(x, z);
  return Math.min((r - rho) / Math.sqrt(1 + s * s), BULB_H - ay);
}

/**
 * 在一个玻璃泡里按重力方向从低到高摆放总质量为 massMg 的颗粒团（带抖动的立方点阵）。
 * bulb = +1 为 y > 0 的泡，-1 为 y < 0 的泡；upY 为局部系里"向上"的 y 符号。
 */
export function seedBulb(rc: number, massMg: number, bulb: 1 | -1, upY: 1 | -1, jitterSeed = 1): Float32Array {
  const mc = clumpMass(rc);
  const s = 2 * rc * 1.02;
  const pts: number[][] = [];
  let rng = jitterSeed >>> 0 || 1;
  const rand = () => {
    rng ^= rng << 13;
    rng ^= rng >>> 17;
    rng ^= rng << 5;
    return (rng >>> 0) / 4294967296;
  };
  const maxR = 0.04;
  for (let y = rc; y < BULB_H; y += s) {
    for (let x = -maxR; x <= maxR; x += s) {
      for (let z = -maxR; z <= maxR; z += s) {
        const px = x + (rand() - 0.5) * 0.1 * rc;
        const pz = z + (rand() - 0.5) * 0.1 * rc;
        const py = bulb * y;
        if (interiorSdfCpu(px, py, pz) > rc * 1.05) pts.push([px, py, pz]);
      }
    }
  }
  pts.sort((a, b) => upY * (a[1] - b[1]));
  const out: number[] = [];
  let left = massMg;
  for (const p of pts) {
    if (left <= 0) break;
    const m = Math.min(mc, left);
    out.push(p[0], p[1], p[2], m);
    left -= m;
  }
  if (left > 1e-6) throw new Error(`玻璃泡装不下 ${massMg} mg 的沙`);
  return new Float32Array(out);
}
