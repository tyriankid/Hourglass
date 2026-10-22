import { SandClock, formatClock } from './clock/SandClock';
import { DebugRenderer } from './debug/DebugRenderer';
import { CAP_T, BULB_H } from './geometry/profile';
import { rigid } from './math/mat4';
import { Renderer } from './render/Renderer';
import { App } from './app/App';
import { loadSettings } from './ui/Overlay';
import { runProbe } from './debug/probes';
import { G0, GRAIN_MASS, W0, sandMassForSeconds } from './physics';
import { SandSim, defaultParams, seedBulb, type FrameInput, type FrameStats } from './sim/SandSim';

const params = new URLSearchParams(location.search);
const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || (navigator.maxTouchPoints > 1 && /Mac/.test(navigator.userAgent));

function fail(msg: string) {
  const el = document.createElement('div');
  el.style.cssText =
    'position:fixed;inset:0;display:flex;align-items:center;justify-content:center;padding:2em;font:16px/1.6 system-ui;color:#555;text-align:center';
  el.textContent = msg;
  document.body.appendChild(el);
  (window as any).__result = { error: msg };
}

async function initGpu(): Promise<GPUDevice | null> {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const want = 256 * 1024 * 1024;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, want),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, want),
    },
  });
  device.lost.then((info) => console.error('GPU 设备丢失', info.message));
  device.addEventListener('uncapturederror', (e) => {
    const msg = (e as GPUUncapturedErrorEvent).error.message;
    console.error('GPU 错误', msg);
    // 测试模式下 GPU 出错立即失败，不空跑
    if (params.get('mode') === 'drain' && !(window as any).__result) (window as any).__result = { error: msg };
  });
  return device;
}

function rotZ(g: [number, number, number], a: number): [number, number, number] {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [c * g[0] - s * g[1], s * g[0] + c * g[1], g[2]];
}

async function runDebug(device: GPUDevice, rc: number, minutes: number) {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const total = sandMassForSeconds(minutes * 60);
  const sim = new SandSim(device, defaultParams(rc, sandMassForSeconds(3600)));
  const renderer = new DebugRenderer(device, canvas, sim);
  const hud = document.createElement('pre');
  hud.style.cssText =
    'position:fixed;left:8px;top:8px;margin:0;padding:8px 10px;background:rgba(255,255,255,.75);font:12px/1.45 ui-monospace,monospace;color:#333;border-radius:6px;pointer-events:none';
  document.body.appendChild(hud);

  let clock = new SandClock(total);
  let simTime = 0;
  let flowAt = 1.5;
  let tilt = 0;
  let flipped = false;
  let maxSpeed = 0;
  let last: FrameStats | null = null;
  let emittedWin = 0;
  let winStart = performance.now();
  let emitRate = 0;
  const reseed = () => {
    sim.seed(seedBulb(rc, total, 1, 1, Math.floor(Math.random() * 1e9)));
    clock = new SandClock(total);
    simTime = 0;
    flowAt = 1.5;
    tilt = 0;
    flipped = false;
  };
  reseed();
  sim.onStats = (s) => {
    last = s;
    maxSpeed = s.maxSpeed;
    clock.add(s.drained);
    emittedWin += s.emitted;
  };

  let dragging = false;
  let lx = 0;
  let ly = 0;
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    lx = e.clientX;
    ly = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', () => (dragging = false));
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    renderer.yaw -= (e.clientX - lx) * 0.006;
    renderer.pitch = Math.max(-1.2, Math.min(1.2, renderer.pitch + (e.clientY - ly) * 0.006));
    lx = e.clientX;
    ly = e.clientY;
  });
  canvas.addEventListener('wheel', (e) => {
    renderer.distance = Math.max(0.08, Math.min(1.2, renderer.distance * Math.exp(e.deltaY * 0.001)));
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') tilt += 0.1;
    if (e.key === 'ArrowRight') tilt -= 0.1;
    if (e.key === 'f') flipped = !flipped;
    if (e.key === 'r') reseed();
    if (e.key === '0') tilt = 0;
  });

  let prev = performance.now();
  let fps = 60;
  let inFlight = 0;
  const frame = (now: number) => {
    if (inFlight >= 2) {
      requestAnimationFrame(frame);
      return;
    }
    const wallDt = Math.min((now - prev) / 1000, 0.1);
    prev = now;
    fps = fps * 0.95 + (1 / Math.max(wallDt, 1e-3)) * 0.05;
    const byStability = Math.ceil(wallDt / 0.0021);
    const byMotion = Math.ceil((maxSpeed * wallDt) / (0.5 * rc));
    const substeps = Math.max(4, Math.min(16, Math.max(byStability, byMotion)));
    const g = rotZ([0, -G0, 0], tilt + (flipped ? Math.PI : 0));
    const input: FrameInput = {
      wallDt,
      substeps,
      gLocal: g,
      omega: [0, 0, 0],
      alpha: [0, 0, 0],
      flow: simTime >= flowAt,
    };
    simTime += wallDt;
    const enc = device.createCommandEncoder();
    sim.encodeFrame(enc, input);
    renderer.encode(enc);
    device.queue.submit([enc.finish()]);
    sim.afterSubmit();
    inFlight++;
    device.queue.onSubmittedWorkDone().then(() => inFlight--);

    if (now - winStart > 1000) {
      emitRate = (emittedWin * 1000) / (now - winStart);
      emittedWin = 0;
      winStart = now;
    }
    const s = last as FrameStats | null;
    hud.textContent = [
      `颗粒团半径 ${(rc * 1000).toFixed(2)} mm   设定 ${minutes} 分钟 = ${(total / 1000).toFixed(2)} g`,
      `FPS ${fps.toFixed(0)}   子步 ${substeps}   最大速度 ${maxSpeed.toFixed(3)} m/s`,
      `剩余 ${formatClock(clock.remainingSeconds)}   已流 ${(clock.drained / 1000).toFixed(3)} g`,
      `沙粒 ${emitRate.toFixed(0)} 粒/s（理论 ${(W0 / GRAIN_MASS).toFixed(0)}）`,
      s ? `颗粒团 ${s.clumpsAlive}   捕获区 ${s.available.toFixed(1)} mg   溢出 ${s.capOverflow}  池满 ${s.spawnFail}  强制沉积 ${s.forced}` : '',
      `倾斜 ${((tilt * 180) / Math.PI).toFixed(0)}°${flipped ? '  已翻转' : ''}   ←/→ 倾斜  F 翻转  R 重置  0 回正`,
    ].join('\n');
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  (window as any).__sim = sim;
}

/** 正式画面（M1 阶段：立着的沙漏，上泡的沙自然流下） */
async function runView(device: GPUDevice, rc: number, minutes: number) {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const total = sandMassForSeconds(minutes * 60);
  const sim = new SandSim(device, defaultParams(rc, sandMassForSeconds(3600)));
  if (params.get('cmode')) Renderer.compositeMode = Number(params.get('cmode'));
  const renderer = new Renderer(device, canvas, sim);
  renderer.model = rigid(new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), [0, CAP_T + BULB_H, 0]);
  if (params.get('scale')) renderer.renderScale = Number(params.get('scale'));
  if (params.get('upto')) renderer.upto = Number(params.get('upto'));
  const hud = document.createElement('div');
  hud.style.cssText =
    'position:fixed;left:0;right:0;bottom:6vh;text-align:center;font:300 28px/1 system-ui,-apple-system,sans-serif;letter-spacing:.08em;color:rgba(60,58,55,.75);pointer-events:none';
  document.body.appendChild(hud);
  const clock = new SandClock(total);
  sim.seed(seedBulb(rc, total, 1, 1, 99));
  let maxSpeed = 0;
  sim.onStats = (s) => {
    maxSpeed = s.maxSpeed;
    if (simTime > 1.0) clock.add(s.drained);
  };
  let dragging = false;
  let lx = 0;
  let ly = 0;
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    lx = e.clientX;
    ly = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', () => (dragging = false));
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    renderer.camera.yaw -= (e.clientX - lx) * 0.005;
    renderer.camera.pitch = Math.max(-0.2, Math.min(0.9, renderer.camera.pitch + (e.clientY - ly) * 0.005));
    lx = e.clientX;
    ly = e.clientY;
  });
  canvas.addEventListener('wheel', (e) => {
    renderer.camera.distance = Math.max(0.25, Math.min(1.0, renderer.camera.distance * Math.exp(e.deltaY * 0.001)));
  });
  if (params.get('yaw')) renderer.camera.yaw = Number(params.get('yaw'));
  if (params.get('pitch')) renderer.camera.pitch = Number(params.get('pitch'));
  if (params.get('dist')) renderer.camera.distance = Number(params.get('dist'));
  let simTime = 0;
  let prev = performance.now();
  let inFlight = 0;
  // 测试用：?frames=N 逐帧同步跑 N 帧（每帧 1/60 s 模拟时间）后截图；?settle=秒 先不渲染只跑物理
  const fixedFrames = Number(params.get('frames') ?? 0);
  if (fixedFrames > 0) {
    const settle = Number(params.get('settle') ?? 0);
    const t0 = performance.now();
    for (let i = 0; i < settle * 60; i++) {
      const enc = device.createCommandEncoder();
      sim.encodeFrame(enc, { wallDt: 1 / 60, substeps: 8, gLocal: [0, -G0, 0], omega: [0, 0, 0], alpha: [0, 0, 0], flow: simTime > 1.0 });
      simTime += 1 / 60;
      device.queue.submit([enc.finish()]);
      sim.afterSubmit();
      await sim.flush();
    }
    const t1 = performance.now();
    for (let i = 0; i < fixedFrames; i++) {
      const enc = device.createCommandEncoder();
      sim.encodeFrame(enc, { wallDt: 1 / 60, substeps: 8, gLocal: [0, -G0, 0], omega: [0, 0, 0], alpha: [0, 0, 0], flow: simTime > 1.0 });
      simTime += 1 / 60;
      renderer.encode(enc, simTime);
      device.queue.submit([enc.finish()]);
      sim.afterSubmit();
      await sim.flush();
      hud.textContent = formatClock(clock.remainingSeconds);
    }
    (window as any).__result = { settleSeconds: (t1 - t0) / 1000, renderSecondsPerFrame: (performance.now() - t1) / 1000 / fixedFrames };
    return;
  }
  const frame = (now: number) => {
    requestAnimationFrame(frame);
    if (inFlight >= 2) return;
    const wallDt = Math.min((now - prev) / 1000, 0.1);
    prev = now;
    const substeps = Math.max(4, Math.min(16, Math.max(Math.ceil(wallDt / 0.0021), Math.ceil((maxSpeed * wallDt) / (0.5 * rc)))));
    const enc = device.createCommandEncoder();
    sim.encodeFrame(enc, { wallDt, substeps, gLocal: [0, -G0, 0], omega: [0, 0, 0], alpha: [0, 0, 0], flow: simTime > 1.0 });
    simTime += wallDt;
    renderer.encode(enc, now / 1000);
    device.queue.submit([enc.finish()]);
    sim.afterSubmit();
    inFlight++;
    device.queue.onSubmittedWorkDone().then(() => inFlight--);
    hud.textContent = formatClock(clock.remainingSeconds);
  };
  requestAnimationFrame(frame);
  (window as any).__sim = sim;
  (window as any).__renderer = renderer;
}

/** 排空测试：用模拟时间尽快跑，测沙流完的用时和质量守恒 */
async function runDrainTest(device: GPUDevice, rc: number, minutes: number) {
  const total = sandMassForSeconds(minutes * 60);
  const p = defaultParams(rc, total);
  const sim = new SandSim(device, p);
  sim.seed(seedBulb(rc, total, 1, 1, 12345));
  const clock = new SandClock(total);
  const fps = 60;
  const dt = 1 / fps;
  let frames = 0;
  let flowing = false;
  let flowFrames = 0;
  let doneAt = -1;
  let stats: FrameStats | null = null;
  let spawnFail = 0;
  let forced = 0;
  let capOverflow = 0;
  let maxClumps = 0;
  let merged = 0;
  const starvation: number[] = [];
  sim.onStats = (s) => {
    stats = s;
    maxClumps = Math.max(maxClumps, s.clumpsAlive);
    merged += s.merged;
    spawnFail += s.spawnFail;
    forced += s.forced;
    capOverflow += s.capOverflow;
    if (!flowing) return;
    const before = clock.drained;
    clock.add(s.drained);
    const expected = W0 * dt;
    if (doneAt < 0 && s.drained < expected * 0.98 && clock.drained < total * 0.999) starvation.push(flowFrames);
    if (doneAt < 0 && clock.done) {
      const need = total - before;
      const frac = s.drained > 0 ? Math.min(1, need / s.drained) : 1;
      doneAt = (flowFrames - 1 + frac) * dt;
    }
  };
  let flushing = false;
  const step = async (flow: boolean) => {
    const dbg = frames < 2;
    if (dbg) console.log(`[drain] step ${frames} encode`);
    const enc = device.createCommandEncoder();
    sim.encodeFrame(enc, { wallDt: dt, substeps: 8, gLocal: [0, -G0, 0], omega: [0, 0, 0], alpha: [0, 0, 0], flow, flushDeposits: flushing });
    device.queue.submit([enc.finish()]);
    if (dbg) console.log(`[drain] step ${frames} submitted`);
    if (dbg) await device.queue.onSubmittedWorkDone();
    if (dbg) console.log(`[drain] step ${frames} gpu done`);
    sim.afterSubmit();
    frames++;
    if (flow) flowFrames++;
    await sim.flush();
    if (dbg) console.log(`[drain] step ${frames - 1} stats read`);
  };
  const t0 = performance.now();
  for (let i = 0; i < 2 * fps; i++) {
    await step(false);
    if (i < 3 || i % 30 === 0) console.log(`[drain] 沉降帧 ${i} 累计耗时 ${((performance.now() - t0) / 1000).toFixed(2)}s`);
  }
  flowing = true;
  const limit = (minutes * 60 + 30) * fps;
  while (doneAt < 0 && flowFrames < limit && !(window as any).__result) {
    await step(true);
    if (flowFrames % (5 * fps) === 0) {
      const s = stats as FrameStats | null;
      console.log(
        `[drain] t=${(flowFrames * dt).toFixed(0)}s 已流 ${(clock.drained / 1000).toFixed(3)}/${(total / 1000).toFixed(3)} g 颗粒团 ${s?.clumpsAlive} 捕获区 ${s?.available.toFixed(1)} mg 耗时 ${((performance.now() - t0) / 1000).toFixed(0)}s`,
      );
    }
  }
  for (let i = 0; i < 3 * fps; i++) await step(true);
  const state = await sim.readState();
  let upper = 0;
  let lower = 0;
  for (let i = 0; i < p.clumpCap; i++) {
    const m = state.pos[i * 4 + 3];
    if (m <= 0) continue;
    if (state.pos[i * 4 + 1] > 0) upper += m;
    else lower += m;
  }
  let inFlight = 0;
  for (let i = 0; i < p.grainCap; i++) if (state.grains[i * 4 + 3] > 0) inFlight++;
  let waiting = 0;
  for (let c = 0; c < sim.nCells; c++) waiting += state.dep[c * 4];
  for (let i = 0; i < p.clumpCap; i++) waiting += state.pend[i];
  const result = {
    minutes,
    rcMm: rc * 1000,
    totalMg: total,
    drainedMg: clock.drained,
    targetSeconds: minutes * 60,
    measuredSeconds: doneAt,
    errorSeconds: doneAt - minutes * 60,
    upperMg: upper,
    lowerMg: lower,
    grainsInFlight: inFlight,
    grainsInFlightMg: inFlight * GRAIN_MASS,
    waitingGrains: waiting,
    massErrorMg: upper + lower + (inFlight + waiting) * GRAIN_MASS - total,
    partialClumps: (() => {
      let n = 0;
      for (let i = 0; i < p.clumpCap; i++) {
        const m = state.pos[i * 4 + 3];
        if (m > 0 && m < sim.mc * 0.999) n++;
      }
      return n;
    })(),
    starvedFrames: starvation.length,
    firstStarvedAt: starvation.length ? starvation[0] * dt : null,
    spawnFail,
    forced,
    capOverflow,
    maxClumps,
    merged,
    fullClumpsEquivalent: total / sim.mc,
    wallSeconds: (performance.now() - t0) / 1000,
  };
  console.log('[drain] result ' + JSON.stringify(result));
  (window as any).__result = result;
}

/** 性能面板（?hud=1）：真机测试时截图发回来就能判断瓶颈 */
function attachHud(app: App, device: GPUDevice) {
  const el = document.createElement('pre');
  el.style.cssText =
    'position:fixed;left:8px;top:8px;margin:0;padding:6px 8px;background:rgba(255,255,255,.7);font:11px/1.4 ui-monospace,monospace;color:#333;border-radius:6px;pointer-events:none;z-index:9';
  document.body.appendChild(el);
  let gpu = '';
  const info = (device as any).adapterInfo ?? null;
  if (info) gpu = [info.vendor, info.architecture, info.description].filter(Boolean).join(' ');
  app.onFrame = (a, s) => {
    el.textContent = [
      `${(1000 / a.frameMs).toFixed(0)} fps  ${a.frameMs.toFixed(1)} ms  子步 ${a.substepsUsed}`,
      `分辨率 ×${a.renderer.renderScale.toFixed(2)}  ${a.renderer.size()}`,
      `颗粒团 ${s?.clumpsAlive ?? '-'} / ${a.sim.p.clumpCap}  r=${(a.rc * 1000).toFixed(2)}mm`,
      `状态 ${a.state}  上泡 ${(s?.massPos ?? 0).toFixed(0)} mg  下泡 ${(s?.massNeg ?? 0).toFixed(0)} mg`,
      gpu ? `GPU ${gpu}` : '',
    ].join('\n');
  };
}

/** 无头测试：按固定步长驱动完整流程，在各阶段请求截图（与 tools/run.mjs 握手） */
async function runAppScenario(app: App) {
  const w = window as any;
  const shot = async (name: string) => {
    console.log(`[scenario] 截图 ${name} ${JSON.stringify(app.debugState)}`);
    w.__shotDone = false;
    w.__shotRequest = name;
    while (!w.__shotDone) await new Promise((r) => setTimeout(r, 100));
  };
  const dt = 1 / 30;
  const until = async (cond: () => boolean, max: number) => {
    for (let i = 0; i < max && !cond(); i++) await app.step(dt);
  };
  const frames = async (n: number) => {
    for (let i = 0; i < n; i++) await app.step(dt);
  };
  await frames(20);
  await until(() => (app as any).ready(), 400);
  await frames(10);
  await shot('1-idle');
  app.tap();
  await frames(20);
  await shot('2-flipping');
  await until(() => app.state === 'running', 200);
  await frames(45);
  await shot('3-running');
  app.tap();
  await frames(18);
  await shot('4-laying');
  await until(() => app.state === 'paused', 200);
  await frames(30);
  await shot('5-paused');
  app.tap();
  await until(() => app.state === 'running', 200);
  await frames(30);
  await shot('6-resumed');
  w.__result = { ok: true, ...app.debugState };
}

/** 无头测试：从翻转一直跑到沙流完，检查完成判定和回到调时间状态（不渲染，只跑物理和状态机） */
async function runFinishScenario(app: App) {
  const w = window as any;
  app.renderer.upto = 0;
  const dt = Number(params.get('dt') ?? 1 / 30);
  const step = () => app.step(dt);
  for (let i = 0; i < 20; i++) await step();
  for (let i = 0; i < 400 && !(app as any).ready(); i++) await step();
  const total = app.debugState;
  app.tap();
  let t = 0;
  let finishedAt = -1;
  let idleAt = -1;
  const target = app.ui.minutes * 60;
  while (t < target + 60) {
    await step();
    t += dt;
    if (finishedAt < 0 && app.state === 'finished') finishedAt = t;
    if (finishedAt >= 0 && app.state === 'idle') {
      idleAt = t;
      break;
    }
    if (Math.round(t / dt) % 300 === 0) console.log(`[finish] t=${t.toFixed(0)}s ${JSON.stringify(app.debugState)}`);
  }
  // 翻转动画 1.35 s 期间沙已经开始流，计时从按下那一刻算起
  w.__result = { target, finishedAt, idleAt, startState: total, end: app.debugState };
}

async function main() {
  const device = await initGpu();
  if (!device) {
    fail('这个浏览器不支持 WebGPU。请使用最新版 Chrome / Edge，或 iOS 26 以上的 Safari。');
    return;
  }
  const mode = params.get('mode') ?? 'app';
  const rc = Number(params.get('rc') ?? (isMobile ? 1.0 : 0.75)) / 1000;
  const minutes = Number(params.get('min') ?? (mode === 'drain' ? 1 : 5));
  if (mode === 'bench') {
    const sim = new SandSim(device, defaultParams(rc, sandMassForSeconds(3600)));
    sim.seed(seedBulb(rc, sandMassForSeconds(minutes * 60), 1, 1, 7));
    await device.queue.onSubmittedWorkDone();
    console.log('[bench] seed done');
    for (let r = 0; r < 2; r++)
      await sim.timedSubstep({ wallDt: 1 / 60, substeps: 1, gLocal: [0, -G0, 0], omega: [0, 0, 0], alpha: [0, 0, 0], flow: true });
    (window as any).__result = { ok: true };
    return;
  }
  if (mode === 'inspect') {
    // 摆好沙后不开孔口跑若干帧，统计颗粒团的高度分布、速度和重叠，检查堆积是否稳定
    const total = sandMassForSeconds(minutes * 60);
    const sp = defaultParams(rc, sandMassForSeconds(3600));
    if (params.get('iters')) sp.iterations = Number(params.get('iters'));
    if (params.get('sor')) sp.sor = Number(params.get('sor'));
    if (params.get('shock')) sp.shockK = Number(params.get('shock'));
    if (params.get('avg')) sp.average = params.get('avg') === '1';
    const sim = new SandSim(device, sp);
    const seeds = seedBulb(rc, total, 1, 1, 99);
    sim.seed(seeds);
    const report = async (label: string) => {
      const st = await sim.readState();
      const ys: number[] = [];
      let n = 0;
      let mass = 0;
      for (let i = 0; i < sim.p.clumpCap; i++) {
        const m = st.pos[i * 4 + 3];
        if (m <= 0) continue;
        n++;
        mass += m;
        ys.push(st.pos[i * 4 + 1]);
      }
      ys.sort((a, b) => a - b);
      const q = (f: number) => (ys[Math.floor(f * (ys.length - 1))] * 1000).toFixed(1);
      // 重叠统计：相互穿透超过两半径之和 5% / 20% 的接触对
      const pts: number[][] = [];
      for (let i = 0; i < sim.p.clumpCap; i++) {
        const m = st.pos[i * 4 + 3];
        if (m > 0) pts.push([st.pos[i * 4], st.pos[i * 4 + 1], st.pos[i * 4 + 2], sim.p.rc * Math.cbrt(m / sim.mc)]);
      }
      let pairs = 0;
      let o5 = 0;
      let o20 = 0;
      let maxO = 0;
      for (let a = 0; a < pts.length; a++)
        for (let b = a + 1; b < pts.length; b++) {
          const dx = pts[a][0] - pts[b][0];
          const dy = pts[a][1] - pts[b][1];
          const dz = pts[a][2] - pts[b][2];
          const rr = pts[a][3] + pts[b][3];
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 >= rr * rr) continue;
          const o = 1 - Math.sqrt(d2) / rr;
          pairs++;
          if (o > 0.05) o5++;
          if (o > 0.2) o20++;
          maxO = Math.max(maxO, o);
        }
      console.log(`[inspect]   接触 ${pairs} 对，重叠>5% ${o5}，>20% ${o20}，最大 ${(maxO * 100).toFixed(0)}%`);
      console.log(`[inspect] ${label} 颗粒团 ${n} 质量 ${(mass / 1000).toFixed(3)} g  y(mm) min ${q(0)} p10 ${q(0.1)} p50 ${q(0.5)} p90 ${q(0.9)} max ${q(1)}`);
    };
    await report('初始');
    let maxV = 0;
    sim.onStats = (s) => (maxV = s.maxSpeed);
    const frames = Number(params.get('frames2') ?? 120);
    for (let f = 1; f <= frames; f++) {
      const enc = device.createCommandEncoder();
      sim.encodeFrame(enc, { wallDt: 1 / 60, substeps: Number(params.get('sub') ?? 8), gLocal: [0, -G0, 0], omega: [0, 0, 0], alpha: [0, 0, 0], flow: false });
      device.queue.submit([enc.finish()]);
      sim.afterSubmit();
      await sim.flush();
      if (f % 30 === 0) {
        await report(`第 ${f} 帧`);
        console.log(`[inspect]   最大速度 ${maxV.toFixed(3)} m/s`);
      }
    }
    (window as any).__result = { ok: true };
    return;
  }
  if (mode === 'onefall') {
    // 最小实验：空中放一颗颗粒团，给定重力方向，看它往哪掉
    const sim = new SandSim(device, defaultParams(rc, sandMassForSeconds(600)));
    const gy = Number(params.get('gy') ?? -1);
    const y0 = Number(params.get('y0') ?? -0.04);
    sim.seed(new Float32Array([0.0, y0, 0.0, sim.mc]));
    const out: string[] = [];
    for (let f = 0; f < 6; f++) {
      const enc = device.createCommandEncoder();
      sim.encodeFrame(enc, { wallDt: 1 / 60, substeps: 8, gLocal: [0, gy * G0, 0], omega: [0, 0, 0], alpha: [0, 0, 0], flow: false });
      device.queue.submit([enc.finish()]);
      sim.afterSubmit();
      await sim.flush();
      const st = await sim.readState();
      out.push((st.pos[1] * 1000).toFixed(3));
    }
    console.log(`[onefall] gy=${gy} y0=${y0 * 1000}mm → y(mm) 每帧: ${out.join(' ')}`);
    (window as any).__result = { out };
    return;
  }
  if (mode === 'restpile') {
    // 下泡摆好 min 分钟的锥形沙堆，沉降 settle 帧后逐帧测量：位移分布、休眠比例
    const rp = defaultParams(rc, sandMassForSeconds(3600));
    if (params.get('shock')) rp.shockK = Number(params.get('shock'));
    if (params.get('iters')) rp.iterations = Number(params.get('iters'));
    const sim = new SandSim(device, rp);
    sim.seed(seedBulb(rc, sandMassForSeconds(minutes * 60), -1, 1, 9, 0.6));
    const settleF = Number(params.get('settle') ?? 300);
    const stepF = async () => {
      const enc = device.createCommandEncoder();
      sim.encodeFrame(enc, { wallDt: 1 / 60, substeps: 8, gLocal: [0, -G0, 0], omega: [0, 0, 0], alpha: [0, 0, 0], flow: true });
      device.queue.submit([enc.finish()]);
      sim.afterSubmit();
      await sim.flush();
    };
    for (let f = 0; f < settleF; f++) {
      await stepF();
      if (f % 60 === 59) {
        const st = await sim.readState();
        let asleep = 0;
        let alive = 0;
        for (let i = 0; i < sim.p.clumpCap; i++) if (st.pos[i * 4 + 3] > 0) { alive++; if (st.rest[i * 4 + 3] < 0) asleep++; }
        console.log(`[restpile] f=${f + 1} 休眠 ${asleep}/${alive}`);
      }
    }
    for (let k = 0; k < 3; k++) {
      const a = (await sim.readState()).pos;
      await stepF();
      const st = await sim.readState();
      const sp: number[] = [];
      let asleep = 0;
      for (let i = 0; i < sim.p.clumpCap; i++) {
        if (st.pos[i * 4 + 3] <= 0) continue;
        if (st.rest[i * 4 + 3] < 0) asleep++;
        sp.push(Math.hypot(st.pos[i * 4] - a[i * 4], st.pos[i * 4 + 1] - a[i * 4 + 1], st.pos[i * 4 + 2] - a[i * 4 + 2]) * 60000);
      }
      // 按"离沙面的深度"分层：每层里速度 > 1 mm/s 的比例、平均速度方向（竖直分量）
      const rows: Array<{ v: number; y: number; r: number; vy: number }> = [];
      for (let i = 0; i < sim.p.clumpCap; i++) {
        if (st.pos[i * 4 + 3] <= 0) continue;
        rows.push({ v: Math.hypot(st.pos[i * 4] - a[i * 4], st.pos[i * 4 + 1] - a[i * 4 + 1], st.pos[i * 4 + 2] - a[i * 4 + 2]) * 60000, y: st.pos[i * 4 + 1], r: Math.hypot(st.pos[i * 4], st.pos[i * 4 + 2]), vy: (st.pos[i * 4 + 1] - a[i * 4 + 1]) * 60000 });
      }
      // 沙面高度近似：同一离轴距离环带里的最高点
      const ring = new Map<number, number>();
      for (const r of rows) { const k = Math.floor(r.r / 0.003); ring.set(k, Math.max(ring.get(k) ?? -1, r.y)); }
      const layers = new Map<number, { n: number; mv: number; vy: number }>();
      for (const r of rows) {
        const depth = (ring.get(Math.floor(r.r / 0.003))! - r.y) * 1000;
        const k = Math.min(5, Math.floor(depth / 4));
        const l = layers.get(k) ?? { n: 0, mv: 0, vy: 0 };
        l.n++;
        if (r.v > 1) { l.mv++; l.vy += r.vy; }
        layers.set(k, l);
      }
      const desc = [...layers.entries()].sort((x, y) => x[0] - y[0]).map(([k, l]) => `${k * 4}-${k * 4 + 4}mm:${((l.mv / l.n) * 100).toFixed(0)}%动(vy ${(l.vy / Math.max(1, l.mv)).toFixed(2)})`).join('  ');
      sp.sort((x, y) => x - y);
      const q = (f: number) => +sp[Math.floor(f * (sp.length - 1))].toFixed(3);
      console.log(`[restpile] 逐帧 mm/s p50 ${q(0.5)} p90 ${q(0.9)} p99 ${q(0.99)} max ${q(1)}  休眠 ${asleep}/${sp.length}\n   深度分层 ${desc}`);
    }
    (window as any).__result = { ok: true };
    return;
  }
  if (mode === 'neck') {
    // 瓶颈现场：沙在上泡，沉降 1 s 后开孔口流 3 s，列出离孔口最近的颗粒团
    const sp = defaultParams(rc, sandMassForSeconds(3600));
    if (params.get('solver')) (sp as any).solver = params.get('solver');
    if (params.get('prestab')) sp.prestab = params.get('prestab') === '1';
    if (params.get('shock')) sp.shockK = Number(params.get('shock'));
    const sim = new SandSim(device, sp);
    // ?flipg=1：整个场景上下颠倒（沙在 -y 泡、重力朝 +y），检查方向不对称的问题
    const gs = params.get('flipg') === '1' ? -1 : 1;
    sim.seed(seedBulb(rc, sandMassForSeconds(minutes * 60), gs as 1 | -1, gs as 1 | -1, Number(params.get('seedn') ?? 5)));
    let drained = 0;
    sim.onStats = (s) => (drained += s.drained);
    const settleF = Number(params.get('settle') ?? 60);
    const flowF = Number(params.get('flowf') ?? 180);
    for (let f = 0; f < settleF + flowF; f++) {
      const enc = device.createCommandEncoder();
      sim.encodeFrame(enc, { wallDt: 1 / 60, substeps: 8, gLocal: [0, -G0 * gs, 0], omega: [0, 0, 0], alpha: [0, 0, 0], flow: f >= settleF });
      device.queue.submit([enc.finish()]);
      sim.afterSubmit();
      await sim.flush();
      if (f % 30 === 29) console.log(`[neck] f=${f} drained=${drained.toFixed(2)} mg`);
      const sprayFrom = Number(params.get('sprayfrom') ?? 0);
      if (params.get('spray') && (f % 10 === 9 || f < 10 || (f >= sprayFrom && f < sprayFrom + 24))) {
        // 向上飞的颗粒团：速度 y 分量 > 5 cm/s
        const st = await sim.readState();
        const vb = device.createBuffer({ size: sim.buf.vel.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const e = device.createCommandEncoder();
        e.copyBufferToBuffer(sim.buf.vel, 0, vb, 0, sim.buf.vel.size);
        device.queue.submit([e.finish()]);
        await vb.mapAsync(GPUMapMode.READ);
        const vv = new Float32Array(vb.getMappedRange().slice(0));
        vb.unmap();
        vb.destroy();
        let up = 0;
        let maxUp = 0;
        let where = '';
        let awake = 0;
        let down = 0;
        for (let i = 0; i < sim.p.clumpCap; i++) {
          if (st.pos[i * 4 + 3] <= 0) continue;
          if (st.rest[i * 4 + 3] >= 0) awake++;
          if (vv[i * 4 + 1] < -0.05) down++;
        }
        for (let i = 0; i < sim.p.clumpCap; i++) {
          if (st.pos[i * 4 + 3] <= 0) continue;
          const vy = vv[i * 4 + 1] * gs;
          if (vy > 0.05) up++;
          if (vy > maxUp) {
            maxUp = vy;
            where = `y=${(st.pos[i * 4 + 1] * 1000).toFixed(1)} r=${(Math.hypot(st.pos[i * 4], st.pos[i * 4 + 2]) * 1000).toFixed(1)} m=${(st.pos[i * 4 + 3] / sim.mc).toFixed(2)} rest=${st.rest[i * 4 + 3].toFixed(0)}`;
          }
        }
        console.log(`[spray] f=${f + 1} 醒着 ${awake} 下落(>5cm/s) ${down} 上飞 ${up} 个，最快 ${(maxUp * 1000).toFixed(0)} mm/s @ ${where}`);
      }
    }
    const st = await sim.readState();
    const vel = await (async () => {
      const b = device.createBuffer({ size: sim.buf.vel.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(sim.buf.vel, 0, b, 0, sim.buf.vel.size);
      device.queue.submit([e.finish()]);
      await b.mapAsync(GPUMapMode.READ);
      const o = new Float32Array(b.getMappedRange().slice(0));
      b.unmap();
      return o;
    })();
    const rows: any[] = [];
    for (let i = 0; i < sim.p.clumpCap; i++) {
      const m = st.pos[i * 4 + 3];
      if (m <= 0) continue;
      const x = st.pos[i * 4], y = st.pos[i * 4 + 1], z = st.pos[i * 4 + 2];
      rows.push({ d: Math.hypot(x, y, z) * 1000, y: +(y * 1000).toFixed(2), r: +(Math.hypot(x, z) * 1000).toFixed(2), m: +(m / sim.mc).toFixed(2), rest: +st.rest[i * 4 + 3].toFixed(0), v: +(Math.hypot(vel[i * 4], vel[i * 4 + 1], vel[i * 4 + 2]) * 1000).toFixed(2) });
    }
    rows.sort((a, b) => a.d - b.d);
    const near = rows.slice(0, 20).map((r) => ({ ...r, d: +r.d.toFixed(2) }));
    console.log('[neck] nearest ' + JSON.stringify(near));
    (window as any).__result = { drained, near: near.length };
    return;
  }
  if (mode === 'stream') {
    // 沙流统计：在飞沙粒的数量、高度分布、横向散布
    const sim = new SandSim(device, defaultParams(rc, sandMassForSeconds(3600)));
    sim.seed(seedBulb(rc, sandMassForSeconds(minutes * 60), 1, 1, 5));
    let emitted = 0;
    sim.onStats = (s) => (emitted += s.emitted);
    for (let f = 0; f < 150; f++) {
      const enc = device.createCommandEncoder();
      sim.encodeFrame(enc, { wallDt: 1 / 60, substeps: 8, gLocal: [0, -G0, 0], omega: [0, 0, 0], alpha: [0, 0, 0], flow: f >= 60 });
      device.queue.submit([enc.finish()]);
      sim.afterSubmit();
      await sim.flush();
    }
    const st = await sim.readState();
    const ys: number[] = [];
    const rs: number[] = [];
    let alive = 0;
    for (let i = 0; i < sim.p.grainCap; i++) {
      if (st.grains[i * 4 + 3] <= 0) continue;
      alive++;
      ys.push(st.grains[i * 4 + 1]);
      rs.push(Math.hypot(st.grains[i * 4], st.grains[i * 4 + 2]));
    }
    const hist = new Array(10).fill(0);
    for (const y of ys) hist[Math.min(9, Math.max(0, Math.floor((-y / 0.09) * 10)))]++;
    rs.sort((a, b) => a - b);
    const pr = (f: number) => ((rs[Math.floor(f * (rs.length - 1))] ?? 0) * 1000).toFixed(2);
    const res = { emitted, alive, expectedInFlight: Math.round(42700 * 0.14), yHist_0to9cm: hist, radius_mm: { p50: pr(0.5), p90: pr(0.9), max: pr(1) } };
    console.log('[stream] ' + JSON.stringify(res));
    (window as any).__result = res;
    return;
  }
  if (mode === 'drain') await runDrainTest(device, rc, minutes);
  else if (mode === 'debug') await runDebug(device, rc, minutes);
  else if (mode === 'view') await runView(device, rc, minutes);
  else {
    const settings = loadSettings();
    if (params.get('min')) settings.minutes = minutes;
    // 画质档位：颗粒团半径、最大像素密度、光子数
    const tiers = {
      high: { rc: 0.75, maxDpr: 2, photons: 768 },
      medium: { rc: 1.0, maxDpr: 1.5, photons: 512 },
      low: { rc: 1.25, maxDpr: 1.25, photons: 384 },
    } as const;
    const q = settings.quality === 'auto' ? (isMobile ? 'medium' : 'high') : settings.quality;
    const tier = tiers[q];
    const appRc = params.get('rc') ? rc : tier.rc / 1000;
    // 诊断用的物理参数覆盖：?damp=0.9&sleep=0.005&iters=4&sub=…
    const ov: Record<string, number> = {};
    for (const [k, key] of [
      ['damp', 'damp'],
      ['sleep', 'sleepV'],
      ['iters', 'iterations'],
      ['sor', 'sor'],
      ['shock', 'shockK'],
    ] as const)
      if (params.get(k)) ov[key] = Number(params.get(k));
    if (params.get('avg')) (ov as any).average = params.get('avg') === '1';
    if (params.get('solver')) (ov as any).solver = params.get('solver');
    const app = new App(device, document.getElementById('view') as HTMLCanvasElement, appRc, settings, tier, ov);
    (window as any).__app = app;
    if (params.get('scale')) {
      app.renderer.renderScale = Number(params.get('scale'));
      app.autoScale = false;
    }
    if (params.get('hud') === '1') attachHud(app, device);
    if (params.get('probe')) {
      if (params.get('render') !== '1') app.renderer.upto = 0;
      await runProbe(app, params.get('probe')!, params);
    } else if (params.get('test') === '1') await runAppScenario(app);
    else if (params.get('test') === 'finish') await runFinishScenario(app);
    else if (params.get('test') === 'restore') {
      // 伪造一个 30 s 前存档的会话：2 分钟的沙，已流 30 s，离开 10 s → 恢复后应剩约 80 s
      const total = sandMassForSeconds(120);
      localStorage.setItem('hourglass.session.v1', JSON.stringify({ total, drained: W0 * 30, savedAt: Date.now() - 10000, paused: false, angle: Math.PI }));
      await app.restore();
      for (let i = 0; i < 6; i++) await app.step(1 / 30);
      (window as any).__shotRequest = 'restored';
      (window as any).__shotDone = false;
      while (!(window as any).__shotDone) await new Promise((r) => setTimeout(r, 100));
      (window as any).__result = app.debugState;
    }
    else {
      await app.restore();
      app.start();
    }
  }
}

main().catch((e) => {
  console.error(e);
  fail(String(e?.message ?? e));
});
