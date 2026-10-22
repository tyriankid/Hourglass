// 探测脚本：用和真机相同的代码路径（App.step）驱动，记录抖动、能否翻转、凝聚时的速度与重叠等指标。
// 用法：?probe=<名字>&min=..&rc=..&dtj=0.3（帧间隔随机抖动比例）

import type { App } from '../app/App';
import { sandVoice } from '../audio/AudioEngine';
import { W0 } from '../physics';

const w = window as any;

function readyParts(app: App) {
  const a = app as any;
  const target = W0 * app.ui.minutes * 60;
  const m = a.downBulb() > 0 ? a.massPos : a.massNeg;
  return {
    massOk: Math.abs(m - target) <= 1,
    diffMg: +(m - target).toFixed(2),
    cooldown: a.cooldown,
    histBusy: a.histBusy,
    maxSpeed: +a.maxSpeed.toFixed(4),
    statsSeen: a.statsSeen,
    state: app.state,
    flipQueued: a.flipQueued,
  };
}

/** 读回颗粒团位置，算两次读回之间每个颗粒的平均速度分布 */
async function snapshot(app: App) {
  const st = await app.sim.readState();
  return st.pos;
}

function motionStats(a: Float32Array, b: Float32Array, dtSec: number) {
  const sp: number[] = [];
  let overlap = 0;
  for (let i = 0; i < a.length / 4; i++) {
    if (a[i * 4 + 3] <= 0 || b[i * 4 + 3] <= 0) continue;
    const dx = b[i * 4] - a[i * 4];
    const dy = b[i * 4 + 1] - a[i * 4 + 1];
    const dz = b[i * 4 + 2] - a[i * 4 + 2];
    const d = Math.hypot(dx, dy, dz);
    if (d > 0.005) continue; // 重新分配的槽位
    sp.push((d / dtSec) * 1000);
  }
  sp.sort((x, y) => x - y);
  const q = (f: number) => +(sp[Math.floor(f * (sp.length - 1))] ?? 0).toFixed(3);
  const moving = sp.filter((v) => v > 1).length;
  return { n: sp.length, p50_mm_s: q(0.5), p90: q(0.9), p99: q(0.99), max: q(1), movingOver1mm_s: moving, overlap };
}

/** 相邻两帧的瞬时速度分布，并给最快的颗粒画像：质量比、离轴距离、高度、离玻璃壁的距离 */
function frameStats(a: Float32Array, b: Float32Array, dtSec: number, mc: number, sdf: (x: number, y: number, z: number) => number) {
  const rows: Array<{ v: number; m: number; r: number; y: number; wall: number; i: number }> = [];
  for (let i = 0; i < a.length / 4; i++) {
    if (a[i * 4 + 3] <= 0 || b[i * 4 + 3] <= 0) continue;
    const d = Math.hypot(b[i * 4] - a[i * 4], b[i * 4 + 1] - a[i * 4 + 1], b[i * 4 + 2] - a[i * 4 + 2]);
    if (d > 0.005) continue;
    rows.push({ v: (d / dtSec) * 1000, m: b[i * 4 + 3] / mc, r: Math.hypot(b[i * 4], b[i * 4 + 2]) * 1000, y: b[i * 4 + 1] * 1000, wall: sdf(b[i * 4], b[i * 4 + 1], b[i * 4 + 2]) * 1000, i });
  }
  rows.sort((x, y) => x.v - y.v);
  const q = (f: number) => +(rows[Math.floor(f * (rows.length - 1))]?.v ?? 0).toFixed(2);
  const fast = rows.filter((r) => r.v > 5);
  const top = rows.slice(-8).map((r) => ({ v: +r.v.toFixed(1), m: +r.m.toFixed(2), r: +r.r.toFixed(1), y: +r.y.toFixed(1), wall: +r.wall.toFixed(2) }));
  const partialFast = fast.filter((r) => r.m < 0.999).length;
  const nearWallFast = fast.filter((r) => r.wall < 1.5).length;
  return { n: rows.length, p50: q(0.5), p90: q(0.9), p99: q(0.99), max: q(1), over5mm_s: fast.length, partialAmongFast: partialFast, nearWallAmongFast: nearWallFast, top };
}

export async function runProbe(app: App, name: string, params: URLSearchParams) {
  const dtj = Number(params.get('dtj') ?? 0);
  const base = 1 / 60;
  let rng = 12345;
  const rand = () => {
    rng ^= rng << 13;
    rng ^= rng >>> 17;
    rng ^= rng << 5;
    return (rng >>> 0) / 4294967296;
  };
  const dtNext = () => base * (1 + dtj * (rand() * 2 - 1));
  let t = 0;
  // 每帧记录沙声参数，按状态汇总：看每个阶段是哪个声部在响
  const voice: Record<string, { n: number; bed: number; clicks: number; slide: number; maxBed: number; maxSlide: number; maxClicks: number }> = {};
  const step = async () => {
    const d = dtNext();
    await app.step(d);
    t += d;
    const st = (app as any).lastStats;
    if (st) {
      const v = sandVoice(st, d);
      const key = app.state;
      const a = (voice[key] ??= { n: 0, bed: 0, clicks: 0, slide: 0, maxBed: 0, maxSlide: 0, maxClicks: 0 });
      a.n++;
      a.bed += v.bed;
      a.clicks += v.clicks;
      a.slide += v.slide;
      a.maxBed = Math.max(a.maxBed, v.bed);
      a.maxSlide = Math.max(a.maxSlide, v.slide);
      a.maxClicks = Math.max(a.maxClicks, v.clicks);
    }
    return d;
  };
  const log = (tag: string, o: unknown) => console.log(`[probe] ${tag} t=${t.toFixed(2)} ${JSON.stringify(o)}`);
  const waitReady = async (maxSec: number) => {
    const t0 = t;
    while (t - t0 < maxSec && !(app as any).ready()) {
      await step();
      if (Math.round(t * 60) % 60 === 0) log('wait', readyParts(app));
    }
    return (app as any).ready();
  };
  const jitterWindow = async (label: string, seconds: number) => {
    const a = await snapshot(app);
    const t0 = t;
    let maxS = 0;
    while (t - t0 < seconds) {
      await step();
      maxS = Math.max(maxS, (app as any).maxSpeed);
    }
    const b = await snapshot(app);
    log(label, { ...motionStats(a, b, t - t0), statsMaxSpeed_mm_s: +(maxS * 1000).toFixed(1) });
  };

  const perFrame = async (label: string, frames: number) => {
    const { interiorSdfCpu } = await import('../sim/SandSim');
    for (let k = 0; k < frames; k++) {
      const a = await snapshot(app);
      const d = await step();
      const st = await app.sim.readState();
      const b = st.pos;
      let asleep = 0;
      let alive = 0;
      const cnt = new Map<number, number>();
      for (let i = 0; i < b.length / 4; i++) {
        if (b[i * 4 + 3] <= 0) continue;
        alive++;
        const w = st.rest[i * 4 + 3];
        if (w < 0) asleep++;
        else cnt.set(Math.min(10, Math.floor(w)), (cnt.get(Math.min(10, Math.floor(w))) ?? 0) + 1);
      }
      log(label, { asleep, alive, awakeCounterHist: Object.fromEntries([...cnt.entries()].sort((x, y) => x[0] - y[0])), ...frameStats(a, b, d, app.sim.mc, interiorSdfCpu) });
    }
  };

  // 连续渲染并逐帧截图（与 tools/run.mjs 握手），之后在 CPU 上算相邻帧差异、拼视频
  const filmFrames = async (tag: string, n: number) => {
    app.renderer.upto = 99;
    for (let k = 0; k < n; k++) {
      await step();
      w.__shotDone = false;
      w.__shotRequest = `${tag}-${String(k).padStart(3, '0')}`;
      while (!w.__shotDone) await new Promise((r) => setTimeout(r, 50));
    }
    app.renderer.upto = 0;
  };

  const result: Record<string, unknown> = {};
  if (name === 'film-rest') {
    // 静置的沙堆：渲染连续帧，看画面是否完全不动
    await waitReady(25);
    for (let i = 0; i < Number(params.get('rest') ?? 240); i++) await step();
    await filmFrames('rest', Number(params.get('n') ?? 24));
  } else if (name === 'frame') {
    await waitReady(25);
    const restFrames = Number(params.get('rest') ?? 180);
    for (let i = 0; i < restFrames; i++) await step();
    await perFrame('frame', 3);
    // 重叠统计（CPU 两两比较）
    const st = await app.sim.readState();
    const pts: number[][] = [];
    for (let i = 0; i < st.pos.length / 4; i++) if (st.pos[i * 4 + 3] > 0) pts.push([st.pos[i * 4], st.pos[i * 4 + 1], st.pos[i * 4 + 2], app.sim.p.rc * Math.cbrt(st.pos[i * 4 + 3] / app.sim.mc)]);
    let pairs = 0;
    let o5 = 0;
    let o10 = 0;
    let maxO = 0;
    let sumO = 0;
    let ymax = -1;
    for (const p of pts) ymax = Math.max(ymax, p[1]);
    for (let a = 0; a < pts.length; a++)
      for (let b = a + 1; b < pts.length; b++) {
        const dx = pts[a][0] - pts[b][0];
        if (dx > 0.003 || dx < -0.003) continue;
        const dy = pts[a][1] - pts[b][1];
        const dz = pts[a][2] - pts[b][2];
        const rr = pts[a][3] + pts[b][3];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= rr * rr) continue;
        const o = 1 - Math.sqrt(d2) / rr;
        pairs++;
        sumO += o;
        if (o > 0.05) o5++;
        if (o > 0.1) o10++;
        maxO = Math.max(maxO, o);
      }
    log('overlap', { pairs, meanPct: +((sumO / pairs) * 100).toFixed(2), over5pct: o5, over10pct: o10, maxPct: +(maxO * 100).toFixed(1), pileTopY_mm: +(ymax * 1000).toFixed(1) });
  } else if (name === 'jitter') {
    // 调时间的静止状态：沙应当完全静止
    const ok = await waitReady(25);
    result.readyAfter = +t.toFixed(2);
    result.ready = ok;
    log('ready', readyParts(app));
    for (let k = 0; k < 3; k++) await jitterWindow('rest', 1);
  } else if (name === 'dial') {
    // 先 1 分钟就绪，再把刻度一下拉到 to 分钟，观察凝聚过程
    app.ui.setMinutes(1);
    await waitReady(20);
    log('ready1', readyParts(app));
    app.ui.setMinutes(Number(params.get('to') ?? 60));
    for (let k = 0; k < 10; k++) await jitterWindow('condense', 1);
    const ok = await waitReady(30);
    log('afterCondense', { ok, ...readyParts(app) });
    for (let k = 0; k < 2; k++) await jitterWindow('rest', 1);
  } else if (name === 'rerun') {
    // 跑完一轮后能否再跑：1 分钟
    app.ui.setMinutes(1);
    await waitReady(20);
    app.tap();
    const t0 = t;
    while (app.state !== 'finished' && t - t0 < 90) {
      await step();
      if (Math.round(t * 60) % 600 === 0) log('run', { ...readyParts(app), remaining: app.clock?.remainingSeconds });
    }
    log('finished', { ...readyParts(app), remaining: app.clock?.remainingSeconds });
    while (app.state !== 'idle' && t - t0 < 100) await step();
    log('idle', readyParts(app));
    const ok = await waitReady(20);
    log('ready-again', { ok, ...readyParts(app) });
    app.tap();
    for (let i = 0; i < 30; i++) await step();
    result.secondRunState = app.state;
  } else if (name === 'reset') {
    app.ui.setMinutes(2);
    await waitReady(20);
    app.tap();
    for (let i = 0; i < 180; i++) await step();
    log('running', readyParts(app));
    (app as any).reset();
    for (let i = 0; i < 30; i++) await step();
    log('after-reset', readyParts(app));
    const ok = await waitReady(30);
    log('ready-after-reset', { ok, ...readyParts(app) });
  }
  result.final = readyParts(app);
  result.voice = Object.fromEntries(
    Object.entries(voice).map(([k, a]) => [
      k,
      { frames: a.n, bed: +(a.bed / a.n).toFixed(4), clicks: +(a.clicks / a.n).toFixed(1), slide: +(a.slide / a.n).toFixed(4), maxBed: +a.maxBed.toFixed(4), maxSlide: +a.maxSlide.toFixed(4), maxClicks: +a.maxClicks.toFixed(0) },
    ]),
  );
  w.__result = result;
}
