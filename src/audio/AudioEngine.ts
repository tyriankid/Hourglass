// 声音：物理驱动的沙声、环境音、生成式背景音乐、结束时的颂钵、刻度尺的轻响。全部程序合成。
// 浏览器要求在用户手势里启动 AudioContext，所以第一次触摸/点击时才真正初始化。

import sandUrl from './sand.worklet.js?url';
import ambientUrl from './ambient.worklet.js?url';
import type { FrameStats } from '../sim/SandSim';
import type { Settings } from '../ui/Overlay';

const AMBIENT_MODE = { none: 0, rain: 1, fire: 2, noise: 3 } as const;

// 背景音乐的和弦（D 大调五声音阶的柔和配置），每个和弦 4 个音（MIDI）
const CHORDS = [
  [50, 57, 62, 66, 69],
  [47, 54, 59, 62, 66],
  [43, 50, 57, 59, 66],
  [45, 52, 57, 61, 64],
  [40, 47, 54, 57, 62],
  [45, 49, 57, 64, 66],
];

const midi = (m: number) => 440 * Math.pow(2, (m - 69) / 12);

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private sand: AudioWorkletNode | null = null;
  private ambient: AudioWorkletNode | null = null;
  private master: GainNode | null = null;
  private sandGain: GainNode | null = null;
  private musicGain: GainNode | null = null;
  private reverb: ConvolverNode | null = null;
  private reverbSend: GainNode | null = null;
  private starting: Promise<void> | null = null;
  private nextChord = 0;
  private chordIdx = 0;
  private smooth = { bed: 0, clicks: 0, amp: 0, glass: 0, slide: 0 };
  private muted = false;

  constructor(private settings: Settings) {
    const unlock = () => {
      this.start();
      if (this.ctx?.state === 'suspended') this.ctx.resume();
    };
    window.addEventListener('pointerdown', unlock, { passive: true });
    window.addEventListener('keydown', unlock);
    document.addEventListener('visibilitychange', () => {
      // 看不见时物理不跑，沙声静音；环境音和音乐继续（锁屏时由系统决定）
      this.muted = document.hidden;
    });
  }

  private start(): Promise<void> {
    if (this.starting) return this.starting;
    this.starting = (async () => {
      const ctx = new AudioContext({ latencyHint: 'playback' });
      this.ctx = ctx;
      await Promise.all([ctx.audioWorklet.addModule(sandUrl), ctx.audioWorklet.addModule(ambientUrl)]);
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -18;
      comp.ratio.value = 3;
      comp.attack.value = 0.01;
      comp.release.value = 0.3;
      comp.connect(ctx.destination);
      this.master = ctx.createGain();
      this.master.gain.value = 0.9;
      this.master.connect(comp);
      this.reverb = ctx.createConvolver();
      this.reverb.buffer = this.impulse(ctx, 4.5);
      this.reverbSend = ctx.createGain();
      this.reverbSend.gain.value = 0.6;
      this.reverbSend.connect(this.reverb);
      this.reverb.connect(this.master);

      this.sand = new AudioWorkletNode(ctx, 'sand', { numberOfOutputs: 1, outputChannelCount: [2] });
      this.sandGain = ctx.createGain();
      this.sand.connect(this.sandGain);
      this.sandGain.connect(this.master);
      const sandVerb = ctx.createGain();
      sandVerb.gain.value = 0.08;
      this.sandGain.connect(sandVerb);
      sandVerb.connect(this.reverbSend);

      this.ambient = new AudioWorkletNode(ctx, 'ambient', { numberOfOutputs: 1, outputChannelCount: [2] });
      this.ambient.connect(this.master);

      this.musicGain = ctx.createGain();
      this.musicGain.connect(this.master);
      this.musicGain.connect(this.reverbSend);
      this.applySettings(this.settings);
    })();
    return this.starting;
  }

  /** 混响的冲激响应：指数衰减的双声道噪声 */
  private impulse(ctx: AudioContext, seconds: number): AudioBuffer {
    const len = Math.floor(ctx.sampleRate * seconds);
    const buf = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let c = 0; c < 2; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3.2) * 0.5;
    }
    return buf;
  }

  applySettings(s: Settings) {
    this.settings = s;
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this.sandGain!.gain.setTargetAtTime(s.sandVolume, t, 0.1);
    this.ambient!.parameters.get('mode')!.setValueAtTime(AMBIENT_MODE[s.ambient], t);
    this.ambient!.parameters.get('gain')!.setTargetAtTime(s.ambientVolume * 0.9, t, 0.2);
    this.musicGain!.gain.setTargetAtTime(s.musicVolume * 0.35, t, 0.3);
  }

  /** 每帧：把物理统计换成沙声参数（平滑过渡） */
  update(stats: FrameStats | null, dt: number) {
    if (!this.ctx || !this.sand || !stats || dt <= 0) return;
    const fdt = Math.max(dt, 1 / 240);
    const impacts = stats.impactsSand / fdt;
    const speed = stats.impactsSand > 0 ? stats.impactSpeedSum / stats.impactsSand : 0;
    const energy = impacts * speed * speed;
    const target = {
      bed: Math.min(0.25, 2.2e-4 * Math.sqrt(energy)),
      clicks: Math.min(2500, impacts * 0.035),
      amp: Math.min(0.6, 0.35 * Math.min(1, speed / 1.2) + 0.08),
      glass: Math.min(600, (stats.impactsGlass / fdt) * 0.15),
      slide: Math.min(0.5, Math.max(0, (stats.maxSpeed - 0.03) / 0.4) * 0.35),
    };
    if (this.muted) {
      target.bed = target.clicks = target.glass = target.slide = 0;
    }
    const k = 1 - Math.exp(-dt / 0.08);
    const s = this.smooth;
    s.bed += (target.bed - s.bed) * k;
    s.clicks += (target.clicks - s.clicks) * k;
    s.amp += (target.amp - s.amp) * k;
    s.glass += (target.glass - s.glass) * k;
    s.slide += (target.slide - s.slide) * k;
    const t = this.ctx.currentTime;
    const p = this.sand.parameters;
    p.get('bed')!.setValueAtTime(s.bed, t);
    p.get('clicks')!.setValueAtTime(s.clicks, t);
    p.get('clickAmp')!.setValueAtTime(s.amp, t);
    p.get('glass')!.setValueAtTime(s.glass, t);
    p.get('slide')!.setValueAtTime(s.slide, t);
    this.music();
  }

  /** 生成式氛围音乐：每隔 8–11 秒换一个和弦，慢起慢落，过混响 */
  private music() {
    const ctx = this.ctx!;
    if (this.settings.musicVolume <= 0.001) return;
    if (ctx.currentTime < this.nextChord) return;
    const t0 = Math.max(ctx.currentTime, this.nextChord);
    const dur = 8 + Math.random() * 3;
    this.nextChord = t0 + dur;
    this.chordIdx = (this.chordIdx + 1 + Math.floor(Math.random() * 2)) % CHORDS.length;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 1400;
    lp.connect(this.musicGain!);
    for (const m of CHORDS[this.chordIdx]) {
      for (const [type, det, amp] of [
        ['sine', -3, 0.05],
        ['triangle', 4, 0.025],
      ] as const) {
        const o = ctx.createOscillator();
        o.type = type;
        o.frequency.value = midi(m);
        o.detune.value = det;
        const g = ctx.createGain();
        g.gain.setValueAtTime(0, t0);
        g.gain.linearRampToValueAtTime(amp, t0 + 3.2);
        g.gain.setTargetAtTime(0, t0 + dur - 1.5, 1.8);
        o.connect(g);
        g.connect(lp);
        o.start(t0);
        o.stop(t0 + dur + 8);
      }
    }
  }

  /** 结束提示：颂钵（不谐和泛音 + 轻微拍频 + 长衰减） */
  chime() {
    const ctx = this.ctx;
    if (!ctx || !this.settings.chime) return;
    const t = ctx.currentTime + 0.05;
    const f0 = 392;
    const partials: Array<[number, number, number]> = [
      [1, 1, 9],
      [2.76, 0.55, 6],
      [5.4, 0.3, 4],
      [8.93, 0.14, 2.5],
    ];
    const out = ctx.createGain();
    out.gain.value = 0.22;
    out.connect(this.master!);
    out.connect(this.reverbSend!);
    for (const [ratio, amp, decay] of partials) {
      for (const beat of [-0.6, 0.6]) {
        const o = ctx.createOscillator();
        o.frequency.value = f0 * ratio + beat * ratio;
        const g = ctx.createGain();
        g.gain.setValueAtTime(0, t);
        g.gain.linearRampToValueAtTime(amp * 0.5, t + 0.012);
        g.gain.setTargetAtTime(0, t + 0.012, decay / 3);
        o.connect(g);
        g.connect(out);
        o.start(t);
        o.stop(t + decay * 2.5);
      }
    }
    // 敲击瞬间的一点噪声
    const n = ctx.createBufferSource();
    const b = ctx.createBuffer(1, Math.floor(ctx.sampleRate * 0.05), ctx.sampleRate);
    const d = b.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (ctx.sampleRate * 0.008));
    n.buffer = b;
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 2600;
    bp.Q.value = 1.2;
    const ng = ctx.createGain();
    ng.gain.value = 0.25;
    n.connect(bp);
    bp.connect(ng);
    ng.connect(out);
    n.start(t);
  }

  /** 刻度尺每跳一格的轻响 */
  tick() {
    const ctx = this.ctx;
    if (!ctx || ctx.state !== 'running') return;
    const t = ctx.currentTime;
    const n = ctx.createBufferSource();
    const b = ctx.createBuffer(1, Math.floor(ctx.sampleRate * 0.012), ctx.sampleRate);
    const d = b.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (ctx.sampleRate * 0.0015));
    n.buffer = b;
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 3800;
    bp.Q.value = 2;
    const g = ctx.createGain();
    g.gain.value = 0.06;
    n.connect(bp);
    bp.connect(g);
    g.connect(this.master!);
    n.start(t);
  }
}
