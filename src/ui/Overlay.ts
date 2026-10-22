// 极简覆盖层：时间、提示、刻度尺（1–60 分钟）、右上角按钮、设置面板。

import './overlay.css';

export type Ambient = 'none' | 'rain' | 'fire' | 'noise';
export type Quality = 'auto' | 'high' | 'medium' | 'low';

export interface Settings {
  minutes: number;
  sandVolume: number;
  ambient: Ambient;
  ambientVolume: number;
  musicVolume: number;
  chime: boolean;
  gyro: boolean;
  quality: Quality;
}

const DEFAULTS: Settings = {
  minutes: 25,
  sandVolume: 0.8,
  ambient: 'none',
  ambientVolume: 0.5,
  musicVolume: 0,
  chime: true,
  gyro: false,
  quality: 'auto',
};

const KEY = 'hourglass.settings.v1';

export function loadSettings(): Settings {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) ?? '{}') };
  } catch {
    return { ...DEFAULTS };
  }
}

export function saveSettings(s: Settings) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* 隐私模式下忽略 */
  }
}

const ICON_GEAR =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>';
const ICON_RESET =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>';

export class Overlay {
  readonly root: HTMLDivElement;
  private timeEl: HTMLDivElement;
  private hintEl: HTMLDivElement;
  private ruler: HTMLCanvasElement;
  private topEl: HTMLDivElement;
  private resetBtn: HTMLButtonElement;
  private panel: HTMLDivElement;
  private veil: HTMLDivElement;
  private idleTimer = 0;

  private value: number;
  private vel = 0;
  private dragging = false;
  private lastX = 0;
  private lastT = 0;
  private lastInt: number;

  onMinutes: (m: number) => void = () => {};
  onReset: () => void = () => {};
  onSettings: (s: Settings) => void = () => {};

  constructor(readonly settings: Settings) {
    this.value = settings.minutes;
    this.lastInt = settings.minutes;
    const root = (this.root = document.createElement('div'));
    root.className = 'hg-ui';
    root.innerHTML = `
      <div class="hg-top"><button class="hg-btn hidden" data-a="reset" title="重置">${ICON_RESET}</button><button class="hg-btn" data-a="gear" title="设置">${ICON_GEAR}</button></div>
      <div class="hg-bottom"><div class="hg-time"></div><div class="hg-hint"></div><canvas class="hg-ruler"></canvas></div>
      <div class="hg-panel hidden"></div>`;
    document.body.appendChild(root);
    this.veil = document.createElement('div');
    this.veil.className = 'hg-veil';
    document.body.appendChild(this.veil);
    this.timeEl = root.querySelector('.hg-time')!;
    this.hintEl = root.querySelector('.hg-hint')!;
    this.ruler = root.querySelector('.hg-ruler')!;
    this.topEl = root.querySelector('.hg-top')!;
    this.resetBtn = root.querySelector('[data-a=reset]')!;
    this.panel = root.querySelector('.hg-panel')!;
    this.resetBtn.addEventListener('click', () => this.onReset());
    root.querySelector('[data-a=gear]')!.addEventListener('click', () => this.panel.classList.toggle('hidden'));
    this.buildPanel();
    this.bindRuler();
    for (const ev of ['pointermove', 'pointerdown', 'keydown', 'wheel']) window.addEventListener(ev, () => this.poke(), { passive: true });
    this.poke();
  }

  /** 有操作时显示提示和按钮，几秒无操作后淡出 */
  poke() {
    this.topEl.classList.remove('hg-fade');
    this.hintEl.classList.remove('hg-fade');
    clearTimeout(this.idleTimer);
    this.idleTimer = window.setTimeout(() => {
      if (!this.panel.classList.contains('hidden')) return;
      this.topEl.classList.add('hg-fade');
      this.hintEl.classList.add('hg-fade');
    }, 4000);
  }

  setTime(text: string) {
    if (this.timeEl.textContent !== text) this.timeEl.textContent = text;
  }

  setHint(text: string) {
    if (this.hintEl.textContent !== text) this.hintEl.textContent = text;
  }

  showRuler(on: boolean) {
    this.ruler.classList.toggle('hidden', !on);
  }

  showReset(on: boolean) {
    this.resetBtn.classList.toggle('hidden', !on);
  }

  /** 离开画面回来补算时盖一层淡淡的幕布 */
  veilOn(on: boolean) {
    this.veil.classList.toggle('on', on);
  }

  get minutes(): number {
    return this.lastInt;
  }

  private emitIfChanged() {
    const m = Math.min(60, Math.max(1, Math.round(this.value)));
    if (m !== this.lastInt) {
      this.lastInt = m;
      this.settings.minutes = m;
      saveSettings(this.settings);
      if ('vibrate' in navigator) navigator.vibrate?.(4);
      this.onMinutes(m);
    }
  }

  private bindRuler() {
    const c = this.ruler;
    c.addEventListener('pointerdown', (e) => {
      this.dragging = true;
      this.vel = 0;
      this.lastX = e.clientX;
      this.lastT = performance.now();
      c.setPointerCapture(e.pointerId);
      e.stopPropagation();
    });
    c.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      const now = performance.now();
      const dx = e.clientX - this.lastX;
      const dv = -dx / this.pxPerMin();
      this.value = Math.min(60.4, Math.max(0.6, this.value + dv));
      const dt = Math.max(1, now - this.lastT);
      this.vel = this.vel * 0.6 + (dv / dt) * 1000 * 0.4;
      this.lastX = e.clientX;
      this.lastT = now;
      this.emitIfChanged();
    });
    const end = () => {
      this.dragging = false;
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.value = Math.min(60, Math.max(1, Math.round(this.value) + Math.sign(e.deltaY || e.deltaX)));
      this.emitIfChanged();
    });
  }

  private pxPerMin() {
    return 16;
  }

  /** 每帧调用：惯性、吸附、重绘刻度尺 */
  update(dt: number) {
    if (!this.dragging) {
      if (Math.abs(this.vel) > 0.3) {
        this.value = Math.min(60, Math.max(1, this.value + this.vel * dt));
        this.vel *= Math.exp(-dt * 4);
      } else {
        this.vel = 0;
        const target = Math.min(60, Math.max(1, Math.round(this.value)));
        this.value += (target - this.value) * (1 - Math.exp(-dt * 12));
      }
      this.emitIfChanged();
    }
    this.drawRuler();
  }

  private drawRuler() {
    const c = this.ruler;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const w = c.clientWidth;
    const h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const ppm = this.pxPerMin();
    const cx = w / 2;
    const lo = Math.max(1, Math.floor(this.value - cx / ppm) - 1);
    const hi = Math.min(60, Math.ceil(this.value + cx / ppm) + 1);
    g.font = `300 11px ${getComputedStyle(this.root).fontFamily}`;
    g.textAlign = 'center';
    for (let m = lo; m <= hi; m++) {
      const x = cx + (m - this.value) * ppm;
      const major = m % 5 === 0 || m === 1;
      g.strokeStyle = major ? 'rgba(52,50,47,0.5)' : 'rgba(52,50,47,0.22)';
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x, 8);
      g.lineTo(x, major ? 26 : 18);
      g.stroke();
      if (major) {
        g.fillStyle = 'rgba(52,50,47,0.45)';
        g.fillText(String(m), x, 42);
      }
    }
    g.strokeStyle = '#a8957a';
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(cx, 4);
    g.lineTo(cx, 30);
    g.stroke();
  }

  private buildPanel() {
    const s = this.settings;
    const p = this.panel;
    const row = (label: string, control: HTMLElement) => {
      const r = document.createElement('div');
      r.className = 'hg-row';
      const l = document.createElement('label');
      l.textContent = label;
      r.append(l, control);
      p.appendChild(r);
    };
    const slider = (key: 'sandVolume' | 'ambientVolume' | 'musicVolume') => {
      const i = document.createElement('input');
      i.type = 'range';
      i.min = '0';
      i.max = '1';
      i.step = '0.01';
      i.value = String(s[key]);
      i.addEventListener('input', () => {
        s[key] = Number(i.value);
        this.changed();
      });
      return i;
    };
    const toggle = (key: 'chime' | 'gyro') => {
      const t = document.createElement('div');
      t.className = 'hg-switch' + (s[key] ? ' on' : '');
      t.addEventListener('click', () => {
        s[key] = !s[key];
        t.classList.toggle('on', s[key]);
        this.changed();
      });
      return t;
    };
    const select = <K extends 'ambient' | 'quality'>(key: K, opts: Array<[Settings[K], string]>) => {
      const el = document.createElement('select');
      for (const [v, t] of opts) {
        const o = document.createElement('option');
        o.value = String(v);
        o.textContent = t;
        el.appendChild(o);
      }
      el.value = String(s[key]);
      el.addEventListener('change', () => {
        (s as any)[key] = el.value;
        this.changed();
      });
      return el;
    };
    row('沙声', slider('sandVolume'));
    row(
      '环境音',
      select('ambient', [
        ['none', '无'],
        ['rain', '雨声'],
        ['fire', '壁炉'],
        ['noise', '白噪音'],
      ]),
    );
    row('环境音音量', slider('ambientVolume'));
    row('背景音乐', slider('musicVolume'));
    row('结束提示音', toggle('chime'));
    row('重力感应', toggle('gyro'));
    row(
      '画质',
      select('quality', [
        ['auto', '自动'],
        ['high', '高'],
        ['medium', '中'],
        ['low', '低'],
      ]),
    );
  }

  private changed() {
    saveSettings(this.settings);
    this.onSettings(this.settings);
  }
}
