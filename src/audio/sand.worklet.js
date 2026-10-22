// 沙声合成器（AudioWorklet）：参数全部来自物理统计，不用录音素材。
// bed    连续沙沙底噪的增益（流量大时颗粒声密到连成一片）
// clicks 可分辨的颗粒噼啪声的速率（次/秒）；clickAmp 其振幅
// glass  沙粒撞玻璃的叮声速率（次/秒）
// slide  沙体成片滑动（翻转、倾斜、凝聚落沙）的柔和沙沙声增益

class SandProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'bed', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'clicks', defaultValue: 0, minValue: 0, maxValue: 20000, automationRate: 'k-rate' },
      { name: 'clickAmp', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'glass', defaultValue: 0, minValue: 0, maxValue: 5000, automationRate: 'k-rate' },
      { name: 'slide', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    this.s = 0x9e3779b9;
    this.voices = [];
    this.nextClick = 0;
    this.nextGlass = 0;
    // 底噪滤波器状态（左右声道各一套）
    this.hp = [0, 0];
    this.hpIn = [0, 0];
    this.lp = [0, 0];
    this.lp2 = [0, 0];
    this.flutter = 1;
    this.flutterT = 1;
    this.slideLp = [0, 0, 0, 0];
    this.slideBp = [0, 0];
  }

  rand() {
    let x = this.s;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.s = x >>> 0;
    return this.s / 4294967296;
  }

  /** 二阶共振器（阻尼正弦），由一个脉冲激励 */
  addVoice(freq, decay, amp, pan) {
    if (this.voices.length > 64) this.voices.shift();
    const r = Math.exp(-1 / (decay * sampleRate));
    const w = (2 * Math.PI * freq) / sampleRate;
    this.voices.push({ a1: 2 * r * Math.cos(w), a2: -r * r, y1: amp * Math.sin(w), y2: 0, life: Math.ceil(decay * sampleRate * 6), gl: Math.sqrt(1 - pan), gr: Math.sqrt(pan) });
  }

  process(_inputs, outputs, params) {
    const out = outputs[0];
    const L = out[0];
    const R = out[1] ?? out[0];
    const n = L.length;
    const bed = params.bed[0];
    const clickRate = params.clicks[0];
    const clickAmp = params.clickAmp[0];
    const glassRate = params.glass[0];
    const slide = params.slide[0];
    const sr = sampleRate;
    const kHp = Math.exp((-2 * Math.PI * 900) / sr);
    const kLp = 1 - Math.exp((-2 * Math.PI * 7000) / sr);
    const kLp2 = 1 - Math.exp((-2 * Math.PI * 5200) / sr);
    // 滑动声：0.9–3.5 kHz 的带通噪声（真实的沙体滑落是柔和的中高频沙沙声，不是低沉的隆隆声）
    const kSlHp = Math.exp((-2 * Math.PI * 900) / sr);
    const kSlLp = 1 - Math.exp((-2 * Math.PI * 3500) / sr);
    for (let i = 0; i < n; i++) {
      // 颗粒噼啪声：泊松过程
      if (clickRate > 0) {
        this.nextClick -= 1;
        while (this.nextClick <= 0) {
          this.nextClick += (-Math.log(1 - this.rand() + 1e-12) * sr) / clickRate;
          const a = clickAmp * Math.pow(this.rand(), 3);
          this.addVoice(2200 + 6500 * this.rand(), 0.0004 + 0.0012 * this.rand(), a, this.rand());
        }
      }
      // 撞玻璃：更清脆、衰减更长的几个模态
      if (glassRate > 0) {
        this.nextGlass -= 1;
        while (this.nextGlass <= 0) {
          this.nextGlass += (-Math.log(1 - this.rand() + 1e-12) * sr) / glassRate;
          const a = 0.05 * clickAmp * Math.pow(this.rand(), 2);
          const p = this.rand();
          this.addVoice(2150 * (1 + 0.02 * (this.rand() - 0.5)), 0.05, a, p);
          this.addVoice(3420 * (1 + 0.02 * (this.rand() - 0.5)), 0.04, a * 0.7, p);
          this.addVoice(5630 * (1 + 0.02 * (this.rand() - 0.5)), 0.028, a * 0.5, p);
        }
      }
      let vl = 0;
      let vr = 0;
      for (let k = this.voices.length - 1; k >= 0; k--) {
        const v = this.voices[k];
        const y = v.a1 * v.y1 + v.a2 * v.y2;
        v.y2 = v.y1;
        v.y1 = y;
        vl += y * v.gl;
        vr += y * v.gr;
        if (--v.life <= 0) this.voices.splice(k, 1);
      }
      // 连续底噪：白噪声 → 高通 → 两级低通，带随机的"呼吸"起伏
      if ((i & 63) === 0) this.flutterT = 0.7 + 0.6 * this.rand();
      this.flutter += (this.flutterT - this.flutter) * 0.004;
      let bl = 0;
      let br = 0;
      for (let c = 0; c < 2; c++) {
        const w = this.rand() * 2 - 1;
        const h = kHp * (this.hp[c] + w - this.hpIn[c]);
        this.hpIn[c] = w;
        this.hp[c] = h;
        this.lp[c] += (h - this.lp[c]) * kLp;
        this.lp2[c] += (this.lp[c] - this.lp2[c]) * kLp2;
        const b = this.lp2[c] * bed * this.flutter;
        if (c === 0) bl = b;
        else br = b;
      }
      // 滑动：带通噪声，跟着底噪的"呼吸"起伏
      let sl = 0;
      let sr2 = 0;
      if (slide > 0) {
        for (let c = 0; c < 2; c++) {
          const w = this.rand() * 2 - 1;
          const h = kSlHp * (this.slideLp[c] + w - this.slideLp[c + 2]);
          this.slideLp[c + 2] = w;
          this.slideLp[c] = h;
        }
        for (let c = 0; c < 2; c++) this.slideBp[c] += (this.slideLp[c] - this.slideBp[c]) * kSlLp;
        sl = this.slideBp[0] * slide * 1.6 * this.flutter;
        sr2 = this.slideBp[1] * slide * 1.6 * this.flutter;
      }
      L[i] = bl + vl + sl;
      R[i] = br + vr + sr2;
    }
    return true;
  }
}

registerProcessor('sand', SandProcessor);
