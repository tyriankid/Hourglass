// 环境音合成器（AudioWorklet），实时生成不循环：
// mode 0 无，1 雨声，2 壁炉，3 柔和的白噪音

class AmbientProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'mode', defaultValue: 0, minValue: 0, maxValue: 3, automationRate: 'k-rate' },
      { name: 'gain', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    this.s = 0x2545f491;
    this.pink = [new Float32Array(7), new Float32Array(7)];
    this.lp = [0, 0, 0, 0, 0, 0];
    this.voices = [];
    this.nextEvent = 0;
    this.fireMod = 1;
    this.fireModT = 1;
    this.crackleBurst = 0;
  }

  rand() {
    let x = this.s;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.s = x >>> 0;
    return this.s / 4294967296;
  }

  /** Paul Kellet 的粉红噪声近似 */
  pinkNoise(c) {
    const b = this.pink[c];
    const w = this.rand() * 2 - 1;
    b[0] = 0.99886 * b[0] + w * 0.0555179;
    b[1] = 0.99332 * b[1] + w * 0.0750759;
    b[2] = 0.969 * b[2] + w * 0.153852;
    b[3] = 0.8665 * b[3] + w * 0.3104856;
    b[4] = 0.55 * b[4] + w * 0.5329522;
    b[5] = -0.7616 * b[5] - w * 0.016898;
    const out = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362;
    b[6] = w * 0.115926;
    return out * 0.11;
  }

  addVoice(freq, decay, amp, pan) {
    if (this.voices.length > 48) this.voices.shift();
    const r = Math.exp(-1 / (decay * sampleRate));
    const w = (2 * Math.PI * freq) / sampleRate;
    this.voices.push({ a1: 2 * r * Math.cos(w), a2: -r * r, y1: amp * Math.sin(w), y2: 0, life: Math.ceil(decay * sampleRate * 6), gl: Math.sqrt(1 - pan), gr: Math.sqrt(pan) });
  }

  process(_inputs, outputs, params) {
    const out = outputs[0];
    const L = out[0];
    const R = out[1] ?? out[0];
    const n = L.length;
    const mode = Math.round(params.mode[0]);
    const gain = params.gain[0];
    if (mode === 0 || gain <= 0) {
      L.fill(0);
      R.fill(0);
      this.voices.length = 0;
      return true;
    }
    const sr = sampleRate;
    const k = (fc) => 1 - Math.exp((-2 * Math.PI * fc) / sr);
    for (let i = 0; i < n; i++) {
      let l = 0;
      let r = 0;
      if (mode === 1) {
        // 雨：粉红噪声底 + 随机雨滴（短促共振）
        for (let c = 0; c < 2; c++) {
          this.lp[c] += (this.pinkNoise(c) - this.lp[c]) * k(4500);
        }
        l = this.lp[0] * 0.9;
        r = this.lp[1] * 0.9;
        this.nextEvent -= 1;
        while (this.nextEvent <= 0) {
          this.nextEvent += (-Math.log(1 - this.rand() + 1e-12) * sr) / 55;
          const heavy = this.rand() < 0.08;
          this.addVoice(heavy ? 900 + 900 * this.rand() : 1800 + 3800 * this.rand(), heavy ? 0.012 : 0.004 + 0.006 * this.rand(), (heavy ? 0.12 : 0.05) * this.rand() ** 2, this.rand());
        }
      } else if (mode === 2) {
        // 壁炉：低频隆隆（缓慢起伏）+ 细碎的嘶嘶 + 成串的噼啪
        if ((i & 255) === 0) this.fireModT = 0.6 + 0.8 * this.rand();
        this.fireMod += (this.fireModT - this.fireMod) * 0.0008;
        for (let c = 0; c < 2; c++) {
          const w = this.rand() * 2 - 1;
          this.lp[c] += (w - this.lp[c]) * k(180);
          this.lp[c + 2] += (this.lp[c] - this.lp[c + 2]) * k(180);
          this.lp[c + 4] += (w - this.lp[c + 4]) * k(3000);
        }
        l = this.lp[2] * 3.2 * this.fireMod + (this.rand() * 2 - 1 - this.lp[4]) * 0.012 * this.fireMod;
        r = this.lp[3] * 3.2 * this.fireMod + (this.rand() * 2 - 1 - this.lp[5]) * 0.012 * this.fireMod;
        this.nextEvent -= 1;
        while (this.nextEvent <= 0) {
          this.nextEvent += (-Math.log(1 - this.rand() + 1e-12) * sr) / 6;
          this.crackleBurst = 2 + Math.floor(this.rand() * 9);
        }
        if (this.crackleBurst > 0 && this.rand() < 0.002) {
          this.crackleBurst--;
          this.addVoice(1500 + 5500 * this.rand(), 0.0006 + 0.002 * this.rand(), 0.25 * this.rand() ** 2, this.rand());
        }
      } else {
        // 柔和的白噪音：粉红噪声再压一点高频
        for (let c = 0; c < 2; c++) {
          this.lp[c] += (this.pinkNoise(c) - this.lp[c]) * k(2600);
        }
        l = this.lp[0] * 1.1;
        r = this.lp[1] * 1.1;
      }
      for (let v = this.voices.length - 1; v >= 0; v--) {
        const o = this.voices[v];
        const y = o.a1 * o.y1 + o.a2 * o.y2;
        o.y2 = o.y1;
        o.y1 = y;
        l += y * o.gl;
        r += y * o.gr;
        if (--o.life <= 0) this.voices.splice(v, 1);
      }
      L[i] = l * gain;
      R[i] = r * gain;
    }
    return true;
  }
}

registerProcessor('ambient', AmbientProcessor);
