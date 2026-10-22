// 沙子时钟：时间完全由流过孔口的沙量换算，沙停时间就停。
import { W0 } from '../physics';

export class SandClock {
  /** 已流过孔口的沙（mg），双精度累加每帧 GPU 读回的流量 */
  drained = 0;

  constructor(public totalMg: number) {}

  add(mg: number) {
    this.drained += mg;
  }

  get elapsedSeconds(): number {
    return this.drained / W0;
  }

  get remainingSeconds(): number {
    return Math.max(0, (this.totalMg - this.drained) / W0);
  }

  get done(): boolean {
    return this.drained >= this.totalMg * (1 - 1e-6);
  }
}

export function formatClock(seconds: number): string {
  // 向上取整，但给 10 ms 容差：实测沙量比目标多零点几毫克时不该多显示一秒
  const s = Math.max(0, Math.ceil(seconds - 0.01));
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}
