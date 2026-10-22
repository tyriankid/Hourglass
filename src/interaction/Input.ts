// 输入：从沙漏上开始拖 = 倾斜（松手回摆）；从背景开始拖 = 环绕相机；双指/滚轮 = 缩放；
// 轻点沙漏 = 开始/暂停/继续；陀螺仪（可选）把手机倾斜映射成沙漏的小角度倾斜。

import type { Renderer } from '../render/Renderer';
import type { HourglassMotion } from './Motion';

const TILT_PER_PX = 0.006;
const MAX_TILT = 0.7;

export class Input {
  onTapHourglass: () => void = () => {};
  onTapBackground: () => void = () => {};

  private pointers = new Map<number, { x: number; y: number; x0: number; y0: number; t0: number; onGlass: boolean }>();
  private pinch0 = 0;
  private dist0 = 0;
  private yawVel = 0;
  private pitchVel = 0;
  private dragTilt: [number, number] | null = null;
  private gyroTilt: [number, number] = [0, 0];
  private gyroRef: [number, number] | null = null;
  private gyroOn = false;
  private keyTilt: [number, number] = [0, 0];

  constructor(private canvas: HTMLCanvasElement, private renderer: Renderer, private motion: HourglassMotion) {
    canvas.addEventListener('pointerdown', (e) => this.down(e));
    canvas.addEventListener('pointermove', (e) => this.move(e));
    canvas.addEventListener('pointerup', (e) => this.up(e));
    canvas.addEventListener('pointercancel', (e) => this.up(e, true));
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        this.zoom(Math.exp(e.deltaY * 0.001));
      },
      { passive: false },
    );
    window.addEventListener('keydown', (e) => {
      if (e.key === ' ') {
        e.preventDefault();
        this.onTapHourglass();
      }
      if (e.key === 'ArrowLeft') this.keyTilt[0] = 0.25;
      if (e.key === 'ArrowRight') this.keyTilt[0] = -0.25;
      if (e.key === 'ArrowUp') this.keyTilt[1] = -0.25;
      if (e.key === 'ArrowDown') this.keyTilt[1] = 0.25;
    });
    window.addEventListener('keyup', (e) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') this.keyTilt[0] = 0;
      if (e.key === 'ArrowUp' || e.key === 'ArrowDown') this.keyTilt[1] = 0;
    });
  }

  private zoom(f: number) {
    const c = this.renderer.camera;
    c.distance = Math.min(1.0, Math.max(0.28, c.distance * f));
  }

  private down(e: PointerEvent) {
    this.canvas.setPointerCapture(e.pointerId);
    const onGlass = this.renderer.hitHourglass(e.clientX, e.clientY);
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, t0: performance.now(), onGlass });
    if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      this.pinch0 = Math.hypot(a.x - b.x, a.y - b.y);
      this.dist0 = this.renderer.camera.distance;
      this.dragTilt = null;
    } else if (onGlass) {
      this.dragTilt = [0, 0];
    }
  }

  private move(e: PointerEvent) {
    const p = this.pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x;
    const dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (this.pinch0 > 0) this.renderer.camera.distance = Math.min(1.0, Math.max(0.28, (this.dist0 * this.pinch0) / d));
      return;
    }
    if (p.onGlass && this.dragTilt) {
      // 抓住沙漏拖动：水平拖 = 绕视线轴倾倒，竖直拖 = 朝向/背离相机倾倒
      this.dragTilt = [
        Math.max(-MAX_TILT, Math.min(MAX_TILT, -(e.clientX - p.x0) * TILT_PER_PX)),
        Math.max(-MAX_TILT, Math.min(MAX_TILT, (e.clientY - p.y0) * TILT_PER_PX)),
      ];
    } else {
      const c = this.renderer.camera;
      c.yaw -= dx * 0.006;
      c.pitch = Math.max(-0.12, Math.min(0.95, c.pitch + dy * 0.004));
      this.yawVel = -dx * 0.006 * 60;
      this.pitchVel = dy * 0.004 * 60;
    }
  }

  private up(e: PointerEvent, cancel = false) {
    const p = this.pointers.get(e.pointerId);
    this.pointers.delete(e.pointerId);
    if (this.pointers.size < 2) this.pinch0 = 0;
    if (!p) return;
    if (p.onGlass) this.dragTilt = null;
    const moved = Math.hypot(e.clientX - p.x0, e.clientY - p.y0);
    if (!cancel && moved < 8 && performance.now() - p.t0 < 350) {
      this.yawVel = 0;
      this.pitchVel = 0;
      if (p.onGlass) this.onTapHourglass();
      else this.onTapBackground();
    }
  }

  /** 开启陀螺仪；iOS 需要在用户手势里请求授权 */
  async enableGyro(on: boolean): Promise<boolean> {
    this.gyroOn = on;
    this.gyroRef = null;
    this.gyroTilt = [0, 0];
    if (!on) return false;
    const DOE = (window as any).DeviceOrientationEvent;
    if (DOE && typeof DOE.requestPermission === 'function') {
      try {
        if ((await DOE.requestPermission()) !== 'granted') return false;
      } catch {
        return false;
      }
    }
    window.addEventListener('deviceorientation', this.onOrient);
    return true;
  }

  private onOrient = (e: DeviceOrientationEvent) => {
    if (!this.gyroOn || e.beta == null || e.gamma == null) return;
    if (!this.gyroRef) this.gyroRef = [e.gamma, e.beta];
    const k = (Math.PI / 180) * 0.8;
    this.gyroTilt = [
      Math.max(-0.6, Math.min(0.6, -(e.gamma - this.gyroRef[0]) * k)),
      Math.max(-0.6, Math.min(0.6, (e.beta - this.gyroRef[1]) * k)),
    ];
  };

  /** 每帧：相机惯性；把屏幕空间的倾斜换成世界系旋转向量，交给沙漏弹簧 */
  update(dt: number) {
    const c = this.renderer.camera;
    if (this.pointers.size === 0) {
      c.yaw += this.yawVel * dt;
      c.pitch = Math.max(-0.12, Math.min(0.95, c.pitch + this.pitchVel * dt));
      const k = Math.exp(-dt * 3.5);
      this.yawVel *= k;
      this.pitchVel *= k;
    }
    const t = this.dragTilt ?? [this.gyroTilt[0] + this.keyTilt[0], this.gyroTilt[1] + this.keyTilt[1]];
    // a1 绕水平视线轴，a2 绕相机右轴
    const fx = Math.sin(c.yaw);
    const fz = Math.cos(c.yaw);
    const rx = Math.cos(c.yaw);
    const rz = -Math.sin(c.yaw);
    this.motion.tiltTarget = [fx * t[0] + rx * t[1], fz * t[0] + rz * t[1]];
  }
}
