// 桌面光照贴图：光子追踪分摊到多帧，一轮追完后收尾成纹理（r 主光系数，b 接触阴影）。

import { BULB_H, CAP_R, CAP_T } from '../geometry/profile';
import { cross, invertRigid, normalize, type Mat4, type Vec3 } from '../math/mat4';
import type { SandSim } from '../sim/SandSim';
import typesSrc from '../sim/wgsl/types.wgsl?raw';
import fnsSrc from '../sim/wgsl/fns.wgsl?raw';
import photonsSrc from './wgsl/photons.wgsl?raw';
import resolveSrc from './wgsl/lm_resolve.wgsl?raw';

export const LIGHTMAP_HALF = 0.22;

export class LightMap {
  readonly tex: GPUTexture;
  private acc: GPUBuffer;
  private ubuf: GPUBuffer;
  private trace: GPUComputePipeline;
  private resolve: GPUComputePipeline;
  private clear: GPUComputePipeline;
  private bgTrace: GPUBindGroup;
  private bgResolve: GPUBindGroup;
  private bgClear: GPUBindGroup;
  private batch = 0;
  private lastMoving = false;
  private cycleModel: Mat4 | null = null;
  private cycleLight: Vec3 = [0, 1, 0];

  /** n：每边光子数；batches：一轮分几帧 */
  constructor(private device: GPUDevice, sim: SandSim, readonly res = 256, public n = 768, public batches = 16) {
    this.tex = device.createTexture({
      size: [res, res],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    // 第一轮追完之前：全亮、无遮挡
    const init = new Uint16Array(res * res * 4);
    for (let i = 0; i < res * res; i++) init[i * 4] = 0x3c00;
    device.queue.writeTexture({ texture: this.tex }, init, { bytesPerRow: res * 8 }, [res, res]);
    this.acc = device.createBuffer({ size: res * res * 4, usage: GPUBufferUsage.STORAGE });
    this.ubuf = device.createBuffer({ size: 16 * 4 * 2 + 16 * 6, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const tm = device.createShaderModule({ label: 'photons', code: [typesSrc, fnsSrc, photonsSrc].join('\n') });
    tm.getCompilationInfo().then((i) => i.messages.forEach((m) => console[m.type === 'error' ? 'error' : 'warn'](`[photons] ${m.lineNum}:${m.linePos} ${m.message}`)));
    this.trace = device.createComputePipeline({ label: 'photons', layout: 'auto', compute: { module: tm, entryPoint: 'main' } });
    const rm = device.createShaderModule({ label: 'lmResolve', code: resolveSrc });
    rm.getCompilationInfo().then((i) => i.messages.forEach((m) => console[m.type === 'error' ? 'error' : 'warn'](`[lmResolve] ${m.lineNum}:${m.linePos} ${m.message}`)));
    this.resolve = device.createComputePipeline({ label: 'lmResolve', layout: 'auto', compute: { module: rm, entryPoint: 'main' } });
    this.clear = device.createComputePipeline({ label: 'lmClear', layout: 'auto', compute: { module: rm, entryPoint: 'main', constants: { CLEAR: 1 } } });
    this.bgTrace = device.createBindGroup({
      layout: this.trace.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: { buffer: sim.buf.P } },
        { binding: 2, resource: { buffer: sim.buf.U, offset: 0, size: 160 } },
        { binding: 3, resource: { buffer: sim.buf.grid } },
        { binding: 4, resource: { buffer: this.acc } },
      ],
    });
    this.bgResolve = device.createBindGroup({
      layout: this.resolve.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: { buffer: this.acc } },
        { binding: 2, resource: this.tex.createView() },
      ],
    });
    this.bgClear = device.createBindGroup({
      layout: this.clear.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ubuf } },
        { binding: 1, resource: { buffer: this.acc } },
        { binding: 2, resource: this.tex.createView() },
      ],
    });
    const e = device.createCommandEncoder();
    const p = e.beginComputePass();
    p.setPipeline(this.clear);
    p.setBindGroup(0, this.bgClear);
    p.dispatchWorkgroups(Math.ceil(res / 8), Math.ceil(res / 8));
    p.end();
    device.queue.submit([e.finish()]);
  }

  private writeUniforms(model: Mat4, light: Vec3, start: number, count: number) {
    const L = normalize(light);
    const pu = normalize(cross(L, Math.abs(L[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]));
    const pv = cross(L, pu);
    const d = new Float32Array(16 * 2 + 4 * 6);
    d.set(model, 0);
    d.set(invertRigid(model), 16);
    d.set([...L, 0], 32);
    d.set([...pu, 0], 36);
    d.set([...pv, 0], 40);
    d.set([model[12], model[13], model[14], 0.13], 44);
    new Uint32Array(d.buffer).set([this.cycleN, start, count, this.res], 48);
    d.set([LIGHTMAP_HALF, CAP_T, CAP_R, BULB_H + CAP_T / 2], 52);
    this.device.queue.writeBuffer(this.ubuf, 0, d);
  }

  private cycleN = 0;
  private cycleBatches = 1;

  /** 每帧调用：追一批光子；一轮结束时收尾成纹理并清零累加器。沙漏在动时用低分辨率、2 帧一轮，影子跟得上 */
  encode(encoder: GPUCommandEncoder, model: Mat4, light: Vec3) {
    const moving = this.cycleModel !== null && model.some((v, i) => Math.abs(v - this.cycleModel![i]) > 1e-4);
    if (moving && this.batch > 0 && this.cycleBatches > 2) {
      // 姿态变了：放弃这一轮，清零重来
      this.batch = 0;
      const g = Math.ceil(this.res / 8);
      const p = encoder.beginComputePass();
      p.setPipeline(this.clear);
      p.setBindGroup(0, this.bgClear);
      p.dispatchWorkgroups(g, g);
      p.end();
    }
    if (this.batch === 0) {
      this.cycleModel = new Float32Array(model);
      this.cycleLight = [...light];
      const fast = moving || this.lastMoving;
      this.cycleN = fast ? Math.round(this.n / 2) : this.n;
      this.cycleBatches = fast ? 2 : this.batches;
    }
    this.lastMoving = moving;
    const total = this.cycleN * this.cycleN;
    const per = Math.ceil(total / this.cycleBatches);
    const start = this.batch * per;
    const count = Math.max(0, Math.min(per, total - start));
    this.writeUniforms(this.cycleModel!, this.cycleLight, start, count);
    const pass = encoder.beginComputePass({ label: 'lightmap' });
    pass.setPipeline(this.trace);
    pass.setBindGroup(0, this.bgTrace);
    pass.dispatchWorkgroups(Math.max(1, Math.ceil(count / 64)));
    this.batch++;
    if (this.batch >= this.cycleBatches) {
      this.batch = 0;
      const g = Math.ceil(this.res / 8);
      pass.setPipeline(this.resolve);
      pass.setBindGroup(0, this.bgResolve);
      pass.dispatchWorkgroups(g, g);
      pass.setPipeline(this.clear);
      pass.setBindGroup(0, this.bgClear);
      pass.dispatchWorkgroups(g, g);
    }
    pass.end();
  }
}
