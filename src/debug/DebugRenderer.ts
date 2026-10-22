// 调试渲染器：直接读物理缓冲画出颗粒团、沙粒和玻璃轮廓线，用来看清物理在干什么。

import { BULB_H, innerRadius, outerRadius } from '../geometry/profile';
import { identity, lookAt, perspective, type Mat4 } from '../math/mat4';
import type { SandSim } from '../sim/SandSim';
import src from './debug.wgsl?raw';

export class DebugRenderer {
  private ctx: GPUCanvasContext;
  private format: GPUTextureFormat;
  private depth: GPUTexture | null = null;
  private camBuf: GPUBuffer;
  private clumpPipe: GPURenderPipeline;
  private grainPipe: GPURenderPipeline;
  private linePipe: GPURenderPipeline;
  private bgClumps: GPUBindGroup;
  private bgGrains: GPUBindGroup;
  private lineBuf: GPUBuffer;
  private lineCount: number;

  yaw = 0;
  pitch = 0.12;
  distance = 0.5;
  target: [number, number, number] = [0, 0, 0];
  model: Mat4 = identity();

  constructor(private device: GPUDevice, private canvas: HTMLCanvasElement, private sim: SandSim) {
    this.ctx = canvas.getContext('webgpu')!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.camBuf = device.createBuffer({ size: 64 * 3 + 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ label: 'debug', code: src });
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const depthStencil: GPUDepthStencilState = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };
    this.clumpPipe = device.createRenderPipeline({
      layout: pl,
      vertex: { module, entryPoint: 'vsClump' },
      fragment: { module, entryPoint: 'fsClump', targets: [{ format: this.format }] },
      depthStencil,
    });
    this.grainPipe = device.createRenderPipeline({
      layout: pl,
      vertex: { module, entryPoint: 'vsGrain' },
      fragment: {
        module,
        entryPoint: 'fsGrain',
        targets: [
          {
            format: this.format,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            },
          },
        ],
      },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    });
    this.linePipe = device.createRenderPipeline({
      layout: pl,
      vertex: {
        module,
        entryPoint: 'vsLine',
        buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }],
      },
      fragment: { module, entryPoint: 'fsLine', targets: [{ format: this.format }] },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    });
    const B = sim.buf;
    this.bgClumps = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 1, resource: { buffer: B.pos } },
        { binding: 2, resource: { buffer: B.vel } },
      ],
    });
    this.bgGrains = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.camBuf } },
        { binding: 1, resource: { buffer: B.grains } },
        { binding: 2, resource: { buffer: B.gvel } },
      ],
    });
    const lines = this.profileLines();
    this.lineCount = lines.length / 3;
    this.lineBuf = device.createBuffer({ size: lines.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.lineBuf, 0, lines);
  }

  /** 玻璃内外轮廓在 x-y 平面和 z-y 平面的截线 */
  private profileLines(): Float32Array {
    const out: number[] = [];
    const N = 96;
    for (const radius of [innerRadius, outerRadius]) {
      for (const [ax, az] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ]) {
        for (const sy of [1, -1]) {
          for (let i = 0; i < N; i++) {
            const y0 = (i / N) * BULB_H;
            const y1 = ((i + 1) / N) * BULB_H;
            const r0 = radius(y0)[0];
            const r1 = radius(y1)[0];
            out.push(ax * r0, sy * y0, az * r0, ax * r1, sy * y1, az * r1);
          }
        }
      }
    }
    return new Float32Array(out);
  }

  private resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h || !this.depth) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.depth?.destroy();
      this.depth = this.device.createTexture({ size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
  }

  encode(encoder: GPUCommandEncoder) {
    this.resize();
    const w = this.canvas.width;
    const h = this.canvas.height;
    const fov = (28 * Math.PI) / 180;
    const cp = Math.cos(this.pitch);
    const eye: [number, number, number] = [
      this.target[0] + this.distance * cp * Math.sin(this.yaw),
      this.target[1] + this.distance * Math.sin(this.pitch),
      this.target[2] + this.distance * cp * Math.cos(this.yaw),
    ];
    const view = lookAt(eye, this.target, [0, 1, 0]);
    const proj = perspective(fov, w / h, 0.01, 5);
    const cam = new Float32Array(52);
    cam.set(view, 0);
    cam.set(proj, 16);
    cam.set(this.model, 32);
    cam.set([this.sim.p.rc, this.sim.mc, 5e-5, (2 * Math.tan(fov / 2)) / h], 48);
    this.device.queue.writeBuffer(this.camBuf, 0, cam);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view: this.ctx.getCurrentTexture().createView(), clearValue: [0.9, 0.89, 0.87, 1], loadOp: 'clear', storeOp: 'store' },
      ],
      depthStencilAttachment: { view: this.depth!.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    pass.setPipeline(this.clumpPipe);
    pass.setBindGroup(0, this.bgClumps);
    pass.draw(6, this.sim.p.clumpCap);
    pass.setPipeline(this.linePipe);
    pass.setBindGroup(0, this.bgClumps);
    pass.setVertexBuffer(0, this.lineBuf);
    pass.draw(this.lineCount);
    pass.setPipeline(this.grainPipe);
    pass.setBindGroup(0, this.bgGrains);
    pass.draw(6, this.sim.p.grainCap);
    pass.end();
  }
}
