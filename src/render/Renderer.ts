// 正式渲染器：场景 → 沙面（impostor → 平滑 → 着色）→ 沙粒 → 玻璃正反面 → 合成与色调映射。

import { profileTable } from '../geometry/profile';
import { identity, invertRigid, lookAt, multiply, perspective, type Mat4, type Vec3 } from '../math/mat4';
import { BULB_H, CAP_R, CAP_T } from '../geometry/profile';
import { FX_CAP, type SandSim } from '../sim/SandSim';
import { capsMesh, glassMesh, type Mesh } from './meshes';
import { LightMap } from './LightMap';
import viewSrc from './wgsl/view.wgsl?raw';
import fullSrc from './wgsl/fullscreen.wgsl?raw';
import sceneSrc from './wgsl/scene.wgsl?raw';
import impostorSrc from './wgsl/sand_impostor.wgsl?raw';
import smoothSrc from './wgsl/sand_smooth.wgsl?raw';
import shadeSrc from './wgsl/sand_shade.wgsl?raw';
import grainsSrc from './wgsl/sand_grains.wgsl?raw';
import glassSrc from './wgsl/glass_mesh.wgsl?raw';
import compositeSrc from './wgsl/composite.wgsl?raw';
import fxSrc from './wgsl/fx.wgsl?raw';

const VIEW_BYTES = 6 * 64 + 5 * 16;

interface Targets {
  w: number;
  h: number;
  sceneColor: GPUTexture;
  sceneDepth: GPUTexture;
  sandRaw: GPUTexture;
  sandTmp: GPUTexture;
  sandSmooth: GPUTexture;
  sandId: GPUTexture;
  sandZ: GPUTexture;
  sandColor: GPUTexture;
  glassFront: GPUTexture;
  glassBack: GPUTexture;
  glassBackZ: GPUTexture;
  bg: Record<string, GPUBindGroup>;
}

export interface Camera {
  yaw: number;
  pitch: number;
  distance: number;
  target: Vec3;
  fovY: number;
}

export class Renderer {
  static compositeMode = 0;
  private ctx: GPUCanvasContext;
  private format: GPUTextureFormat;
  private viewBuf: GPUBuffer;
  private profBuf: GPUBuffer;
  readonly lightMap: LightMap;
  private samp: GPUSampler;
  private pipes: Record<string, GPURenderPipeline> = {};
  private glass: { vb: GPUBuffer; ib: GPUBuffer; n: number };
  private caps: { vb: GPUBuffer; ib: GPUBuffer; n: number };
  private t: Targets | null = null;
  private frame = 0;

  camera: Camera = { yaw: 0, pitch: 0.1, distance: 0.62, target: [0, 0.097, 0], fovY: (24 * Math.PI) / 180 };
  /** 沙漏局部系 → 世界 */
  model: Mat4 = identity();
  /** 指向主光的世界方向 */
  lightDir: Vec3 = [-0.45, 0.85, 0.35];
  lightIntensity = 1.0;
  exposure = 1.0;
  /** 内部渲染分辨率相对于设备像素的比例 */
  renderScale = 1.0;
  maxDpr = 2;
  /** 调试：只执行到第几个通道（1 场景 … 7 合成） */
  upto = 99;

  constructor(private device: GPUDevice, private canvas: HTMLCanvasElement, private sim: SandSim) {
    this.ctx = canvas.getContext('webgpu')!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.viewBuf = device.createBuffer({ size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const table = profileTable();
    this.profBuf = device.createBuffer({ size: table.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.profBuf, 0, table);
    this.samp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.lightMap = new LightMap(device, sim);
    this.glass = this.upload(glassMesh());
    this.caps = this.upload(capsMesh());
    this.buildPipelines();
  }

  private upload(m: Mesh) {
    const vb = this.device.createBuffer({ size: m.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(vb, 0, m.vertices);
    const ib = this.device.createBuffer({ size: m.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(ib, 0, m.indices);
    return { vb, ib, n: m.indices.length };
  }

  private module(label: string, ...srcs: string[]) {
    const code = [viewSrc, ...srcs].join('\n');
    const m = this.device.createShaderModule({ label, code });
    m.getCompilationInfo().then((info) => {
      for (const msg of info.messages) {
        const line = code.split('\n')[msg.lineNum - 1] ?? '';
        (msg.type === 'error' ? console.error : console.warn)(`[${label}] ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}\n  > ${line.trim()}`);
      }
    });
    return m;
  }

  private buildPipelines() {
    const d = this.device;
    const meshLayout: GPUVertexBufferLayout = {
      arrayStride: 24,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
      ],
    };
    const scene = this.module('scene', sceneSrc);
    this.pipes.backdrop = d.createRenderPipeline({
      label: 'backdrop',
      layout: 'auto',
      vertex: { module: scene, entryPoint: 'vsFull' },
      fragment: { module: scene, entryPoint: 'fsBackdrop', targets: [{ format: 'rgba16float' }] },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'always' },
    });
    this.pipes.caps = d.createRenderPipeline({
      label: 'caps',
      layout: 'auto',
      vertex: { module: scene, entryPoint: 'vsCap', buffers: [meshLayout] },
      fragment: { module: scene, entryPoint: 'fsCap', targets: [{ format: 'rgba16float' }] },
      primitive: { cullMode: 'back' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    });
    const imp = this.module('impostor', impostorSrc);
    this.pipes.impostor = d.createRenderPipeline({
      label: 'impostor',
      layout: 'auto',
      vertex: { module: imp, entryPoint: 'vs' },
      fragment: { module: imp, entryPoint: 'fs', targets: [{ format: 'r32float' }, { format: 'r32uint' }] },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    const sm = this.module('smooth', fullSrc, smoothSrc);
    for (const [name, dx, dy] of [
      ['smoothH', 1, 0],
      ['smoothV', 0, 1],
    ] as const) {
      this.pipes[name] = d.createRenderPipeline({
        label: name,
        layout: 'auto',
        vertex: { module: sm, entryPoint: 'vsFull' },
        fragment: { module: sm, entryPoint: 'fs', targets: [{ format: 'r32float' }], constants: { DIR_X: dx, DIR_Y: dy } },
      });
    }
    const sh = this.module('shade', fullSrc, shadeSrc);
    this.pipes.shade = d.createRenderPipeline({
      label: 'shade',
      layout: 'auto',
      vertex: { module: sh, entryPoint: 'vsFull' },
      fragment: { module: sh, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
    });
    const gr = this.module('grains', grainsSrc);
    this.pipes.grains = d.createRenderPipeline({
      label: 'grains',
      layout: 'auto',
      vertex: { module: gr, entryPoint: 'vs' },
      fragment: {
        module: gr,
        entryPoint: 'fs',
        targets: [
          {
            format: 'rgba16float',
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            },
          },
        ],
      },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    });
    const fx = this.module('fx', fxSrc);
    this.pipes.fx = d.createRenderPipeline({
      label: 'fx',
      layout: 'auto',
      vertex: { module: fx, entryPoint: 'vs' },
      fragment: {
        module: fx,
        entryPoint: 'fs',
        targets: [{ format: 'rgba16float', blend: { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'zero', dstFactor: 'one' } } }],
      },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    });
    const gl = this.module('glass', glassSrc);
    this.pipes.glassFront = d.createRenderPipeline({
      label: 'glassFront',
      layout: 'auto',
      vertex: { module: gl, entryPoint: 'vs', buffers: [meshLayout] },
      fragment: { module: gl, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { cullMode: 'back' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'less' },
    });
    this.pipes.glassBack = d.createRenderPipeline({
      label: 'glassBack',
      layout: 'auto',
      vertex: { module: gl, entryPoint: 'vs', buffers: [meshLayout] },
      fragment: { module: gl, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
      primitive: { cullMode: 'front' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    const co = this.module('composite', fullSrc, compositeSrc);
    this.pipes.composite = d.createRenderPipeline({
      label: 'composite',
      layout: 'auto',
      vertex: { module: co, entryPoint: 'vsFull' },
      fragment: { module: co, entryPoint: 'fs', targets: [{ format: this.format }], constants: { MODE: Renderer.compositeMode } },
    });
  }

  private makeTargets(w: number, h: number): Targets {
    const d = this.device;
    const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const tex = (format: GPUTextureFormat, usage = RT) => d.createTexture({ size: [w, h], format, usage });
    const t: Targets = {
      w,
      h,
      sceneColor: tex('rgba16float'),
      sceneDepth: tex('depth32float'),
      sandRaw: tex('r32float'),
      sandTmp: tex('r32float'),
      sandSmooth: tex('r32float'),
      sandId: tex('r32uint'),
      sandZ: tex('depth24plus', GPUTextureUsage.RENDER_ATTACHMENT),
      sandColor: tex('rgba16float'),
      glassFront: tex('rgba16float'),
      glassBack: tex('rgba16float'),
      glassBackZ: tex('depth24plus', GPUTextureUsage.RENDER_ATTACHMENT),
      bg: {},
    };
    const ub = [
      { binding: 0, resource: { buffer: this.viewBuf } },
      { binding: 1, resource: { buffer: this.profBuf } },
    ];
    const P = this.pipes;
    const B = this.sim.buf;
    const bg = (pipe: GPURenderPipeline, extra: GPUBindGroupEntry[], withProfile = true) =>
      d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [...(withProfile ? ub : ub.slice(0, 1)), ...extra] });
    t.bg.backdrop = bg(
      P.backdrop,
      [
        { binding: 2, resource: this.lightMap.tex.createView() },
        { binding: 3, resource: this.samp },
      ],
      true,
    );
    t.bg.caps = d.createBindGroup({ layout: P.caps.getBindGroupLayout(0), entries: [ub[0]] });
    t.bg.impostor = d.createBindGroup({ layout: P.impostor.getBindGroupLayout(0), entries: [ub[0], { binding: 2, resource: { buffer: B.pos } }] });
    t.bg.smoothH = bg(P.smoothH, [{ binding: 2, resource: t.sandRaw.createView() }], false);
    t.bg.smoothV = bg(P.smoothV, [{ binding: 2, resource: t.sandTmp.createView() }], false);
    t.bg.shade = d.createBindGroup({
      layout: P.shade.getBindGroupLayout(0),
      entries: [
        ub[0],
        { binding: 2, resource: t.sandSmooth.createView() },
        { binding: 3, resource: t.sandId.createView() },
        { binding: 4, resource: { buffer: B.pos } },
        { binding: 5, resource: { buffer: B.vel } },
      ],
    });
    t.bg.grains = d.createBindGroup({
      layout: P.grains.getBindGroupLayout(0),
      entries: [ub[0], { binding: 2, resource: { buffer: B.grains } }, { binding: 3, resource: { buffer: B.gvel } }],
    });
    t.bg.fx = d.createBindGroup({
      layout: P.fx.getBindGroupLayout(0),
      entries: [ub[0], { binding: 2, resource: { buffer: B.fxPos } }, { binding: 3, resource: { buffer: B.fxVel } }],
    });
    t.bg.glassFront = d.createBindGroup({ layout: P.glassFront.getBindGroupLayout(0), entries: [ub[0]] });
    t.bg.glassBack = d.createBindGroup({ layout: P.glassBack.getBindGroupLayout(0), entries: [ub[0]] });
    t.bg.composite = d.createBindGroup({
      layout: P.composite.getBindGroupLayout(0),
      entries: [
        ...ub,
        { binding: 2, resource: t.sceneColor.createView() },
        { binding: 3, resource: t.sceneDepth.createView() },
        { binding: 4, resource: t.sandColor.createView() },
        { binding: 5, resource: t.sandSmooth.createView() },
        { binding: 6, resource: t.glassFront.createView() },
        { binding: 7, resource: t.glassBack.createView() },
        { binding: 8, resource: this.samp },
      ],
    });
    return t;
  }

  private ensureTargets(): Targets {
    const dpr = Math.min(window.devicePixelRatio || 1, this.maxDpr) * this.renderScale;
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (!this.t || this.t.w !== w || this.t.h !== h) {
      if (this.t) {
        for (const v of Object.values(this.t)) if (v instanceof GPUTexture) v.destroy();
      }
      this.canvas.width = w;
      this.canvas.height = h;
      this.t = this.makeTargets(w, h);
    }
    return this.t;
  }

  /** 屏幕坐标（CSS 像素）是否落在沙漏上：把沙漏当成屏幕上的一个胶囊体 */
  hitHourglass(x: number, y: number): boolean {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    const view = lookAt(this.eye(), this.camera.target, [0, 1, 0]);
    const vp = multiply(perspective(this.camera.fovY, w / h, 0.02, 10), view);
    const toScreen = (lx: number, ly: number, lz: number) => {
      const m = this.model;
      const wx = m[0] * lx + m[4] * ly + m[8] * lz + m[12];
      const wy = m[1] * lx + m[5] * ly + m[9] * lz + m[13];
      const wz = m[2] * lx + m[6] * ly + m[10] * lz + m[14];
      const cx = vp[0] * wx + vp[4] * wy + vp[8] * wz + vp[12];
      const cy = vp[1] * wx + vp[5] * wy + vp[9] * wz + vp[13];
      const cw = vp[3] * wx + vp[7] * wy + vp[11] * wz + vp[15];
      return [((cx / cw + 1) / 2) * w, ((1 - cy / cw) / 2) * h, cw];
    };
    const a = toScreen(0, BULB_H + CAP_T, 0);
    const b = toScreen(0, -BULB_H - CAP_T, 0);
    const depth = (a[2] + b[2]) / 2;
    const r = ((CAP_R * 1.2) / (2 * Math.tan(this.camera.fovY / 2) * depth)) * h;
    const abx = b[0] - a[0];
    const aby = b[1] - a[1];
    const t = Math.max(0, Math.min(1, ((x - a[0]) * abx + (y - a[1]) * aby) / (abx * abx + aby * aby || 1)));
    return Math.hypot(x - (a[0] + abx * t), y - (a[1] + aby * t)) < r;
  }

  size(): string {
    return this.t ? `${this.t.w}×${this.t.h}` : '-';
  }

  /** 相机的世界位置 */
  eye(): Vec3 {
    const c = this.camera;
    const cp = Math.cos(c.pitch);
    return [c.target[0] + c.distance * cp * Math.sin(c.yaw), c.target[1] + c.distance * Math.sin(c.pitch), c.target[2] + c.distance * cp * Math.cos(c.yaw)];
  }

  private writeView(t: Targets, time: number) {
    const eye = this.eye();
    const view = lookAt(eye, this.camera.target, [0, 1, 0]);
    const proj = perspective(this.camera.fovY, t.w / t.h, 0.02, 10);
    const projInv = invertPerspective(proj);
    const data = new Float32Array(VIEW_BYTES / 4);
    data.set(view, 0);
    data.set(proj, 16);
    data.set(invertRigid(view), 32);
    data.set(projInv, 48);
    data.set(this.model, 64);
    data.set(invertRigid(this.model), 80);
    data.set([...eye, 1], 96);
    data.set([t.w, t.h, 1 / t.w, 1 / t.h], 100);
    data.set([this.sim.p.rc, this.sim.mc, 5e-5, (2 * Math.tan(this.camera.fovY / 2)) / t.h], 104);
    const l = this.lightDir;
    const ll = Math.hypot(l[0], l[1], l[2]);
    data.set([l[0] / ll, l[1] / ll, l[2] / ll, this.lightIntensity], 108);
    data.set([time, this.frame, this.exposure, 0], 112);
    this.device.queue.writeBuffer(this.viewBuf, 0, data);
  }

  encode(encoder: GPUCommandEncoder, time: number) {
    const t = this.ensureTargets();
    this.frame++;
    this.writeView(t, time);
    this.lightMap.encode(encoder, this.model, this.lightDir);
    const P = this.pipes;
    const view = (x: GPUTexture) => x.createView();

    // 1. 场景：背景、桌面、端盖
    let pass = encoder.beginRenderPass({
      colorAttachments: [{ view: view(t.sceneColor), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      depthStencilAttachment: { view: view(t.sceneDepth), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    pass.setPipeline(P.backdrop);
    pass.setBindGroup(0, t.bg.backdrop);
    pass.draw(3);
    pass.setPipeline(P.caps);
    pass.setBindGroup(0, t.bg.caps);
    pass.setVertexBuffer(0, this.caps.vb);
    pass.setIndexBuffer(this.caps.ib, 'uint32');
    pass.drawIndexed(this.caps.n);
    pass.end();
    if (this.upto < 2) return;

    // 2. 沙面 impostor：视空间距离 + 颗粒团 id
    pass = encoder.beginRenderPass({
      colorAttachments: [
        { view: view(t.sandRaw), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        { view: view(t.sandId), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
      ],
      depthStencilAttachment: { view: view(t.sandZ), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    pass.setPipeline(P.impostor);
    pass.setBindGroup(0, t.bg.impostor);
    pass.draw(6, this.sim.p.clumpCap);
    pass.end();
    if (this.upto < 3) return;

    // 3. 平滑（横、竖两遍）
    for (const [pipe, bgName, dst] of [
      [P.smoothH, 'smoothH', t.sandTmp],
      [P.smoothV, 'smoothV', t.sandSmooth],
    ] as const) {
      pass = encoder.beginRenderPass({ colorAttachments: [{ view: view(dst), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, t.bg[bgName]);
      pass.draw(3);
      pass.end();
    }

    if (this.upto < 4) return;
    // 4. 沙面着色 + 5. 沙粒（与沙面做深度测试）
    pass = encoder.beginRenderPass({
      colorAttachments: [{ view: view(t.sandColor), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
    });
    pass.setPipeline(P.shade);
    pass.setBindGroup(0, t.bg.shade);
    pass.draw(3);
    pass.end();
    if (this.upto < 5) return;
    pass = encoder.beginRenderPass({
      colorAttachments: [{ view: view(t.sandColor), loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: view(t.sandZ), depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    pass.setPipeline(P.grains);
    pass.setBindGroup(0, t.bg.grains);
    pass.draw(6, this.sim.p.grainCap);
    pass.setPipeline(P.fx);
    pass.setBindGroup(0, t.bg.fx);
    pass.draw(6, FX_CAP);
    pass.end();

    if (this.upto < 6) return;
    // 6. 玻璃正面（被端盖挡住的部分不画）和背面
    pass = encoder.beginRenderPass({
      colorAttachments: [{ view: view(t.glassFront), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
      depthStencilAttachment: { view: view(t.sceneDepth), depthLoadOp: 'load', depthStoreOp: 'store', depthReadOnly: false },
    });
    pass.setPipeline(P.glassFront);
    pass.setBindGroup(0, t.bg.glassFront);
    pass.setVertexBuffer(0, this.glass.vb);
    pass.setIndexBuffer(this.glass.ib, 'uint32');
    pass.drawIndexed(this.glass.n);
    pass.end();
    pass = encoder.beginRenderPass({
      colorAttachments: [{ view: view(t.glassBack), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
      depthStencilAttachment: { view: view(t.glassBackZ), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    pass.setPipeline(P.glassBack);
    pass.setBindGroup(0, t.bg.glassBack);
    pass.setVertexBuffer(0, this.glass.vb);
    pass.setIndexBuffer(this.glass.ib, 'uint32');
    pass.drawIndexed(this.glass.n);
    pass.end();

    if (this.upto < 7) return;
    // 7. 合成到画布
    pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(P.composite);
    pass.setBindGroup(0, t.bg.composite);
    pass.draw(3);
    pass.end();
  }
}

/** 透视投影矩阵的逆（只用于还原视空间方向） */
function invertPerspective(p: Mat4): Mat4 {
  const o = new Float32Array(16);
  const a = p[0];
  const b = p[5];
  const c = p[10];
  const d = p[14];
  o[0] = 1 / a;
  o[5] = 1 / b;
  o[11] = 1 / d;
  o[14] = -1;
  o[15] = c / d;
  return o;
}
