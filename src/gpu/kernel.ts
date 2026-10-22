// compute 内核的小工具：每个内核只声明自己用到的缓冲，绑定号按列表顺序生成，
// WGSL 声明和 TS 侧的绑定组来自同一份列表，不会对不上。
// 每个着色器阶段的存储缓冲不超过 8 个（部分手机的上限）。

export type BindKind = 'uniform' | 'read' | 'rw';

export interface BindSpec {
  name: string;
  kind: BindKind;
  type: string;
  /** 每子步不同的 uniform：用动态偏移切换 */
  dynamic?: boolean;
  /** 动态 uniform 每个槽位的字节数 */
  size?: number;
}

export interface Kernel {
  label: string;
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
  specs: BindSpec[];
}

export const WG = 64;

function declare(specs: BindSpec[]): string {
  return specs
    .map((s, i) => {
      const space = s.kind === 'uniform' ? 'uniform' : s.kind === 'read' ? 'storage, read' : 'storage, read_write';
      return `@group(0) @binding(${i}) var<${space}> ${s.name}: ${s.type};`;
    })
    .join('\n');
}

/**
 * 组装并编译一个内核。
 * 源码顺序：types（结构体和常量）→ 绑定声明 → fns（共用函数）→ body（内核本体，入口名为 main）。
 */
export function makeKernel(
  device: GPUDevice,
  label: string,
  parts: { types: string; fns: string; body: string },
  specs: BindSpec[],
  constants: Record<string, number> = {},
): Kernel {
  const storageCount = specs.filter((s) => s.kind !== 'uniform').length;
  if (storageCount > 8) throw new Error(`${label}: ${storageCount} 个存储缓冲，超过 8 个上限`);
  const code = [parts.types, declare(specs), parts.fns, parts.body].join('\n');
  const module = device.createShaderModule({ label, code });
  module.getCompilationInfo().then((info) => {
    for (const m of info.messages) {
      const line = code.split('\n')[m.lineNum - 1] ?? '';
      const msg = `[${label}] ${m.type} ${m.lineNum}:${m.linePos} ${m.message}\n  > ${line.trim()}`;
      if (m.type === 'error') console.error(msg);
      else console.warn(msg);
    }
  });
  const layout = device.createBindGroupLayout({
    label,
    entries: specs.map((s, i) => ({
      binding: i,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: s.kind === 'uniform' ? 'uniform' : s.kind === 'read' ? 'read-only-storage' : 'storage',
        hasDynamicOffset: !!s.dynamic,
      },
    })) as GPUBindGroupLayoutEntry[],
  });
  const pipeline = device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: 'main', constants },
  });
  return { label, pipeline, layout, specs };
}

/** 诊断用：逐个异步编译，返回每个内核的编译耗时（ms） */
export async function timeKernelCompile(
  device: GPUDevice,
  label: string,
  parts: { types: string; fns: string; body: string },
  specs: BindSpec[],
): Promise<number> {
  const code = [parts.types, declare(specs), parts.fns, parts.body].join('\n');
  const t0 = performance.now();
  const module = device.createShaderModule({ label, code });
  const layout = device.createBindGroupLayout({
    entries: specs.map((s, i) => ({
      binding: i,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: s.kind === 'uniform' ? 'uniform' : s.kind === 'read' ? 'read-only-storage' : 'storage' },
    })) as GPUBindGroupLayoutEntry[],
  });
  await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: 'main' },
  });
  return performance.now() - t0;
}

export function bindKernel(device: GPUDevice, k: Kernel, buffers: Record<string, GPUBuffer>): GPUBindGroup {
  return device.createBindGroup({
    label: k.label,
    layout: k.layout,
    entries: k.specs.map((s, i) => {
      const buffer = buffers[s.name];
      if (!buffer) throw new Error(`${k.label}: 缺少缓冲 ${s.name}`);
      return { binding: i, resource: s.dynamic ? { buffer, size: s.size } : { buffer } };
    }),
  });
}

export function groups(n: number, size = WG): number {
  return Math.max(1, Math.ceil(n / size));
}
