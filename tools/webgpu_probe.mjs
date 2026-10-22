// 探测无头 Chromium 能否拿到 WebGPU 适配器
import { chromium } from 'playwright';
const flagsets = [
  ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader'],
  ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader'],
  ['--enable-unsafe-webgpu'],
];
for (const args of flagsets) {
  const b = await chromium.launch({ args, headless: true, channel: process.env.CH || undefined });
  const p = await b.newPage();
  await p.goto('file:///tmp/probe.html');
  const r = await p.evaluate(async () => {
    if (!navigator.gpu) return 'no navigator.gpu';
    const a = await navigator.gpu.requestAdapter();
    if (!a) return 'no adapter';
    const info = a.info || {};
    return JSON.stringify({ vendor: info.vendor, arch: info.architecture, desc: info.description, maxBuf: a.limits.maxStorageBufferBindingSize, maxInv: a.limits.maxComputeInvocationsPerWorkgroup, feats: [...a.features] });
  }).catch(e => 'err ' + e.message);
  console.log(args.join(' '), '=>', r);
  await b.close();
}
