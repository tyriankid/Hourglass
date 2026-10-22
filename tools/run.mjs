// 无头 Chromium（SwiftShader WebGPU）跑页面：截图或等 window.__result。
// 用法：node tools/run.mjs "<query>" [截图路径] [等待毫秒] [actions.json]
import { build, preview } from 'vite';
import { chromium } from 'playwright';
import os from 'node:os';

const [, , query = '', shot = '', waitMs = '6000', actions = ''] = process.argv;
const libs = `${os.homedir()}/.local/chromelibs/root/usr/lib/x86_64-linux-gnu`;
const T0 = Date.now();
const log = (m) => console.log(`[run +${((Date.now() - T0) / 1000).toFixed(1)}s] ${m}`);
// 先打包一份快照再用静态服务跑，测试过程中改代码不会让页面热更新
const outDir = `.scratch/dist-${process.pid}`;
await build({
  logLevel: 'error',
  build: { outDir, emptyOutDir: true, rollupOptions: process.env.PAGE ? { input: { main: 'index.html', extra: process.env.PAGE } } : undefined },
});
const server = await preview({ preview: { port: 0, host: '127.0.0.1' }, build: { outDir }, logLevel: 'error' });
const url = server.resolvedUrls?.local?.[0];
log('snapshot served ' + url);
const browser = await chromium.launch({
  headless: true,
  args: [
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
    // 没有这组参数，WebGPU 画布的交换链在无头 SwiftShader 下建不出来，GPU 进程直接丢设备
    '--use-angle=swiftshader',
    '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
    ...(process.env.GPUFLAGS ? process.env.GPUFLAGS.split(' ') : []),
    ...(process.env.GPULOG ? ['--enable-logging=stderr', '--v=0'] : []),
  ],
  env: { ...process.env, LD_LIBRARY_PATH: libs },
});
log('browser launched');
const page = await browser.newPage({ viewport: { width: 900, height: 1000 } });
page.on('console', (m) => console.log(`[page:${m.type()}]`, m.text()));
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(process.env.PAGE ? `${url}${process.env.PAGE}` : `${url}?${query}`);
log('page loaded ' + url);
if (/test=(1|restore)|probe=film/.test(query)) {
  // 场景测试：页面请求截图时截图，直到出结果
  const base = shot || '.scratch/shots/scenario';
  for (;;) {
    const st = await page.evaluate(() => ({ req: window.__shotRequest, res: window.__result }));
    if (st.res) {
      console.log('RESULT', JSON.stringify(st.res));
      break;
    }
    if (st.req) {
      await page.screenshot({ path: `${base}-${st.req}.png` });
      log('shot ' + st.req);
      await page.evaluate(() => {
        window.__shotRequest = null;
        window.__shotDone = true;
      });
    }
    await page.waitForTimeout(500);
  }
} else if (/frames=/.test(query)) {
  await page.waitForFunction(() => window.__result, null, { timeout: 0, polling: 1000 });
  console.log('RESULT', JSON.stringify(await page.evaluate(() => window.__result)));
  await page.waitForTimeout(500);
  if (shot) await page.screenshot({ path: shot });
} else if (/mode=(drain|compile|bench|inspect|stream|neck)|test=finish|probe=/.test(query) || process.env.PAGE) {
  await page.waitForFunction(() => window.__result, null, { timeout: 0, polling: 2000 });
  console.log('RESULT', JSON.stringify(await page.evaluate(() => window.__result), null, 2));
} else {
  if (actions) {
    for (const a of JSON.parse(actions)) {
      if (a.wait) await page.waitForTimeout(a.wait);
      if (a.key) await page.keyboard.press(a.key);
      if (a.shot) await page.screenshot({ path: a.shot });
    }
  } else {
    await page.waitForTimeout(Number(waitMs));
  }
  if (shot) await page.screenshot({ path: shot });
}
await browser.close();
await server.close();
const { rmSync } = await import('node:fs');
rmSync(outDir, { recursive: true, force: true });
