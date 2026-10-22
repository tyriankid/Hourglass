# 沙漏

一个 ASMR 放松向的专注计时器。画面中心是一个 3D 沙漏，里面的沙子用真实物理模拟。设定时间后会生成相应量的沙，沙全部流完的那一刻，恰好就是设定的时间。

- 设定 1–60 分钟：拖动底部刻度尺，沙在下方玻璃泡里由微光凝聚出来、落下堆积。
- 轻点沙漏：翻转并开始计时。计时中轻点会把沙漏横放，起到暂停作用，沙停时间就停；再点一次立起来继续。
- 从沙漏上开始拖动可以倾斜、晃动它，从背景开始拖动则是环绕观察；双指或滚轮缩放。设置里可以打开重力感应。
- 声音全部由物理驱动、程序合成，包括沙声、雨声、壁炉、白噪音、背景音乐和结束时的颂钵。

需要支持 WebGPU 的浏览器：最新版 Chrome / Edge，或 iOS 26 及以上的 Safari。

## 运行

```bash
npm install
npm run dev      # 本地开发（WebGPU 需要 localhost 或 HTTPS）
npm test         # 单元测试
npm run build
```

推送到 `main` 后，GitHub Actions 会自动发布到 GitHub Pages。网址后加 `?hud=1` 可以显示性能面板。

## 原理

完整设计见 [`docs/design.md`](docs/design.md)。

- **沙子就是时钟**：时间读数 = 已流过瓶颈的沙量 ÷ 标称流量。沙在流，时间就走；横放、倾斜导致断流，时间就停。
- **瓶颈流量**：用 Beverloo 定律计算，W = C·ρ·√g·(D − k·d)^2.5。真实沙漏能计时，靠的就是流量与上方沙层高度无关。孔径 1 mm、沙粒 0.1 mm 时，W ≈ 59 mg/s，60 分钟约 213 g 沙。
- **多尺度物理**（GPU compute，WGSL）：
  - 整体沙体：PBD 颗粒团，有真实接触和静、动摩擦，自然形成休止角、漏斗和崩塌。
  - 瓶颈：孔口边界条件，按 Beverloo 流量从捕获区抽走沙。
  - 沙流：0.1 mm 的真实沙粒，数量一比一（每秒约 4.3 万粒），扫掠碰撞后沉积长成颗粒团。
  - 所有计算都在沙漏自身坐标系里进行，翻转、倾斜、晃动对物理来说就是有效重力的旋转加上惯性力。
- **渲染**（原生 WebGPU）：
  - 沙面：球体 impostor 加双边平滑，颗粒质感锚定在颗粒团上，所以随沙一起流动。
  - 玻璃：对内外两层做解析折射和菲涅尔反射。
  - 桌面：用光子追踪算阴影和焦散。

## 测试工具

无头 Chromium（SwiftShader WebGPU）：

```bash
node tools/run.mjs "mode=drain&rc=1.25&min=1"        # 排空测试：计时误差、质量守恒
node tools/run.mjs "mode=inspect&min=10&rc=1.25"      # 堆积稳定性：高度分布、颗粒重叠
node tools/run.mjs "mode=stream&min=2&rc=1.25"        # 沙流统计
node tools/run.mjs "test=1&min=5&rc=1.0&scale=0.75" .scratch/shots/sc   # 完整流程逐阶段截图
```
