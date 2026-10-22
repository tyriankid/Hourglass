// 物理常数与沙漏的真实尺度。单位：长度 m，时间 s，质量 mg（毫克）。
// 质量用 mg 是为了让 GPU 上的 f32 数值落在舒服的范围里（颗粒团几 mg，沙粒约 0.0014 mg）。

export const G0 = 9.81;

/** 沙的堆积密度 1550 kg/m³，换成 mg/m³ */
export const RHO_BULK = 1550 * 1e6;
/** 石英砂颗粒本身的密度 2650 kg/m³，换成 mg/m³ */
export const RHO_GRAIN = 2650 * 1e6;

/** 沙粒直径 0.1 mm（细沙） */
export const GRAIN_D = 1.0e-4;
/** 瓶颈孔径 1.0 mm */
export const ORIFICE_D = 1.0e-3;

/** Beverloo 定律的经验常数 */
export const BEVERLOO_C = 0.58;
export const BEVERLOO_K = 1.5;

/** 颗粒团的随机堆积率：一个半径 r 的颗粒团代表 (4/3πr³)/PACKING 的沙 */
export const PACKING = 0.62;

/** 单粒沙的质量（mg） */
export const GRAIN_MASS = RHO_GRAIN * (Math.PI / 6) * GRAIN_D ** 3;

/**
 * Beverloo 定律：颗粒物从孔口流出的质量流量（mg/s）。
 * gAxial 是有效重力沿孔口轴向的分量；≤ 0 时不流。
 */
export function beverlooRate(gAxial: number, D = ORIFICE_D, d = GRAIN_D): number {
  if (gAxial <= 0) return 0;
  const eff = D - BEVERLOO_K * d;
  if (eff <= 0) return 0;
  return BEVERLOO_C * RHO_BULK * Math.sqrt(gAxial) * eff ** 2.5;
}

/** 立着时的标称流量 W0（mg/s），也是沙子时钟的换算常数 */
export const W0 = beverlooRate(G0);

/** 设定 T 秒需要的沙量（mg） */
export function sandMassForSeconds(seconds: number): number {
  return W0 * seconds;
}

/** 半径 r 的颗粒团代表的沙量（mg） */
export function clumpMass(r: number): number {
  return (RHO_BULK * (4 / 3) * Math.PI * r ** 3) / PACKING;
}
