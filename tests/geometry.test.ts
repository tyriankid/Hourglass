import { describe, expect, it } from 'vitest';
import {
  BULB_H,
  NECK_R,
  bulbVolume,
  innerRadius,
  outerRadius,
  profileTable,
  volumeBelow,
  widestY,
  PROFILE_N,
} from '../src/geometry/profile';
import { G0, GRAIN_MASS, RHO_BULK, W0, beverlooRate, clumpMass, sandMassForSeconds } from '../src/physics';

describe('Beverloo 流量', () => {
  it('1 mm 孔径、0.1 mm 细沙约 59 mg/s', () => {
    expect(W0).toBeGreaterThan(58);
    expect(W0).toBeLessThan(61);
  });
  it('流量与 √g 成正比，重力为 0 或反向时不流', () => {
    expect(beverlooRate(G0 / 4)).toBeCloseTo(W0 / 2, 6);
    expect(beverlooRate(0)).toBe(0);
    expect(beverlooRate(-1)).toBe(0);
  });
  it('60 分钟约 213 g，1 分钟约 3.6 g', () => {
    expect(sandMassForSeconds(3600) / 1000).toBeGreaterThan(205);
    expect(sandMassForSeconds(3600) / 1000).toBeLessThan(220);
    expect(sandMassForSeconds(60) / 1000).toBeCloseTo(3.56, 1);
  });
  it('单粒 0.1 mm 沙约 1.39 µg，每秒约 4.3 万粒', () => {
    expect(GRAIN_MASS * 1000).toBeCloseTo(1.39, 2);
    expect(W0 / GRAIN_MASS).toBeGreaterThan(40000);
    expect(W0 / GRAIN_MASS).toBeLessThan(45000);
  });
  it('0.75 mm 颗粒团约 4.4 mg', () => {
    expect(clumpMass(0.00075)).toBeCloseTo(4.42, 1);
  });
});

describe('玻璃轮廓', () => {
  it('瓶颈处半径就是孔半径，轮廓连续且非负', () => {
    expect(innerRadius(0)[0]).toBeCloseTo(NECK_R, 9);
    for (let i = 0; i <= 1000; i++) {
      const ay = (i / 1000) * BULB_H;
      expect(innerRadius(ay)[0]).toBeGreaterThan(0);
      expect(outerRadius(ay)[0]).toBeGreaterThan(innerRadius(ay)[0]);
    }
  });

  it('从瓶颈到最宽处，壁面与水平面夹角 ≥ 50°（远陡于约 33° 的休止角），沙能流光', () => {
    const top = widestY();
    const maxSlope = Math.tan((40 * Math.PI) / 180); // dr/dy ≤ tan(90° − 50°)
    for (let i = 0; i <= 1000; i++) {
      const ay = (i / 1000) * top;
      expect(innerRadius(ay)[1]).toBeLessThanOrEqual(maxSlope);
    }
  });

  it('单个泡的容积 ≥ 60 分钟沙量体积的 1.35 倍', () => {
    const sandVolume = sandMassForSeconds(3600) / RHO_BULK;
    expect(bulbVolume() / sandVolume).toBeGreaterThanOrEqual(1.35);
  });

  it('1 分钟的沙在上泡里能堆到瓶颈上方至少 1.2 cm，看得清楚', () => {
    const v = sandMassForSeconds(60) / RHO_BULK;
    let h = 0;
    while (volumeBelow(h) < v) h += 0.0002;
    expect(h).toBeGreaterThan(0.012);
  });

  it('GPU 采样表与解析函数一致', () => {
    const t = profileTable();
    expect(t.length).toBe((PROFILE_N + 2) * 4);
    const i = 37;
    const ay = (i / PROFILE_N) * BULB_H;
    expect(t[i * 4]).toBeCloseTo(innerRadius(ay)[0], 7);
    expect(t[i * 4 + 2]).toBeCloseTo(outerRadius(ay)[0], 7);
    expect(t[(PROFILE_N + 1) * 4]).toBeCloseTo(BULB_H, 7);
  });
});
