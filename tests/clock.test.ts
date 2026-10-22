import { describe, expect, it } from 'vitest';
import { SandClock, formatClock } from '../src/clock/SandClock';
import { W0, sandMassForSeconds } from '../src/physics';

describe('沙子时钟', () => {
  it('时间只由流过的沙量决定', () => {
    const c = new SandClock(sandMassForSeconds(1500));
    expect(c.remainingSeconds).toBeCloseTo(1500, 9);
    c.add(W0 * 600);
    expect(c.elapsedSeconds).toBeCloseTo(600, 9);
    expect(c.remainingSeconds).toBeCloseTo(900, 9);
    expect(c.done).toBe(false);
  });
  it('逐帧累加 90 000 次（25 分钟 @60fps）没有可见漂移', () => {
    const c = new SandClock(sandMassForSeconds(1500));
    const perFrame = Math.fround(W0 / 60);
    for (let i = 0; i < 90000; i++) c.add(perFrame);
    expect(Math.abs(c.elapsedSeconds - 1500)).toBeLessThan(0.01);
    expect(c.done).toBe(true);
  });
  it('倒计时显示向上取整', () => {
    expect(formatClock(1500)).toBe('25:00');
    expect(formatClock(59.2)).toBe('01:00');
    expect(formatClock(0.4)).toBe('00:01');
    expect(formatClock(0)).toBe('00:00');
    expect(formatClock(120.00006)).toBe('02:00');
  });
});
