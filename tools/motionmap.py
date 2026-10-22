"""把连续帧里每个像素"变化超过阈值"的次数叠加成热力图，叠在最后一帧上，看画面里哪里在动。
用法：python3 tools/motionmap.py '.scratch/shots/fr-run-*.png' out.png"""
import glob, sys
import numpy as np
from PIL import Image

fs = sorted(glob.glob(sys.argv[1]))
frames = [np.asarray(Image.open(f).convert('L'), dtype=np.float32) for f in fs]
acc = np.zeros_like(frames[0])
for a, b in zip(frames[:-1], frames[1:]):
    acc += (np.abs(a - b) > 6)
h, w = acc.shape
for name, (y0, y1) in {'上泡': (0, int(h * 0.48)), '瓶颈+沙流': (int(h * 0.48), int(h * 0.62)), '下泡': (int(h * 0.62), int(h * 0.8)), '桌面以下': (int(h * 0.8), h)}.items():
    reg = acc[y0:y1, int(w * 0.3):int(w * 0.7)]
    print(f'{name}: 有变化的像素 {int((reg > 0).sum())}，频繁变化（>1/3 帧）{int((reg > len(fs) / 3).sum())}')
img = np.asarray(Image.open(fs[-1]).convert('RGB')).astype(np.float32)
heat = np.clip(acc / max(1, len(fs) - 1), 0, 1)[..., None]
img = img * (1 - heat * 0.85) + np.array([255, 30, 30]) * heat * 0.85
Image.fromarray(img.astype(np.uint8)).save(sys.argv[2])
