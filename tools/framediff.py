"""相邻帧的像素差异：在画面中部（沙漏所在区域）统计平均绝对差和变化像素比例，量化肉眼可见的抖动。
用法：python3 tools/framediff.py '.scratch/shots/fr-rest-*.png'"""
import glob, sys
import numpy as np
from PIL import Image

files = sorted(glob.glob(sys.argv[1]))
prev = None
rows = []
for f in files:
    im = np.asarray(Image.open(f).convert('L'), dtype=np.float32)
    h, w = im.shape
    roi = im[int(h * 0.12):int(h * 0.78), int(w * 0.3):int(w * 0.7)]
    if prev is not None:
        d = np.abs(roi - prev)
        rows.append((d.mean(), (d > 6).mean() * 100, d.max()))
    prev = roi
if rows:
    a = np.array(rows)
    print(f"帧数 {len(files)}  平均绝对差 {a[:,0].mean():.3f}  变化像素(>6/255) {a[:,1].mean():.3f}%  最大差 {a[:,2].max():.0f}")
