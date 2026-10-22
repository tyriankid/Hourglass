#!/bin/bash
# 构建并把产物推到 gh-pages 分支（GitHub Pages），手机和电脑打开固定网址测试
set -e
cd "$(dirname "$0")/.."
npx tsc --noEmit
npx vite build --logLevel error --outDir .scratch/pages --emptyOutDir
touch .scratch/pages/.nojekyll
rev=$(git rev-parse --short HEAD)
cd .scratch/pages
rm -rf .git
git init -q -b gh-pages
git add -A
git -c user.name=tyriankid -c user.email=tyriankid@163.com commit -q -m "deploy $rev"
git push -q -f git@github-hourglass:tyriankid/Hourglass.git gh-pages
echo "已部署 $rev"
