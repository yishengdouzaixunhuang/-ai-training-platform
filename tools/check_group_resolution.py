# -*- coding: utf-8 -*-
"""检查一个图片目录里"组内分辨率是否一致"（三视图必须像素对齐）。

用法：
  python tools/check_group_resolution.py <目录> [--out 报告.md]
约定：文件名形如 `<组名>_te_0/1/2.<ext>`；组内分辨率不一致 = 三视图无法像素对齐。
"""
from __future__ import annotations
import os, re, sys, time, collections
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

IMG = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
PAT = re.compile(r'^(.*)_te_(\d+)$')


def scan(root, workers=24, log=print):
    files = [f for f in os.listdir(root) if f.lower().endswith(IMG)]
    log('  图片 %d 张，读文件头…' % len(files))
    t0 = time.time()
    def sz(f):
        try:
            with Image.open(os.path.join(root, f)) as im:
                return f, im.size
        except Exception:
            return f, (0, 0)
    sizes = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f, s in ex.map(sz, files):
            sizes[f] = s
    log('  读完 %.0fs' % (time.time() - t0))
    groups = collections.defaultdict(dict)
    unnamed = []
    for f, s in sizes.items():
        m = PAT.match(os.path.splitext(f)[0])
        if m:
            groups[m.group(1)][int(m.group(2))] = s
        else:
            unnamed.append(f)
    bad = {k: v for k, v in groups.items() if len({tuple(x) for x in v.values()}) > 1}
    short = {k: v for k, v in groups.items() if len(v) < 3}
    return dict(files=files, groups=groups, bad=bad, short=short, unnamed=unnamed,
                dist=collections.Counter(tuple(s) for s in sizes.values()))


def main():
    root = sys.argv[1]
    out = None
    if '--out' in sys.argv:
        out = sys.argv[sys.argv.index('--out') + 1]
    r = scan(root)
    print('  识别 %d 组；**组内分辨率不一致 %d 组**；视图不足 3 张 %d 组；未按命名规则 %d 个文件'
          % (len(r['groups']), len(r['bad']), len(r['short']), len(r['unnamed'])))
    if r['bad']:
        print('\n  === 不一致的组 ===')
        for k, v in sorted(r['bad'].items()):
            print('    %-34s %s' % (k[:34], '  '.join(
                ('te%d=%dx%d' % (i, v[i][0], v[i][1])) if i in v else ('te%d=缺' % i) for i in (0, 1, 2))))
    if r['short']:
        print('\n  === 视图不足 3 张 ===')
        for k, v in sorted(r['short'].items())[:10]:
            print('    %-34s 只有 %s' % (k[:34], sorted(v)))
    print('\n  分辨率分布前 8: ' + '  '.join('%dx%d×%d' % (a, b, c) for (a, b), c in r['dist'].most_common(8)))
    if out:
        L = ['# 组内分辨率检查报告\n\n目录：`%s`\n\n' % root,
             '- 图片 %d 张，识别 %d 组\n- **组内分辨率不一致 %d 组**\n- 视图不足 3 张 %d 组\n\n' % (len(r['files']), len(r['groups']), len(r['bad']), len(r['short']))]
        if r['bad']:
            L.append('## 不一致的组\n\n| 组名 | te_0 | te_1 | te_2 |\n|---|---|---|---|\n')
            for k, v in sorted(r['bad'].items()):
                L.append('| %s | %s | %s | %s |\n' % (k, *[('%dx%d' % v[i]) if i in v else '缺' for i in (0, 1, 2)]))
        if r['short']:
            L.append('\n## 视图不足 3 张\n\n')
            for k, v in sorted(r['short'].items()):
                L.append('- %s：%s\n' % (k, sorted(v)))
        open(out, 'w', encoding='utf-8').write(''.join(L))
        print('  报告:', out)


if __name__ == '__main__':
    main()
