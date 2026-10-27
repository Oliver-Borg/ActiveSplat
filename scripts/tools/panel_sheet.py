#!/usr/bin/env python
# Contact sheet from the in-run image evaluation panels (image_eval/<view>.jpg = simulator | map | difference) of several runs: one row per run,
# one column per view, each cell the simulator view above the map's render at the same pose.
#   python scripts/tools/panel_sheet.py --runs "label=results/<run>" "label=results/<run>" --views trajectory:1 trajectory:3 trajectory:5 novel:2 novel:4 --out sheet.jpg [--width 384]
# Runs that took different paths show different places in the same column (native-planner runs diverge).
import argparse
import os

from PIL import Image, ImageDraw

def panel_path(run_dir:str, view:str) -> str:
    '''view is a panel name (trajectory_0200) or <prefix>:<n>, the n-th panel (from 0) of that prefix, as panel names differ between runs.'''
    folder = os.path.join(run_dir, 'image_eval')
    if ':' in view:
        prefix, index = view.split(':')
        return os.path.join(folder, sorted(name for name in os.listdir(folder) if name.startswith(prefix + '_'))[int(index)])
    return os.path.join(folder, view + '.jpg')

def cell(run_dir:str, view:str, width:int) -> Image.Image:
    panel = Image.open(panel_path(run_dir, view))
    third = panel.width // 3
    parts = [panel.crop((i * third, 0, (i + 1) * third, panel.height)).resize((width, round(panel.height * width / third))) for i in range(2)]
    out = Image.new('RGB', (width, parts[0].height * 2), 'white')
    out.paste(parts[0], (0, 0))
    out.paste(parts[1], (0, parts[0].height))
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='label=run_dir')
    parser.add_argument('--views', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--width', type=int, default=384)
    args = parser.parse_args()
    rows = []
    for item in args.runs:
        label, run_dir = item.split('=', 1)
        cells = [cell(run_dir, view, args.width) for view in args.views]
        rows.append((label, cells))
    gap, header = 6, 22
    cell_h = rows[0][1][0].height
    sheet = Image.new('RGB', (len(args.views) * (args.width + gap), len(rows) * (cell_h + header + gap)), 'white')
    draw = ImageDraw.Draw(sheet)
    for r, (label, cells) in enumerate(rows):
        y = r * (cell_h + header + gap)
        draw.text((4, y + 4), f'{label}   (top: simulator, bottom: map)', fill='black')
        for c, im in enumerate(cells):
            sheet.paste(im, (c * (args.width + gap), y + header))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out, quality=88)

if __name__ == '__main__':
    main()
