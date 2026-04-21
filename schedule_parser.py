#!/usr/bin/env python3
"""
Schedule Parser for Vitéz Steven
---------------------------------
Reads a schedule screenshot where:
  - The first visible day column header is highlighted BLUE
  - Steven's name cell is highlighted BLUE
Derives the full grid from those two anchors and outputs a Google Calendar CSV.

Usage:
    python3 schedule_parser.py <screenshot.png> [--year 2026] [--month 5] [--output out.csv]

Requirements:
    pip install Pillow numpy
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Blue anchor color range (the highlighted cells you set manually)
# Adjust tolerance if your blue is slightly different
BLUE_HUE_MIN = (0, 80, 180)    # min R, G, B
BLUE_HUE_MAX = (120, 180, 255)  # max R, G, B
BLUE_MIN_PIXELS = 30            # minimum blue pixels to count as a blue cell

# Symbol classification thresholds
YELLOW_BORDER_THRESHOLD = 180   # R and G must exceed this, B must be below
YELLOW_B_MAX = 120
LIGHT_BLUE_MIN = 150            # for normal "10" cells
GRAY_DIFF_MAX = 25              # R/G/B similarity threshold for gray/dash cells

# Days of week in German schedule headers
WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
WEEKEND = {"Sa", "So"}

# ---------------------------------------------------------------------------
# Blue cell detection
# ---------------------------------------------------------------------------

def is_blue_pixel(r, g, b):
    return (BLUE_HUE_MIN[0] <= r <= BLUE_HUE_MAX[0] and
            BLUE_HUE_MIN[1] <= g <= BLUE_HUE_MAX[1] and
            BLUE_HUE_MIN[2] <= b <= BLUE_HUE_MAX[2])


def find_blue_regions(arr):
    """Return list of (x1, y1, x2, y2) bounding boxes of blue regions."""
    mask = np.zeros(arr.shape[:2], dtype=bool)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            r, g, b = arr[y, x]
            if is_blue_pixel(r, g, b):
                mask[y, x] = True

    # Find connected components via simple flood-fill grouping
    visited = np.zeros_like(mask)
    regions = []

    def bbox(ys, xs):
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] and not visited[y, x]:
                # BFS
                queue = [(y, x)]
                ys, xs = [], []
                while queue:
                    cy, cx = queue.pop()
                    if cy < 0 or cy >= mask.shape[0] or cx < 0 or cx >= mask.shape[1]:
                        continue
                    if visited[cy, cx] or not mask[cy, cx]:
                        continue
                    visited[cy, cx] = True
                    ys.append(cy)
                    xs.append(cx)
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        queue.append((cy+dy, cx+dx))
                if len(ys) >= BLUE_MIN_PIXELS:
                    regions.append(bbox(np.array(ys), np.array(xs)))

    return regions


def find_anchors(arr):
    """
    Find the two blue anchor cells:
      - Header cell: topmost blue region  → defines column 1 x-position and cell width
      - Name cell:   leftmost blue region → defines Steven's row y-position and cell height
    Returns (header_box, name_box) as (x1,y1,x2,y2) tuples.
    """
    regions = find_blue_regions(arr)
    if len(regions) < 2:
        raise ValueError(f"Expected 2 blue anchor regions, found {len(regions)}. "
                         "Make sure both the first day column header and Steven's name cell are highlighted blue.")

    # Sort by y (top to bottom) → header cell comes first
    regions_sorted_y = sorted(regions, key=lambda r: r[1])
    header_box = regions_sorted_y[0]  # topmost = day header

    # Sort by x (left to right) → name cell comes first
    regions_sorted_x = sorted(regions, key=lambda r: r[0])
    name_box = regions_sorted_x[0]    # leftmost = name cell

    if header_box == name_box:
        raise ValueError("Only one unique blue region found. Need two separate blue cells.")

    return header_box, name_box

# ---------------------------------------------------------------------------
# Grid derivation
# ---------------------------------------------------------------------------

def derive_grid(header_box, name_box):
    """
    From the two anchor boxes derive:
      - cell_w, cell_h
      - col1_x: x start of first data column
      - steven_y: y start of Steven's row
    """
    hx1, hy1, hx2, hy2 = header_box
    nx1, ny1, nx2, ny2 = name_box

    cell_w = hx2 - hx1
    cell_h = hy2 - hy1

    col1_x = hx1   # first data column starts at header cell x
    steven_y = ny1  # Steven's row starts at name cell y

    return cell_w, cell_h, col1_x, steven_y


def scan_columns(arr, col1_x, steven_y, cell_w, cell_h, img_width):
    """
    Scan rightward from col1_x, extracting cell arrays for Steven's row.
    Stop when we run out of image or hit a clearly empty region.
    Returns list of numpy arrays (one per day column).
    """
    cells = []
    x = col1_x
    while x + cell_w <= img_width:
        cell = arr[steven_y:steven_y + cell_h, x:x + cell_w]
        cells.append((x, cell))
        x += cell_w
    return cells

# ---------------------------------------------------------------------------
# Cell classification
# ---------------------------------------------------------------------------

def has_yellow_border(cell):
    """Check if cell has a yellow border (Ferien / Weiterbildung)."""
    border = np.concatenate([
        cell[:2, :].reshape(-1, 3),
        cell[-2:, :].reshape(-1, 3),
        cell[:, :2].reshape(-1, 3),
        cell[:, -2:].reshape(-1, 3),
    ])
    avg = border.mean(axis=0)
    return (avg[0] > YELLOW_BORDER_THRESHOLD and
            avg[1] > YELLOW_BORDER_THRESHOLD - 20 and
            avg[2] < YELLOW_B_MAX)


def has_dienst_lower(cell):
    """
    Check for Dienst symbol (P+letter) in lower half of cell.
    Dienst lower half has a green/teal tinge distinct from the blue 10 block.
    """
    h = cell.shape[0]
    lower = cell[h//2:, 3:-3]
    if lower.size == 0:
        return False
    avg = lower.reshape(-1, 3).mean(axis=0)
    # Dienst lower half: greenish-teal, G and B higher than R, not the cyan of normal 10
    return (avg[1] > avg[0] + 15 and avg[2] > avg[0] + 10 and avg[0] < 180)


def has_letter_in_upper(cell, letter):
    """
    Heuristic: check if upper half of cell has yellow border (for combined cells).
    """
    h = cell.shape[0]
    upper = cell[:h//2, :]
    border = np.concatenate([
        upper[:2, :].reshape(-1, 3),
        upper[:, :2].reshape(-1, 3),
        upper[:, -2:].reshape(-1, 3),
    ])
    if border.size == 0:
        return False
    avg = border.reshape(-1, 3).mean(axis=0)
    return (avg[0] > YELLOW_BORDER_THRESHOLD and
            avg[1] > YELLOW_BORDER_THRESHOLD - 20 and
            avg[2] < YELLOW_B_MAX)


def is_normal_workday(cell):
    """Light blue 10 block."""
    center = cell[3:-3, 3:-3]
    if center.size == 0:
        return False
    avg = center.reshape(-1, 3).mean(axis=0)
    return (avg[2] > avg[0] + 20 and avg[1] > avg[0] + 10 and avg[0] > 130)


def is_dash_frei(cell):
    """Gray or lightly tinted dash cell."""
    center = cell[3:-3, 3:-3]
    if center.size == 0:
        return False
    avg = center.reshape(-1, 3).mean(axis=0)
    diff_rg = abs(int(avg[0]) - int(avg[1]))
    diff_gb = abs(int(avg[1]) - int(avg[2]))
    return avg[0] > 140 and diff_rg < GRAY_DIFF_MAX and diff_gb < GRAY_DIFF_MAX


def classify_cell(cell):
    """
    Returns one of:
      'Ferien', 'Weiterbildung', 'Dienst', 'Ferien+Dienst',
      'Weiterbildung+Dienst', 'Frei', 'Normal', '?'
    """
    yellow = has_yellow_border(cell)
    dienst_lower = has_dienst_lower(cell)
    normal = is_normal_workday(cell)
    frei = is_dash_frei(cell)

    # Combined: yellow upper + dienst lower
    if yellow and dienst_lower:
        # Distinguish F vs W by checking center of upper half for dark blue (F has dark bg)
        h = cell.shape[0]
        upper_center = cell[2:h//2-2, 4:-4]
        if upper_center.size > 0:
            uc_avg = upper_center.reshape(-1, 3).mean(axis=0)
            if uc_avg[2] > 100 and uc_avg[0] < 100:
                return "Ferien+Dienst"
        return "Weiterbildung+Dienst"

    if yellow:
        # Distinguish Ferien (F, dark blue bg) from Weiterbildung (W, dark blue bg)
        # Both have dark blue center — differentiate by checking for 'W' shape pixels
        # As a heuristic: Weiterbildung cells often have Wo2 text which adds lighter pixels
        # For now flag as uncertain between the two — use '?' and let user confirm
        # Actually from training: both F and W have identical color profile
        # We'll use the center darkness: F tends to be slightly darker blue
        center = cell[4:-4, 4:-4]
        if center.size > 0:
            avg = center.reshape(-1, 3).mean(axis=0)
            if avg[2] > avg[0] + 30 and avg[0] < 80:
                return "Ferien"        # very dark blue center = F
            elif avg[2] > avg[0] + 15:
                return "Weiterbildung" # slightly lighter = W
        return "Ferien?"

    if normal and dienst_lower:
        return "Dienst"   # combined 10/Ps cell

    if dienst_lower:
        return "Dienst"

    if normal:
        return "Normal"

    if frei:
        return "Frei"

    return "?"

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

import calendar as cal_module

def day_of_week(year, month, day):
    """Return German day abbreviation."""
    weekday = cal_module.weekday(year, month, day)
    return WEEKDAYS[weekday]


def days_in_month(year, month):
    return cal_module.monthrange(year, month)[1]

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def to_csv(assignments, output_path):
    """Write Google Calendar CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "Start Date", "End Date", "All Day Event", "Description"])
        for date_str, assignment in assignments:
            for entry in assignment.split("+"):
                entry = entry.strip()
                writer.writerow([entry, date_str, date_str, "TRUE", ""])
    print(f"Saved CSV: {output_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse schedule screenshot for Vitéz Steven.")
    parser.add_argument("image", help="Path to screenshot PNG/JPG")
    parser.add_argument("--year",  type=int, required=True, help="Year of the schedule (e.g. 2026)")
    parser.add_argument("--month", type=int, required=True, help="Month of the first visible day (1-12)")
    parser.add_argument("--day",   type=int, default=1,     help="Day number of the first visible column (default: 1)")
    parser.add_argument("--output", default="schedule.csv", help="Output CSV filename")
    parser.add_argument("--debug", action="store_true", help="Save debug contact sheet")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    arr = np.array(img)

    print(f"Image size: {img.size}")
    print("Searching for blue anchor cells...")

    try:
        header_box, name_box = find_anchors(arr)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Header cell (day anchor): x={header_box[0]}-{header_box[2]}, y={header_box[1]}-{header_box[3]}")
    print(f"  Name cell  (row anchor):  x={name_box[0]}-{name_box[2]}, y={name_box[1]}-{name_box[3]}")

    cell_w, cell_h, col1_x, steven_y = derive_grid(header_box, name_box)
    print(f"  Cell size: {cell_w}w x {cell_h}h px")
    print(f"  Grid origin: col1_x={col1_x}, steven_y={steven_y}")

    cells = scan_columns(arr, col1_x, steven_y, cell_w, cell_h, img.width)
    print(f"  Found {len(cells)} day columns")

    # Build date sequence starting from args.year, args.month, args.day
    year, month, day = args.year, args.month, args.day
    assignments = []
    uncertain = []
    skipped = 0

    print("\nClassification:")
    for i, (x_pos, cell_arr) in enumerate(cells):
        dow = day_of_week(year, month, day)
        is_weekend = dow in WEEKEND
        date_str = f"{month:02d}/{day:02d}/{year}"
        label = classify_cell(cell_arr)

        # Advance date
        current_day = day
        current_month = month
        current_year = year
        day += 1
        if day > days_in_month(year, month):
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1

        # Apply rules
        if label == "Normal":
            skipped += 1
            continue
        if label == "Frei" and is_weekend:
            skipped += 1
            continue
        if label == "Frei" and not is_weekend:
            assignments.append((date_str, "Frei"))
            print(f"  {date_str} {dow}: Frei")
            continue
        if "?" in label:
            uncertain.append((date_str, dow, label))
            print(f"  {date_str} {dow}: {label} ← UNCERTAIN")
            continue

        assignments.append((date_str, label))
        print(f"  {date_str} {dow}: {label}")

    print(f"\nTotal: {len(assignments)} entries, {skipped} skipped (normal/weekend), {len(uncertain)} uncertain")

    if uncertain:
        print("\nUncertain cells requiring manual confirmation:")
        for date_str, dow, label in uncertain:
            print(f"  {date_str} {dow}: {label}")

    if args.debug:
        # Save contact sheet of all cells
        from PIL import ImageDraw
        scale = 4
        pad = 3
        sh = cell_h * scale + 20
        sw = len(cells) * (cell_w * scale + pad) + pad
        sheet = Image.new("RGB", (sw, sh), (220, 220, 220))
        draw = ImageDraw.Draw(sheet)
        for i, (_, cell_arr) in enumerate(cells):
            cell_img = Image.fromarray(cell_arr)
            scaled = cell_img.resize((cell_w * scale, cell_h * scale), Image.NEAREST)
            xp = pad + i * (cell_w * scale + pad)
            sheet.paste(scaled, (xp, 0))
            draw.text((xp + 2, cell_h * scale + 4), str(i+1), fill=(0,0,0))
        sheet.save("debug_cells.png")
        print("Saved debug_cells.png")

    to_csv(assignments, args.output)
    print("\nDone. Import the CSV into Google Calendar via:")
    print("  calendar.google.com → Settings → Import & Export → Import")


if __name__ == "__main__":
    main()
