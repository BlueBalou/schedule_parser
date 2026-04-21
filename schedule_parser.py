#!/usr/bin/env python3
"""
Schedule Parser for Vitéz Steven
---------------------------------
Uses TWO MAGENTA (#ff00ff) anchor blocks and reference cell images to extract
schedule data from a screenshot and produce a Google Calendar CSV.

PREPARATION:
  1. Place reference cell images (24x24 PNG) in ./refimages/
  2. Open the schedule screenshot in GIMP
  3. Paint TWO 24x24 magenta (#ff00ff) filled rectangles:
     - One directly ABOVE the first data cell (same x-alignment)
     - One directly LEFT of the first data cell (same y-alignment)
     Both touch the top-left corner of the first data cell:

                   [mag_top 24x24]
     [mag_left 24x24] [first cell] [cell 2] [cell 3] ...

  4. Save as PNG (compression 0, no interlacing)

REFERENCE IMAGES (in ./refimages/):
  Blue_10_block.png                          → Normal (skip)
  Yellow_border_F.png                        → Ferien
  Yellow_border_Wo2.png                      → Weiterbildung
  Split_cell_10__top_Ps_bottom.png           → Dienst
  Split_cell_plain_blue_dash_top_Ps_bottom.png → Dienst
  Split_cell_W__top_Ps_bottom.png            → Weiterbildung+Dienst
  Plain_bue_dash_bright.png                  → Frei
  Plain_bue_dash_dark.png                    → Frei
  (Add more as needed — just follow the naming pattern)

Usage:
  python3 schedule_parser.py screenshot.png --year 2026 --month 4 --day 27
  python3 schedule_parser.py screenshot.png --year 2026 --month 4 --day 27 --debug
"""

import argparse
import calendar as cal_module
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_LINE = 1         # grid line width in pixels

# Magenta detection
MAGENTA_R_MIN = 200
MAGENTA_G_MAX = 50
MAGENTA_B_MIN = 200
MAGENTA_MIN_PIXELS = 100

# Reference image directory
REF_DIR = "./refimages"

# Mapping from reference filename to calendar category
# Add new reference images here — the script auto-loads them
REF_MAP = {
    "Blue_10_block.png":                          "Normal",
    "Yellow_border_F.png":                        "Ferien",
    "Yellow_border_Wo2.png":                      "Weiterbildung",
    "Split_cell_10__top_Ps_bottom.png":           "Dienst",
    "Split_cell_plain_blue_dash_top_Ps_bottom.png": "Dienst",
    "Split_cell_W__top_Ps_bottom.png":            "Weiterbildung+Dienst",
    "Plain_bue_dash_bright.png":                  "Frei",
    "Plain_bue_dash_dark.png":                    "Frei",
    "Yellow_border_W.png":                        "Weiterbildung",
}

WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
WEEKEND = {"Sa", "So"}

# ---------------------------------------------------------------------------
# Reference image loading
# ---------------------------------------------------------------------------

def cell_signature(arr):
    """
    Compute a 9-dimensional color signature for a cell:
    [full_R, full_G, full_B, upper_R, upper_G, upper_B, lower_R, lower_G, lower_B]
    """
    f = arr.astype(float)
    h = f.shape[0]
    full = f.reshape(-1, 3).mean(axis=0)
    upper = f[:h//2].reshape(-1, 3).mean(axis=0)
    lower = f[h//2:].reshape(-1, 3).mean(axis=0)
    return np.concatenate([full, upper, lower])


def load_references(ref_dir):
    """Load reference images and build signature database."""
    ref_sigs = []
    ref_labels = []
    ref_names = []

    for fname, label in REF_MAP.items():
        path = os.path.join(ref_dir, fname)
        if not os.path.exists(path):
            print(f"  WARNING: Reference image not found: {path}")
            continue
        arr = np.array(Image.open(path).convert("RGB"))
        sig = cell_signature(arr)
        ref_sigs.append(sig)
        ref_labels.append(label)
        ref_names.append(fname)

    if not ref_sigs:
        print(f"ERROR: No reference images found in {ref_dir}")
        sys.exit(1)

    return np.array(ref_sigs), ref_labels, ref_names


def classify_cell(cell_arr, ref_sigs, ref_labels, ref_names):
    """
    Classify a cell by finding the nearest reference image (Euclidean distance
    on the 9-dim color signature).
    Returns (label, distance, matched_reference_name).
    """
    sig = cell_signature(cell_arr)
    distances = np.linalg.norm(ref_sigs - sig, axis=1)
    best_idx = np.argmin(distances)
    return ref_labels[best_idx], distances[best_idx], ref_names[best_idx]

# ---------------------------------------------------------------------------
# Magenta anchor detection
# ---------------------------------------------------------------------------

def find_magenta_anchors(arr):
    """
    Find two magenta anchor blocks (each 24x24):
      - mag_top:  placed ABOVE the first data cell → defines column x-position
      - mag_left: placed LEFT of the first data cell → defines row y-position

    Layout:
                  [mag_top]
      [mag_left]  [first data cell] [cell 2] ...

    Both touch the top-left corner of the first data cell.
    Returns (mag_top_box, mag_left_box, pixel_count) where each box is (x1,y1,x2,y2),
    or (None, None, pixel_count) on failure.
    """
    mask = (
        (arr[:, :, 0] >= MAGENTA_R_MIN) &
        (arr[:, :, 1] <= MAGENTA_G_MAX) &
        (arr[:, :, 2] >= MAGENTA_B_MIN)
    )

    count = int(mask.sum())
    if count < MAGENTA_MIN_PIXELS:
        return None, None, count

    # Find connected components via flood fill
    visited = np.zeros_like(mask)
    regions = []
    ys_all, xs_all = np.where(mask)

    for start_y, start_x in zip(ys_all, xs_all):
        if visited[start_y, start_x]:
            continue
        queue = [(int(start_y), int(start_x))]
        component_ys = []
        component_xs = []
        head = 0
        while head < len(queue):
            cy, cx = queue[head]
            head += 1
            if cy < 0 or cy >= mask.shape[0] or cx < 0 or cx >= mask.shape[1]:
                continue
            if visited[cy, cx] or not mask[cy, cx]:
                continue
            visited[cy, cx] = True
            component_ys.append(cy)
            component_xs.append(cx)
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                queue.append((cy+dy, cx+dx))

        if len(component_ys) >= 50:  # minimum pixels for a real anchor
            regions.append((
                min(component_xs), min(component_ys),
                max(component_xs), max(component_ys)
            ))

    if len(regions) < 2:
        return None, None, count

    # Identify which is top and which is left:
    # mag_top has the smaller y1 (higher up)
    # mag_left has the smaller x1 (further left)
    # Since they share a corner, mag_top is higher and mag_left is more to the left
    regions_by_y = sorted(regions, key=lambda r: r[1])
    mag_top = regions_by_y[0]   # topmost = above first cell
    mag_left = regions_by_y[1]  # lower = left of first cell

    return mag_top, mag_left, count

# ---------------------------------------------------------------------------
# Grid derivation
# ---------------------------------------------------------------------------

def derive_grid(mag_top, mag_left):
    """
    From two magenta anchor blocks, derive the grid origin.

    Layout:
                  [mag_top]
      [mag_left]  [first data cell]

    - mag_top defines the x-position: first cell x = mag_top x1
      (mag_top is directly above the first cell, same x-alignment)
    - mag_left defines the y-position: first cell y = mag_left y1
      (mag_left is directly left of the first cell, same y-alignment)
    - Cell size is derived from mag_top width and mag_left height
    """
    tx1, ty1, tx2, ty2 = mag_top
    lx1, ly1, lx2, ly2 = mag_left

    cell_w = tx2 - tx1 + 1  # width from top anchor
    cell_h = ly2 - ly1 + 1  # height from left anchor

    first_cell_x = tx1       # same x as top anchor
    steven_y = ly1            # same y as left anchor

    return first_cell_x, steven_y, cell_w, cell_h


def extract_cells(arr, first_cell_x, steven_y, cell_w, cell_h, img_width):
    """Extract all data cells from Steven's row."""
    cells = []
    x = first_cell_x
    pitch = cell_w + GRID_LINE
    while x + cell_w <= img_width:
        cell = arr[steven_y:steven_y + cell_h, x:x + cell_w]
        if cell.shape[0] == cell_h and cell.shape[1] == cell_w:
            cells.append((x, cell))
        x += pitch
    return cells

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def day_of_week(year, month, day):
    return WEEKDAYS[cal_module.weekday(year, month, day)]

def days_in_month(year, month):
    return cal_module.monthrange(year, month)[1]

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def to_csv(assignments, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "Start Date", "End Date", "All Day Event", "Description"])
        for date_str, assignment in assignments:
            for entry in assignment.split("+"):
                writer.writerow([entry.strip(), date_str, date_str, "TRUE", ""])
    print(f"Saved CSV: {output_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse schedule screenshot for Vitéz Steven.",
        epilog="Place a 24x24 magenta (#ff00ff) anchor at the top-left corner "
               "of the first data cell before running."
    )
    parser.add_argument("image", help="Path to screenshot PNG")
    parser.add_argument("--year",  type=int, required=True)
    parser.add_argument("--month", type=int, required=True, help="Month of first visible day")
    parser.add_argument("--day",   type=int, default=1,     help="Day number of first visible column")
    parser.add_argument("--output", default="schedule.csv")
    parser.add_argument("--refdir", default=REF_DIR, help="Path to reference images directory")
    parser.add_argument("--debug", action="store_true", help="Save debug images and verbose output")
    parser.add_argument("--threshold", type=float, default=60.0,
                        help="Max distance to accept a match (default: 60). "
                             "Higher = more tolerant, lower = stricter.")
    args = parser.parse_args()

    # --- Load image ---
    img = Image.open(args.image).convert("RGB")
    arr = np.array(img)
    print(f"Image: {img.size[0]}x{img.size[1]}")

    # --- Load references ---
    print(f"Loading references from {args.refdir}/")
    ref_sigs, ref_labels, ref_names = load_references(args.refdir)
    print(f"  Loaded {len(ref_sigs)} reference(s)")

    # --- Find magenta anchors ---
    print("Searching for magenta anchors...")
    mag_top, mag_left, pixel_count = find_magenta_anchors(arr)

    if mag_top is None or mag_left is None:
        print(f"\nERROR: Need 2 magenta anchor blocks, found fewer ({pixel_count} magenta pixels).")
        print("Instructions:")
        print("  1. Open screenshot in GIMP")
        print("  2. Paint TWO 24x24px filled rectangles with #ff00ff (magenta)")
        print("  3. Place one ABOVE the first data cell (same x-alignment)")
        print("  4. Place one LEFT of the first data cell (same y-alignment)")
        print("  5. Both should touch the top-left corner of the first data cell")
        print("  6. Export as PNG (compression 0)")
        sys.exit(1)

    tx1, ty1, tx2, ty2 = mag_top
    lx1, ly1, lx2, ly2 = mag_left
    tw, th = tx2-tx1+1, ty2-ty1+1
    lw, lh = lx2-lx1+1, ly2-ly1+1
    print(f"  Top anchor:  ({tx1},{ty1})-({tx2},{ty2}), size {tw}x{th}")
    print(f"  Left anchor: ({lx1},{ly1})-({lx2},{ly2}), size {lw}x{lh}")

    # --- Derive grid (cell size auto-detected from anchors) ---
    first_cell_x, steven_y, cell_w, cell_h = derive_grid(mag_top, mag_left)
    cell_pitch = cell_w + GRID_LINE
    print(f"  Auto-detected cell size: {cell_w}x{cell_h}, pitch={cell_pitch}")
    print(f"  Grid: first_cell_x={first_cell_x}, steven_y={steven_y}")

    # --- Extract cells ---
    cells = extract_cells(arr, first_cell_x, steven_y, cell_w, cell_h, img.width)
    print(f"  Extracted {len(cells)} day columns")

    if len(cells) == 0:
        print("ERROR: No cells extracted. Check anchor placement.")
        sys.exit(1)

    # --- Classify each cell ---
    year, month, day = args.year, args.month, args.day
    assignments = []
    uncertain = []
    skipped = 0

    print(f"\nClassification (threshold={args.threshold}):")
    for i, (x_pos, cell_arr) in enumerate(cells):
        dow = day_of_week(year, month, day)
        is_weekend = dow in WEEKEND
        date_str = f"{month:02d}/{day:02d}/{year}"
        display_date = f"{day:02d}.{month:02d}.{year}"

        label, dist, matched = classify_cell(cell_arr, ref_sigs, ref_labels, ref_names)

        # Advance date
        day += 1
        if day > days_in_month(year, month):
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1

        # Check confidence
        if dist > args.threshold:
            uncertain.append((date_str, display_date, dow, label, dist, matched))
            if args.debug:
                print(f"  {display_date} {dow}: {label:25s} dist={dist:5.1f} ← UNCERTAIN (>{args.threshold}) [{matched}]")
            continue

        if args.debug:
            print(f"  {display_date} {dow}: {label:25s} dist={dist:5.1f} [{matched}]")

        # Apply skip rules
        if label == "Normal":
            skipped += 1
            continue
        if label == "Frei" and is_weekend:
            skipped += 1
            continue
        if label == "Frei" and not is_weekend:
            assignments.append((date_str, "Frei"))
            if not args.debug:
                print(f"  {display_date} {dow}: Frei")
            continue

        assignments.append((date_str, label))
        if not args.debug:
            print(f"  {display_date} {dow}: {label}")

    print(f"\nSummary: {len(assignments)} entries, {skipped} skipped, {len(uncertain)} uncertain")

    if uncertain:
        print("\nUncertain cells (distance > threshold) — please check:")
        for date_str, display_date, dow, label, dist, matched in uncertain:
            print(f"  {display_date} {dow}: best guess={label}, dist={dist:.1f}, matched={matched}")

    # --- Debug output ---
    if args.debug:
        # Grid overlay on original
        debug_img = img.copy()
        draw = ImageDraw.Draw(debug_img)
        # Draw anchor boxes
        draw.rectangle([tx1, ty1, tx2, ty2], outline="magenta", width=2)
        draw.rectangle([lx1, ly1, lx2, ly2], outline="magenta", width=2)
        # Draw cell grid
        for i, (x_pos, _) in enumerate(cells):
            draw.rectangle(
                [x_pos, steven_y, x_pos + cell_w - 1, steven_y + cell_h - 1],
                outline="red", width=1
            )
        debug_img.save("debug_grid.png")
        print("Saved debug_grid.png")

        # Contact sheet
        scale = 4
        pad = 3
        n = len(cells)
        sh = cell_h * scale + 25
        sw = n * (cell_w * scale + pad) + pad
        if sw > 0:
            sheet = Image.new("RGB", (sw, sh), (220, 220, 220))
            sdraw = ImageDraw.Draw(sheet)
            for i, (_, cell_arr) in enumerate(cells):
                cell_img = Image.fromarray(cell_arr)
                scaled = cell_img.resize(
                    (cell_w * scale, cell_h * scale), Image.NEAREST)
                xp = pad + i * (cell_w * scale + pad)
                sheet.paste(scaled, (xp, 0))
                sdraw.text((xp + 2, cell_h * scale + 4), str(i+1), fill=(0,0,0))
            sheet.save("debug_cells.png")
            print("Saved debug_cells.png")

    # --- CSV ---
    to_csv(assignments, args.output)
    print(f"\nDone. Import {args.output} into Google Calendar via:")
    print("  calendar.google.com → Settings → Import & Export → Import")


if __name__ == "__main__":
    main()
