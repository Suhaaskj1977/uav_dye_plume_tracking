import glob
import os
import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================
IMAGE_FOLDER        = r"C:\Users\SUHAAS K J\Downloads\QUAD500"           # ← change to your folder path
VIDEO_PATH          = "dye_evolution.mp4"
VIDEO_FPS           = 5
GREEN_EXCESS_THRESH = 50
PAD                 = 20                  # pixels of padding around tight dye crop

ROI_X, ROI_Y = 1750, 2620
ROI_W, ROI_H = 530, 430

CANVAS_W, CANVAS_H = 800, 600

# ============================================================
# PASS 1 — find the largest tight dye bbox across all frames
#           so canvas zoom stays fixed throughout
# ============================================================
files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.JPG")))
if not files:
    files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))

print(f"Found {len(files)} images")
print("Pass 1: finding global dye extent...")

max_w, max_h = 0, 0

for path in files:
    img = cv2.imread(path)
    if img is None:
        continue
    crop = img[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
    b, g, r = cv2.split(crop)
    ge = g.astype(np.int16) - ((r.astype(np.int16) + b.astype(np.int16)) // 2)
    mask = (ge > GREEN_EXCESS_THRESH).astype(np.uint8) * 255
    k = np.ones((15,15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            _, _, w, h = cv2.boundingRect(c)
            max_w = max(max_w, w + 2*PAD)
            max_h = max(max_h, h + 2*PAD)

if max_w == 0:
    print("ERROR: no dye found"); exit(1)

print(f"Max dye size across all frames: {max_w} x {max_h} px")

# ============================================================
# PASS 2 — build video
# ============================================================
print(f"Pass 2: building video...")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_PATH, fourcc, VIDEO_FPS, (CANVAS_W, CANVAS_H))

detected = 0
for i, path in enumerate(files):
    filename = os.path.basename(path)
    img = cv2.imread(path)
    if img is None:
        continue

    # Segment dye
    crop = img[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
    b, g, r = cv2.split(crop)
    ge = g.astype(np.int16) - ((r.astype(np.int16) + b.astype(np.int16)) // 2)
    mask = (ge > GREEN_EXCESS_THRESH).astype(np.uint8) * 255
    k = np.ones((15,15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Plain dark background
    canvas = np.full((CANVAS_H, CANVAS_W, 3), (25, 18, 10), dtype=np.uint8)

    dye_found = False
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            dye_found = True
            detected += 1
            x, y, w, h = cv2.boundingRect(c)

            # Tight crop around just dye bbox with padding
            x1 = max(0, x - PAD);    y1 = max(0, y - PAD)
            x2 = min(ROI_W, x+w+PAD); y2 = min(ROI_H, y+h+PAD)

            tight_crop = crop[y1:y2, x1:x2].copy()
            tight_mask = mask[y1:y2, x1:x2]

            # Zero out everything that is not dye
            dye_only = np.zeros_like(tight_crop)
            dye_only[tight_mask > 0] = tight_crop[tight_mask > 0]

            # Scale up to fixed size (max_w x max_h) so zoom is consistent
            scale = min((CANVAS_W - 80) / max_w, (CANVAS_H - 80) / max_h)
            out_w = int(max_w * scale)
            out_h = int(max_h * scale)

            # Embed dye_only inside a blank frame of size max_w x max_h
            scale = min((CANVAS_W - 80) / max_w,
            (CANVAS_H - 80) / max_h)

            out_w = int((x2 - x1) * scale)
            out_h = int((y2 - y1) * scale)

            resized = cv2.resize(tight_crop, (out_w, out_h),
                                interpolation=cv2.INTER_LINEAR)

            xo = (CANVAS_W - out_w) // 2
            yo = (CANVAS_H - out_h) // 2

            canvas[yo:yo+out_h, xo:xo+out_w] = resized

    # Text
    cv2.putText(canvas, filename, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Frame {i+1}/{len(files)}", (12, CANVAS_H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130,130,130), 1, cv2.LINE_AA)
    label = "DYE DETECTED" if dye_found else "NOT DETECTED"
    color = (40,220,40) if dye_found else (40,40,180)
    cv2.putText(canvas, label, (CANVAS_W-180, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    writer.write(canvas)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(files)} done...")

writer.release()
print(f"\nVideo saved → {VIDEO_PATH}")
print(f"Dye detected in {detected}/{len(files)} frames")