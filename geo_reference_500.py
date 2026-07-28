import csv
import glob
import math
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
IMAGE_FOLDER         = "QUAD500"           # ← change to your folder path
TELEMETRY_CSV        = "matched_image_log_500.csv"
OUTPUT_CSV           = "georef_centroids_500.csv"
PLOT_PATH            = "plume_trajectory_500.png"

CAMERA_FOV_H_DEG     = 94.4
IMAGE_W              = 5568
IMAGE_H              = 4176

GREEN_EXCESS_THRESH  = 90
MIN_CONTOUR_AREA     = 500                 # px² inside ROI crop
MIN_DT               = 0.5                # seconds — ignore smaller gaps
MAX_PROJ_DIST_M      = 200.0              # sanity check — skip if >200m from drone

# Fixed ROI — same for every image
ROI_X, ROI_Y = 1750, 2620
ROI_W, ROI_H = 530, 430

# ============================================================
# Camera intrinsics
# ============================================================
fov_h_rad = math.radians(CAMERA_FOV_H_DEG)
fx = IMAGE_W / (2.0 * math.tan(fov_h_rad / 2.0))
fy = fx

# ============================================================
# Rotation helpers
# ============================================================
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

R_CAM_TO_BODY = np.array([[ 0, -1,  0],
                           [ 1,  0,  0],
                           [ 0,  0,  1]], dtype=float)

# ============================================================
# Geo-projection
# ============================================================
def pixel_to_latlon(cx, cy, img_w, img_h, drone_lat, drone_lon, alt_m,
                    roll_deg, pitch_deg, yaw_deg):
    scale_x   = IMAGE_W / img_w
    scale_y   = IMAGE_H / img_h
    fx_scaled = fx / scale_x
    fy_scaled = fy / scale_y

    dx = (cx - img_w / 2.0) / fx_scaled
    dy = (cy - img_h / 2.0) / fy_scaled
    ray_cam = np.array([dx, dy, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)

    ray_body = R_CAM_TO_BODY @ ray_cam

    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    R_body_to_ned = rot_z(y) @ rot_y(p) @ rot_x(r)
    ray_ned = R_body_to_ned @ ray_body

    if ray_ned[2] <= 0:
        return None, None

    t = alt_m / ray_ned[2]
    offset_north_m = t * ray_ned[0]
    offset_east_m  = t * ray_ned[1]

    # Sanity check — projected point must be within MAX_PROJ_DIST_M of drone
    dist = math.hypot(offset_north_m, offset_east_m)
    if dist > MAX_PROJ_DIST_M:
        return None, None

    lat_target = drone_lat + offset_north_m / 111111.0
    lon_target = drone_lon + offset_east_m  / (111111.0 * math.cos(math.radians(drone_lat)))
    return lat_target, lon_target

# ============================================================
# STEP 1: Load telemetry
# ============================================================
log_by_filename = {}
with open(TELEMETRY_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        log_by_filename[row["filename"]] = row

print(f"Loaded {len(log_by_filename)} telemetry entries")
print(f"ROI: x={ROI_X} y={ROI_Y} w={ROI_W} h={ROI_H}\n")

# ============================================================
# STEP 2: Process each image
# ============================================================
files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.JPG")))
if not files:
    files = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))

print(f"Found {len(files)} images in '{IMAGE_FOLDER}'")

results      = []
skipped_log  = 0
skipped_read = 0
skipped_proj = 0
detected     = 0

for i, path in enumerate(files):
    filename = path.replace("\\", "/").split("/")[-1]

    # Progress counter
    if (i + 1) % 50 == 0 or (i + 1) == len(files):
        print(f"  [{i+1}/{len(files)}] processed...")

    if filename not in log_by_filename:
        skipped_log += 1
        continue

    log       = log_by_filename[filename]
    drone_lat = float(log["img_lat"])
    drone_lon = float(log["img_lon"])
    alt_m     = float(log["img_alt_m"])
    roll_deg  = float(log["roll"])
    pitch_deg = float(log["pitch"])
    yaw_deg   = float(log["yaw"])
    ts_sec    = float(log["img_timestamp_sec"]) if log["img_timestamp_sec"] else None

    img = cv2.imread(path)
    if img is None:
        skipped_read += 1
        continue

    img_h, img_w = img.shape[:2]

    # ── Crop to fixed ROI ────────────────────────────────────
    roi_crop = img[ROI_Y: ROI_Y + ROI_H, ROI_X: ROI_X + ROI_W]

    # ── Green excess segmentation ────────────────────────────
    b, g, r = cv2.split(roi_crop)
    green_excess = g.astype(np.int16) - ((r.astype(np.int16) + b.astype(np.int16)) // 2)
    mask = (green_excess > GREEN_EXCESS_THRESH).astype(np.uint8) * 255

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Find centroid ────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx_roi, cy_roi = None, None

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > MIN_CONTOUR_AREA:
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx_roi = int(M["m10"] / M["m00"])
                cy_roi = int(M["m01"] / M["m00"])
                detected += 1

    # ── Convert ROI → full image coords ─────────────────────
    cx_full = cx_roi + ROI_X if cx_roi is not None else None
    cy_full = cy_roi + ROI_Y if cy_roi is not None else None

    # ── Geo-projection ───────────────────────────────────────
    plume_lat, plume_lon = None, None
    if cx_full is not None:
        plume_lat, plume_lon = pixel_to_latlon(
            cx_full, cy_full, img_w, img_h,
            drone_lat, drone_lon, alt_m,
            roll_deg, pitch_deg, yaw_deg
        )
        if plume_lat is None:
            skipped_proj += 1

    results.append({
        "filename":      filename,
        "img_datetime":  log["img_datetime"],
        "timestamp_sec": ts_sec,
        "drone_lat":     drone_lat,
        "drone_lon":     drone_lon,
        "alt_m":         alt_m,
        "roll_deg":      roll_deg,
        "pitch_deg":     pitch_deg,
        "yaw_deg":       yaw_deg,
        "px_cx_roi":     cx_roi,
        "px_cy_roi":     cy_roi,
        "px_cx_full":    cx_full,
        "px_cy_full":    cy_full,
        "plume_lat":     plume_lat,
        "plume_lon":     plume_lon,
        "v_north_mps":   None,
        "v_east_mps":    None,
        "speed_mps":     None,
    })

print(f"\nDetection summary:")
print(f"  Total images     : {len(files)}")
print(f"  No telemetry     : {skipped_log}")
print(f"  Could not read   : {skipped_read}")
print(f"  Dye detected     : {detected}")
print(f"  Projection failed: {skipped_proj}")

# ============================================================
# STEP 3: Compute frame-to-frame velocity
# ============================================================
for i in range(1, len(results)):
    curr = results[i]
    prev = results[i - 1]

    if (curr["plume_lat"] is None or prev["plume_lat"] is None or
            curr["timestamp_sec"] is None or prev["timestamp_sec"] is None):
        continue

    dt = curr["timestamp_sec"] - prev["timestamp_sec"]
    if dt < MIN_DT:
        continue

    mean_lat = (curr["plume_lat"] + prev["plume_lat"]) / 2.0
    d_north  = (curr["plume_lat"] - prev["plume_lat"]) * 111111.0
    d_east   = (curr["plume_lon"] - prev["plume_lon"]) * 111111.0 * math.cos(math.radians(mean_lat))

    curr["v_north_mps"] = d_north / dt
    curr["v_east_mps"]  = d_east  / dt
    curr["speed_mps"]   = math.hypot(curr["v_north_mps"], curr["v_east_mps"])

# ============================================================
# STEP 4: Save CSV
# ============================================================
fieldnames = [
    "filename", "img_datetime", "timestamp_sec",
    "drone_lat", "drone_lon", "alt_m",
    "roll_deg", "pitch_deg", "yaw_deg",
    "px_cx_roi", "px_cy_roi", "px_cx_full", "px_cy_full",
    "plume_lat", "plume_lon",
    "v_north_mps", "v_east_mps", "speed_mps",
]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved {len(results)} rows → {OUTPUT_CSV}")

# ============================================================
# STEP 5: Velocity summary
# ============================================================
velocities = [r for r in results if r["speed_mps"]  is not None]
valid      = [r for r in results if r["plume_lat"]  is not None
                                 and r["timestamp_sec"] is not None]

if velocities:
    vn = [r["v_north_mps"] for r in velocities]
    ve = [r["v_east_mps"]  for r in velocities]
    sp = [r["speed_mps"]   for r in velocities]

    print("\nVelocity summary:")
    print(f"  Mean v_north : {np.mean(vn):.4f} m/s")
    print(f"  Mean v_east  : {np.mean(ve):.4f} m/s")
    print(f"  Mean speed   : {np.mean(sp):.4f} m/s")
    print(f"  Max  speed   : {np.max(sp):.4f} m/s")
    heading = math.degrees(math.atan2(np.mean(ve), np.mean(vn))) % 360
    print(f"  Mean heading : {heading:.1f}°  (0=N, 90=E)")

if len(valid) >= 2:
    first = valid[0]
    last  = valid[-1]
    total_dt    = last["timestamp_sec"]  - first["timestamp_sec"]
    mean_lat    = (first["plume_lat"]    + last["plume_lat"])  / 2.0
    total_north = (last["plume_lat"]     - first["plume_lat"]) * 111111.0
    total_east  = (last["plume_lon"]     - first["plume_lon"]) * 111111.0 * math.cos(math.radians(mean_lat))
    total_dist  = math.hypot(total_north, total_east)
    bulk_speed  = total_dist / total_dt
    bulk_heading = math.degrees(math.atan2(total_east, total_north)) % 360

    print(f"\nBulk displacement (first → last valid frame):")
    print(f"  Total time   : {total_dt:.1f} s")
    print(f"  Total dist   : {total_dist:.2f} m")
    print(f"  Bulk speed   : {bulk_speed:.4f} m/s  ← most reliable")
    print(f"  Bulk heading : {bulk_heading:.1f}°")

# ============================================================
# STEP 6: Plots
# ============================================================
if len(valid) >= 2:
    lats    = [r["plume_lat"] for r in valid]
    lons    = [r["plume_lon"] for r in valid]
    ref_lat, ref_lon = lats[0], lons[0]
    north_m = [(la - ref_lat) * 111111.0 for la in lats]
    east_m  = [(lo - ref_lon) * 111111.0 * math.cos(math.radians(ref_lat)) for lo in lons]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: spatial trajectory
    ax = axes[0]
    sc = ax.scatter(east_m, north_m, c=range(len(east_m)), cmap="viridis", s=40, zorder=3)
    ax.plot(east_m, north_m, "k--", linewidth=0.6, zorder=2)
    ax.scatter([east_m[0]],  [north_m[0]],  color="green", s=100, zorder=4, label="Start")
    ax.scatter([east_m[-1]], [north_m[-1]], color="red",   s=100, zorder=4, label="End")
    ax.annotate("", xy=(east_m[-1], north_m[-1]), xytext=(east_m[0], north_m[0]),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    plt.colorbar(sc, ax=ax, label="Frame index")
    ax.set_xlabel("East offset (m)")
    ax.set_ylabel("North offset (m)")
    ax.set_title("Plume centroid trajectory (500 images)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Right: speed over time
    ax2 = axes[1]
    if velocities:
        ts0   = valid[0]["timestamp_sec"]
        times = [r["timestamp_sec"] - ts0 for r in velocities]
        sp    = [r["speed_mps"] for r in velocities]
        ax2.plot(times, sp, color="steelblue", linewidth=1.0)
        ax2.axhline(bulk_speed,    color="red",    linestyle="--", label=f"Bulk  {bulk_speed:.3f} m/s")
        ax2.axhline(np.mean(sp),   color="orange", linestyle="--", label=f"Mean  {np.mean(sp):.3f} m/s")
        ax2.set_xlabel("Time from start (s)")
        ax2.set_ylabel("Speed (m/s)")
        ax2.set_title("Frame-to-frame speed (500 images)")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()
    print(f"\nPlot saved → {PLOT_PATH}")