import csv
import math
import numpy as np
from scipy.io import loadmat
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import glob
from datetime import datetime

# --------------------------------------------------
# Helper: parse EXIF
# --------------------------------------------------

def get_exif_data(img):
    exif_data = {}
    info = img._getexif()
    if info is None:
        return None
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            gps_data = {}
            for t in value:
                sub_decoded = GPSTAGS.get(t, t)
                gps_data[sub_decoded] = value[t]
            exif_data["GPSInfo"] = gps_data
        else:
            exif_data[decoded] = value
    return exif_data


def convert_to_degrees(value):
    d, m, s = float(value[0]), float(value[1]), float(value[2])
    return d + m / 60.0 + s / 3600.0


def get_lat_lon_alt(exif_data):
    gps = exif_data.get("GPSInfo", None)
    if gps is None:
        return None, None, None
    lat = convert_to_degrees(gps["GPSLatitude"])
    if gps.get("GPSLatitudeRef", "N") != "N":
        lat = -lat
    lon = convert_to_degrees(gps["GPSLongitude"])
    if gps.get("GPSLongitudeRef", "E") != "E":
        lon = -lon
    alt = gps.get("GPSAltitude", None)
    if alt is not None:
        alt = float(alt)
    return lat, lon, alt


def get_timestamp(exif_data):
    dt_str = exif_data.get("DateTimeOriginal", None)
    subsec = exif_data.get("SubsecTimeOriginal", "0")
    if dt_str is None:
        return None, None
    try:
        dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        subsec_sec = int(subsec) / 10000.0
        total_sec = dt.timestamp() + subsec_sec
        return total_sec, f"{dt_str}.{subsec}"
    except Exception:
        return None, None


# --------------------------------------------------
# STEP 1: Load images from quad25/
# --------------------------------------------------

print("Loading image EXIF data ...")
files = sorted(glob.glob("QUAD500/*.JPG"))
images = []

for path in files:
    img = Image.open(path)
    exif = get_exif_data(img)
    filename = path.replace("\\", "/").split("/")[-1]

    lat, lon, alt = None, None, None
    ts_sec, ts_str = None, None

    if exif:
        lat, lon, alt = get_lat_lon_alt(exif)
        ts_sec, ts_str = get_timestamp(exif)

    images.append({
        "filename": filename,
        "img_datetime": ts_str,
        "img_timestamp_sec": ts_sec,
        "img_lat": lat,
        "img_lon": lon,
        "img_alt_m": alt,
    })

print(f"  Loaded {len(images)} images.")


# --------------------------------------------------
# STEP 2: Load AHR2 from logfile2.mat
# --------------------------------------------------

print("Loading log file ...")
data = loadmat("logfile2.mat", struct_as_record=False, squeeze_me=True)

ahr2 = data["AHR2"]

log_time_us = ahr2[:, 1]
log_roll    = ahr2[:, 2]
log_pitch   = ahr2[:, 3]
log_yaw     = ahr2[:, 4]
log_alt     = ahr2[:, 5]
log_lat     = ahr2[:, 6]
log_lon     = ahr2[:, 7]

print(f"  Loaded {len(log_time_us)} log rows.")


# --------------------------------------------------
# STEP 3: Match each image to nearest log row by lat/lon
# --------------------------------------------------

print("Matching images to log rows by nearest lat/lon ...")

log_lat_arr = np.array(log_lat)
log_lon_arr = np.array(log_lon)

records = []

for img in images:
    img_lat = img["img_lat"]
    img_lon = img["img_lon"]

    if img_lat is None or img_lon is None:
        records.append({**img,
                        "log_idx": None, "log_time_us": None,
                        "log_lat": None, "log_lon": None, "log_alt": None,
                        "roll": None, "pitch": None, "yaw": None,
                        "latlon_dist_deg": None})
        continue

    # Euclidean distance in lat/lon degrees (sufficient at small scales)
    dists = np.sqrt((log_lat_arr - img_lat) ** 2 + (log_lon_arr - img_lon) ** 2)
    best = int(np.argmin(dists))

    records.append({
        **img,
        "log_idx":        best,
        "log_time_us":    float(log_time_us[best]),
        "log_lat":        float(log_lat[best]),
        "log_lon":        float(log_lon[best]),
        "log_alt":        float(log_alt[best]),
        "roll":           float(log_roll[best]),
        "pitch":          float(log_pitch[best]),
        "yaw":            float(log_yaw[best]),
        "latlon_dist_deg": float(dists[best]),
    })


# --------------------------------------------------
# STEP 4: Save merged CSV
# --------------------------------------------------

output_csv = "matched_image_log_500.csv"
fieldnames = [
    "filename", "img_datetime", "img_timestamp_sec",
    "img_lat", "img_lon", "img_alt_m",
    "log_idx", "log_time_us",
    "log_lat", "log_lon", "log_alt",
    "roll", "pitch", "yaw",
    "latlon_dist_deg",
]

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"\nSaved {len(records)} rows to {output_csv}")

# --------------------------------------------------
# STEP 5: Quick sanity check — print first 5 rows
# --------------------------------------------------

print("\nFirst 5 matches:")
for r in records[:5]:
    print(f"  {r['filename']} | img_lat={r['img_lat']:.6f} log_lat={r['log_lat']:.6f} "
          f"| dist={r['latlon_dist_deg']:.6f} deg "
          f"| roll={r['roll']:.2f} pitch={r['pitch']:.2f} yaw={r['yaw']:.2f}")

# Flag any suspiciously large mismatches
bad = [r for r in records if r["latlon_dist_deg"] is not None and r["latlon_dist_deg"] > 0.001]
if bad:
    print(f"\nWARNING: {len(bad)} images have lat/lon match distance > 0.001 deg (~100 m). Check those rows.")
else:
    print("\nAll matches look good (distance < 0.001 deg).")
