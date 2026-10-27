import base64, json, os, sqlite3, time, uuid
from pathlib import Path
import cv2, numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
# Vercel's filesystem is read-only except /tmp, and data there doesn't persist
# between invocations. Locally this keeps using the project's own data/ folder.
DATA_DIR = Path("/tmp") if os.environ.get("VERCEL") else BASE / "data"
UPLOADS = DATA_DIR / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
DB = DATA_DIR / "phyvosense.db"

def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

with db() as c:
    c.execute("""CREATE TABLE IF NOT EXISTS analyses(
        id TEXT PRIMARY KEY, created REAL, batch TEXT, filename TEXT, result TEXT)""")

# ---------- image analysis ----------
def crop_plate(img):
    b, g, r = [c.astype(np.int16) for c in cv2.split(img)]
    mask = ((g > 90) & (g > r + 25) & (g > b + 25)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img.shape[:2]
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        if w * h > 0.15 * H * W:
            mx, my = int(w * .03), int(h * .03)
            return (x + mx, y + my, w - 2 * mx, h - 2 * my), True
    return (0, 0, W, H), False

def parse_layout(text):
    out = []
    for t in text.replace(";", ",").split(","):
        t = t.strip().lower()
        if not t:
            continue
        if t in ("?", "s", "sample", "ext"):
            out.append(None)
        else:
            try:
                v = float(t)
            except ValueError:
                raise HTTPException(400, f"Lane '{t}' is not a number or '?'.")
            if v <= 0:
                raise HTTPException(400, "Standard amounts must be above zero.")
            out.append(v)
    if not 3 <= len(out) <= 12:
        raise HTTPException(400, "Lane layout needs 3 to 12 lanes.")
    if sum(v is not None for v in out) < 2:
        raise HTTPException(400, "Mark at least 2 lanes as standards (numbers, in µg).")
    if all(v is not None for v in out):
        raise HTTPException(400, "Mark at least 1 lane as a sample with '?'.")
    return out

def analyze(img, layout, origin=0.90, front=0.06, tol=0.06, volume=None):
    s = 1600 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    (x, y, w, h), found = crop_plate(img)
    plate = img[y:y + h, x:x + w].copy()
    g = cv2.GaussianBlur(plate[:, :, 1].astype(np.float32), (0, 0), 2)
    n = len(layout)
    oy, fy = origin * h, front * h
    lo, hi = int(fy + 0.02 * h), int(oy - 0.04 * h)
    if hi - lo < 50:
        raise HTTPException(400, "Origin and front lines leave too little room. Check the percentages.")
    ys = np.arange(lo, hi)
    rf = (oy - ys) / (oy - fy)
    lane_w = w / n
    k = max(3, int(h / 120)) | 1
    sigs = []
    for i in range(n):
        x0, x1 = int(i * lane_w + lane_w * .25), int((i + 1) * lane_w - lane_w * .25)
        p = g[lo:hi, x0:x1].mean(axis=1)
        p = np.convolve(np.pad(p, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        sigs.append(np.clip(np.percentile(p, 90) - p, 0, None))
    std_idx = [i for i, v in enumerate(layout) if v is not None]
    rf_t = float(rf[int(np.argmax(sum(sigs[i] for i in std_idx)))])
    win = np.abs(rf - rf_t) <= tol
    areas = [float(sg[win].sum()) for sg in sigs]
    warnings = []
    if not found:
        warnings.append("Plate edges were not detected, so the whole photo was used. Retake with the plate filling the frame.")
    amts = np.array([layout[i] for i in std_idx]); ars = np.array([areas[i] for i in std_idx])
    m, b = np.polyfit(amts, ars, 1)
    ss = ((ars - (m * amts + b)) ** 2).sum(); st = ((ars - ars.mean()) ** 2).sum()
    r2 = float(1 - ss / st) if st > 0 else 0.0
    if len(std_idx) < 3:
        warnings.append("Only 2 standards: the calibration has no way to check linearity. Use 3 or more.")
    if r2 < 0.95:
        warnings.append(f"Calibration fit is weak (R² = {r2:.3f}). Treat sample values as unreliable.")
    if m <= 0:
        raise HTTPException(400, "Standards do not get darker with amount. Check the lane layout order.")
    lanes = []
    for i in range(n):
        sg = sigs[i]
        noise = float(np.median(np.abs(sg[~win] - np.median(sg[~win]))) * 1.4826) if (~win).any() else 0.0
        peak = float(sg[win].max())
        detected = peak > max(3 * noise, 3.0)
        lane = {"lane": i + 1, "type": "standard" if layout[i] is not None else "sample",
                "area": round(areas[i], 1), "detected": detected,
                "peak_rf": round(float(rf[win][np.argmax(sg[win])]), 3)}
        if layout[i] is not None:
            lane["known_ug"] = layout[i]
            lane["est_ug"] = round((areas[i] - b) / m, 3)
        else:
            est = (areas[i] - b) / m
            if not detected:
                lane["status"] = "not detected"
            elif est < amts.min():
                lane["status"] = f"below lowest standard (<{amts.min():g} µg)"; lane["est_ug"] = round(max(est, 0), 3)
            elif est > amts.max():
                lane["status"] = "above range (extrapolated)"; lane["est_ug"] = round(est, 3)
                warnings.append(f"Lane {i+1} is above the highest standard. Dilute and rerun for a reliable value.")
            else:
                lane["status"] = "in range"; lane["est_ug"] = round(est, 3)
            if volume and "est_ug" in lane:
                lane["mg_per_ml"] = round(lane["est_ug"] / volume, 3)
        lanes.append(lane)
    # annotated image
    ann = plate.copy()
    yr = lambda r: int(oy - r * (oy - fy))
    cv2.rectangle(ann, (0, yr(rf_t + tol)), (w - 1, yr(rf_t - tol)), (0, 140, 255), 2)
    cv2.line(ann, (0, int(oy)), (w, int(oy)), (0, 255, 255), 1)
    cv2.line(ann, (0, int(fy)), (w, int(fy)), (255, 255, 0), 1)
    for i in range(n):
        if i: cv2.line(ann, (int(i * lane_w), 0), (int(i * lane_w), h), (255, 255, 255), 1)
        cv2.putText(ann, str(i + 1), (int(i * lane_w + lane_w / 2 - 6), h - 8), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    a = 700 / ann.shape[1]
    if a < 1: ann = cv2.resize(ann, None, fx=a, fy=a, interpolation=cv2.INTER_AREA)
    ann_b64 = base64.b64encode(cv2.imencode(".jpg", ann)[1]).decode()
    step = max(1, len(rf) // 140)
    return {"plate_found": found, "rf_target": round(rf_t, 3), "tol": tol,
            "calibration": {"slope": float(m), "intercept": float(b), "r2": round(r2, 4), "n": len(std_idx)},
            "lanes": lanes, "warnings": warnings, "annotated": ann_b64,
            "profiles": {"rf": [round(float(v), 3) for v in rf[::step]],
                         "signal": [[round(float(v), 2) for v in sg[::step]] for sg in sigs]}}

def demo_image():
    rng = np.random.default_rng(7)
    img = np.zeros((680, 440, 3), np.uint8)
    img[40:640, 40:400] = (50, 210, 40)
    img = img.astype(np.float32)
    x0, w, y0, h = 40 + .03 * 360, .94 * 360, 40 + .03 * 600, .94 * 600
    yy, xx = np.mgrid[0:680, 0:440]
    amounts = [3.0, 5.0, 2.5, 1.25, 0.63]
    for i, a in enumerate(amounts):
        cx = x0 + (i + .5) * w / 5
        cy = y0 + (.90 - .5 * .84) * h
        img[:, :, 1] -= 22 * a * np.exp(-((xx - cx) ** 2) / (2 * 12 ** 2) - ((yy - cy) ** 2) / (2 * 9 ** 2))
        oy = y0 + .90 * h
        img[:, :, 1] -= 70 * np.exp(-((xx - cx) ** 2 + (yy - oy) ** 2) / (2 * 6 ** 2))
    img[:, :, 1] += rng.normal(0, 2, img.shape[:2])
    return np.clip(img, 0, 255).astype(np.uint8)

def store(res, filename, batch, raw):
    aid = uuid.uuid4().hex[:10]
    (UPLOADS / f"{aid}.jpg").write_bytes(raw)
    with db() as c:
        c.execute("INSERT INTO analyses VALUES(?,?,?,?,?)", (aid, time.time(), batch, filename, json.dumps(res)))
    res.update(id=aid, batch=batch, filename=filename)
    return res

# ---------- API ----------
app = FastAPI(title="PhyvoSense CordyScan MVP")

@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...), layout: str = Form("?,5.0,2.5,1.25,0.63"),
                      batch: str = Form(""), origin: float = Form(90), front: float = Form(6),
                      tol: float = Form(6), volume: float = Form(0)):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "That file is not a readable image (use JPG or PNG).")
    if not (0 < front < origin < 100):
        raise HTTPException(400, "Front % must be smaller than origin % (both measured from the top).")
    res = analyze(img, parse_layout(layout), origin / 100, front / 100, tol / 100, volume or None)
    return store(res, file.filename, batch, raw)

@app.post("/api/demo")
def api_demo():
    img = demo_image()
    raw = cv2.imencode(".jpg", img)[1].tobytes()
    res = analyze(img, parse_layout("?,5.0,2.5,1.25,0.63"))
    return store(res, "synthetic-demo.jpg", "Demo plate (true sample = 3.0 µg)", raw)

@app.get("/api/history")
def history():
    with db() as c:
        rows = c.execute("SELECT id,created,batch,filename,result FROM analyses ORDER BY created DESC LIMIT 50").fetchall()
    out = []
    for r in rows:
        lanes = json.loads(r["result"])["lanes"]
        out.append({"id": r["id"], "created": r["created"], "batch": r["batch"], "filename": r["filename"],
                    "samples": [l.get("est_ug") for l in lanes if l["type"] == "sample"]})
    return out

@app.get("/api/analysis/{aid}")
def get_analysis(aid: str):
    with db() as c:
        r = c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
    if not r:
        raise HTTPException(404, "Analysis not found.")
    res = json.loads(r["result"]); res.update(id=aid, batch=r["batch"], filename=r["filename"], created=r["created"])
    return res

app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
