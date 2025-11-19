# app.py
import os
import time
import uuid
import json
import re
from datetime import datetime
from urllib.parse import quote, unquote

from flask import Flask, request, render_template, send_from_directory, jsonify, g, send_file
from werkzeug.utils import secure_filename
from PIL import Image

# --- try load YOLO if available; in case user doesn't have ultralytics installed we'll still start
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

# --- SQLAlchemy DB setup ---
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

# --- optional embeddings (sentence-transformers) ---
try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDING_AVAILABLE = True
except Exception:
    EMBEDDING_AVAILABLE = False

# -------------------------
# Config
# -------------------------
app = Flask(__name__)

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
HEATMAP_DIR = os.path.join(UPLOAD_DIR, "heatmaps")
OUTPUT_BASE = os.path.abspath(os.path.join(BASE_DIR, "runs", "detect", "predict"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Path to your YOLO model - change as needed
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")  # adjust this; example path

# In-memory index for quick lookup
AUDIT_STORE = {}

# -------------------------
# Load model (lazy if ultralytics present)
MODEL_VERSION = "custom-best"
_model = None
if ULTRALYTICS_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        # load immediately but guarded
        _model = YOLO(MODEL_PATH)
        MODEL_VERSION = getattr(_model, "model", None) or MODEL_VERSION
    except Exception as e:
        app.logger.warning("Failed to load YOLO model at startup: %s", e)
        _model = None

def get_model():
    global _model
    if not ULTRALYTICS_AVAILABLE:
        return None
    if _model is None:
        # lazy load
        try:
            _model = YOLO(MODEL_PATH)
        except Exception as e:
            app.logger.exception("Failed to load model: %s", e)
            _model = None
    return _model

# -------------------------
# Database (SQLite) config
DB_PATH = os.path.join(BASE_DIR, "audits.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_PATH}"
engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=False, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

class Audit(Base):
    __tablename__ = "audits"
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(64), index=True, unique=True)
    path = Column(String(256))
    method = Column(String(16))
    request_body = Column(Text)
    response_snippet = Column(Text)
    duration_ms = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    extra = Column(Text)

Base.metadata.create_all(bind=engine)

# -------------------------
# Utilities
def now_ts():
    return datetime.utcnow().isoformat() + "Z"

def calibrate_confidence(raw_conf):
    cal = max(0.0, min(1.0, raw_conf - 0.02))
    return round(cal, 4)

def compute_quality_score(probabilities, metadata=None):
    if not probabilities:
        return 0.0
    confidence = max(probabilities.values()) if probabilities else 0.0
    completeness = 1.0
    explainability = 1.0
    score = confidence * 0.7 + completeness * 0.2 + explainability * 0.1
    return round(score, 4)

def append_audit(audit):
    """Write audit to DB and keep runtime index."""
    db = SessionLocal()
    try:
        a = Audit(
            request_id=audit.get("request_id"),
            path=audit.get("path"),
            method=audit.get("method"),
            request_body=json.dumps(audit.get("request_body"), default=str) if audit.get("request_body") else None,
            response_snippet=(audit.get("response_snippet")[:2000] if audit.get("response_snippet") else None),
            duration_ms=audit.get("duration_ms"),
            timestamp=datetime.utcnow(),
            extra=json.dumps(audit, default=str)
        )
        db.add(a)
        db.commit()
        db.refresh(a)
        if a.request_id:
            AUDIT_STORE[a.request_id] = audit
    except Exception as e:
        app.logger.exception("Failed to write audit to DB: %s", e)
        if audit.get("request_id"):
            AUDIT_STORE[audit.get("request_id")] = audit
    finally:
        try:
            db.close()
        except Exception:
            pass

def generate_heatmap_stub(file_path, out_dir):
    # placeholder - return None or a placeholder URL path if you implement later
    return None

def unique_filename(filename):
    """Return a secure filename with a short uuid suffix to avoid overwrite."""
    filename = secure_filename(filename)
    if not filename:
        return f"{uuid.uuid4().hex}.jpg"
    name, ext = os.path.splitext(filename)
    if not ext:
        ext = ".jpg"
    return f"{name}_{uuid.uuid4().hex}{ext}"

# -------------------------
# Request timing middleware
@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    try:
        if request.path.startswith("/api/v1/") or request.path == "/predict":
            duration_ms = int((time.time() - getattr(g, "start_time", time.time())) * 1000)
            body = None
            try:
                body = request.get_json(silent=True)
            except Exception:
                body = None
            try:
                snippet = response.get_data(as_text=True)[:2000]
            except Exception:
                snippet = ""
            audit = {
                "request_id": (body or {}).get("request_id") if isinstance(body, dict) else None,
                "path": request.path,
                "method": request.method,
                "request_body": body if body is not None else "<multipart or form data>",
                "response_status": response.status_code,
                "response_snippet": snippet,
                "duration_ms": duration_ms,
                "timestamp": now_ts()
            }
            append_audit(audit)
    except Exception:
        app.logger.exception("audit middleware error")
    return response

# -------------------------
# Safe result serving: only allow files under a few base dirs
ALLOWED_OUTPUT_DIRS = [UPLOAD_DIR, HEATMAP_DIR, OUTPUT_BASE]

def resolve_safe_path(encoded_or_raw_path):
    """Given a path (possibly URL-encoded absolute path), return safe absolute path
       only if it lies under one of ALLOWED_OUTPUT_DIRS."""
    try:
        # decode potential URL-encoding
        p = unquote(encoded_or_raw_path)
        p_abs = os.path.abspath(p)
        for base in ALLOWED_OUTPUT_DIRS:
            base_abs = os.path.abspath(base)
            try:
                if os.path.commonpath([p_abs, base_abs]) == base_abs:
                    return p_abs
            except Exception:
                continue
    except Exception:
        pass
    return None

# -------------------------
# Detection parsing helper (de-duplicate)
def parse_results_boxes(results, filename):
    detections = []
    probs = {}
    try:
        if len(results) and hasattr(results[0], "boxes"):
            for box in results[0].boxes:
                try:
                    cls = int(box.cls)
                except Exception:
                    cls = int(getattr(box, "class", 0))
                label = getattr(results[0], "names", {}).get(cls, str(cls)) if hasattr(results[0], "names") else str(cls)
                try:
                    conf = float(box.conf)
                except Exception:
                    conf = float(getattr(box, "confidence", 0.0))
                coords = None
                try:
                    xyxy = getattr(box, "xyxy", None)
                    if xyxy is not None:
                        vals = None
                        try:
                            vals = xyxy.cpu().numpy().tolist()
                        except Exception:
                            try:
                                vals = xyxy.tolist()
                            except Exception:
                                vals = None
                        if vals:
                            if isinstance(vals[0], (list, tuple)):
                                x1, y1, x2, y2 = map(float, vals[0])
                            else:
                                x1, y1, x2, y2 = map(float, vals)
                            w = x2 - x1
                            h = y2 - y1
                            coords = [int(x1), int(y1), int(w), int(h)]
                except Exception:
                    coords = None

                detections.append({"label": label, "confidence": round(conf, 4), "box": coords})
                probs[label] = max(probs.get(label, 0.0), conf)
    except Exception:
        app.logger.exception("parse_results_boxes error")
    return detections, probs

# -------------------------
# Routes: index and predict (HTML)
@app.route("/", methods=["GET"])
def index():
    try:
        return render_template("index.html")
    except Exception:
        return """
            <html><body>
            <h3>Model Assistant</h3>
            <form action="/predict" method="post" enctype="multipart/form-data">
              <input type="file" name="image" accept="image/*"/>
              <button type="submit">Upload & Predict</button>
            </form>
            <p>Or open <a href="/chatbot">/chatbot</a></p>
            </body></html>
        """

@app.route("/predict", methods=["POST"])
def predict_html():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    filename = unique_filename(file.filename or f"{uuid.uuid4().hex}.jpg")
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    t0 = time.time()
    model = get_model()
    if model:
        results = model.predict(file_path, save=True)
    else:
        results = []
    processing_ms = int((time.time() - t0) * 1000)

    try:
        im = Image.open(file_path)
        img_w, img_h = im.size
    except Exception:
        img_w = img_h = None

    detections, probs = parse_results_boxes(results, filename)

    raw_conf = max(probs.values()) if probs else 0.0
    calibrated = calibrate_confidence(raw_conf)
    quality_score = compute_quality_score(probs)
    request_id = str(uuid.uuid4())
    heatmap_url = generate_heatmap_stub(file_path, HEATMAP_DIR)

    audit = {
        "request_id": request_id,
        "type": "html_predict",
        "uploaded_filename": filename,
        "detections": detections,
        "probabilities": probs,
        "confidence": round(raw_conf, 4),
        "calibrated_confidence": calibrated,
        "quality_score": quality_score,
        "model_version": MODEL_VERSION,
        "processing_ms": processing_ms,
        "timestamp": now_ts()
    }
    append_audit(audit)

    # Try to find annotated output saved by YOLO
    output_file = None
    try:
        output_dir = results[0].save_dir if (results and hasattr(results[0], "save_dir")) else OUTPUT_BASE
        candidate = os.path.join(output_dir, filename)
        if os.path.exists(candidate):
            output_file = os.path.abspath(candidate)
        else:
            for root, _, files in os.walk(output_dir):
                for f in files:
                    if f == filename:
                        output_file = os.path.abspath(os.path.join(root, f))
                        break
                if output_file:
                    break
    except Exception:
        output_file = None

    result_image_url = None
    if output_file:
        # return encoded absolute path but only the allowed-serving route can serve if in allowed dirs
        result_image_url = "/result/" + quote(output_file, safe="")

    try:
        return render_template(
            "result.html",
            uploaded_image=filename,
            result_image_url=result_image_url,
            detections=detections,
            confidence=round(raw_conf, 4),
            calibrated_confidence=calibrated,
            quality_score=quality_score,
            request_id=request_id,
            heatmap_url=heatmap_url
        )
    except Exception:
        return jsonify({
            "request_id": request_id,
            "detections": detections,
            "confidence": round(raw_conf, 4),
            "calibrated_confidence": calibrated,
            "quality_score": quality_score,
            "heatmap_url": heatmap_url,
            "result_image": output_file
        })

# -------------------------
# API predict (multipart form-data)
@app.route("/api/v1/predict", methods=["POST"])
def predict_api():
    request_id = str(uuid.uuid4())
    t0 = time.time()

    if 'image' in request.files:
        file = request.files['image']
        filename = unique_filename(file.filename or f"{request_id}.jpg")
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)
    else:
        return jsonify({"error": "No image provided. Use multipart form-data field 'image'."}), 400

    model = get_model()
    if model:
        results = model.predict(file_path, save=False)
    else:
        results = []
    processing_ms = int((time.time() - t0) * 1000)

    detections, probs = parse_results_boxes(results, filename)

    raw_conf = max(probs.values()) if probs else 0.0
    calibrated = calibrate_confidence(raw_conf)
    quality_score = compute_quality_score(probs)

    heatmap_url = generate_heatmap_stub(file_path, HEATMAP_DIR)

    resp = {
        "request_id": request_id,
        "detections": detections,
        "probabilities": {k: round(v,4) for k,v in probs.items()},
        "confidence": round(raw_conf,4),
        "calibrated_confidence": calibrated,
        "quality_score": quality_score,
        "heatmap_url": heatmap_url,
        "model_version": MODEL_VERSION,
        "processing_ms": processing_ms,
        "timestamp": now_ts()
    }

    audit = {
        "request_id": request_id,
        "type": "api_predict",
        "request_form": "<multipart image>",
        "detections": detections,
        "probabilities": resp["probabilities"],
        "confidence": resp["confidence"],
        "calibrated_confidence": resp["calibrated_confidence"],
        "quality_score": resp["quality_score"],
        "model_version": resp["model_version"],
        "processing_ms": resp["processing_ms"],
        "timestamp": resp["timestamp"]
    }
    append_audit(audit)

        # Try to return normally; if Flask's JSON encoder fails (non-serializable object),
    # fall back to converting everything to JSON-serializable primitives using str() for unknown types.
    try:
        return jsonify(resp), 200
    except TypeError:
        app.logger.warning("resp contains non-serializable objects; falling back to str() conversion")
        # convert non-serializable objects to strings
        safe_json_text = json.dumps(resp, default=lambda o: f"<<non-serializable: {type(o).__name__}>> {str(o)[:200]}")
        # parse back to a Python object (so Flask's jsonify sees a normal dict/list)
        try:
            safe_obj = json.loads(safe_json_text)
            return jsonify(safe_obj), 200
        except Exception:
            # as a last resort return the JSON string as a plain response
            return app.response_class(response=safe_json_text, status=200, mimetype="application/json")


# -------------------------
# Logs & health endpoints
@app.route("/api/v1/logs", methods=["GET"])
def get_logs():
    request_id = request.args.get("request_id")
    db = SessionLocal()
    try:
        if request_id:
            rec = db.query(Audit).filter(Audit.request_id == request_id).first()
            if not rec:
                return jsonify({"error": "request_id not found"}), 404
            extra_obj = None
            try:
                extra_obj = json.loads(rec.extra) if rec.extra else None
            except Exception:
                extra_obj = rec.extra
            return jsonify({
                "request_id": rec.request_id,
                "path": rec.path,
                "method": rec.method,
                "request_body": rec.request_body,
                "response_snippet": rec.response_snippet,
                "duration_ms": rec.duration_ms,
                "timestamp": rec.timestamp.isoformat(),
                "extra": extra_obj
            }), 200

        rows = db.query(Audit).order_by(Audit.id.desc()).limit(20).all()
        result = []
        for r in rows:
            result.append({
                "request_id": r.request_id,
                "path": r.path,
                "method": r.method,
                "duration_ms": r.duration_ms,
                "timestamp": r.timestamp.isoformat()
            })
        return jsonify(result[::-1]), 200
    finally:
        db.close()

@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else None
    }), 200

# -------------------------
# Improved Fruit-status interpreter (better label matching)
def interpret_fruit_status(audit):
    if not audit:
        return "unknown", "No audit information available to determine fruit status.", {}
    try:
        if isinstance(audit, str):
            audit = json.loads(audit)
    except Exception:
        pass

    detections = audit.get("detections") or []
    probs = audit.get("probabilities") or {}
    labels = [((d.get("label") or "").lower(), d.get("confidence", 0.0)) for d in detections]
    counts = {}
    for lbl, conf in labels:
        counts[lbl] = counts.get(lbl, 0) + 1

    # expanded keywords & heuristic rules (catch "bad_", "bad", numeric day codes)
    keywords = {
        "ripe": ["ripe", "ripen", "ripe_"],
        "unripe": ["unripe", "raw", "green"],
        "rotten": ["rot", "rotten", "mold", "spoiled", "decay", "overripe", "bad", "bad_"],
        "fresh": ["fresh", "good", "healthy"],
        "damaged": ["bruise", "blemish", "damage", "scratch", "wormhole", "hole", "pest"]
    }

    status_scores = {k: 0.0 for k in keywords}
    top_label = None
    top_conf = 0.0

    for lbl, conf in labels:
        if conf > top_conf:
            top_conf = conf
            top_label = lbl
        for st, keys in keywords.items():
            for kw in keys:
                if kw in lbl:
                    status_scores[st] += conf

    # additional heuristic: labels that contain 'bad' or 'bad_' strongly suggest rotten
    for lbl, conf in labels:
        if "bad_" in lbl or lbl.startswith("bad") or ("bad" in lbl and re.search(r"\d", lbl)):
            status_scores["rotten"] += conf * 0.8

    if any(v > 0 for v in status_scores.values()):
        chosen = max(status_scores.items(), key=lambda x: x[1])[0]
        reason = f"Detected labels suggest '{chosen}' (score breakdown: {status_scores}). Top label: {top_label} ({round(top_conf,4)})"
        return chosen, reason, {"top_label": top_label, "top_confidence": round(top_conf,4), "counts": counts, "status_scores": status_scores, "detections": detections}

    if top_label:
        if any(w in top_label for w in ["ripe", "ripen"]):
            return "ripe", f"Top label '{top_label}' indicates ripeness (confidence {round(top_conf,4)}).", {"top_label": top_label, "top_confidence": round(top_conf,4), "detections": detections}
        if any(w in top_label for w in ["unripe", "green", "raw"]):
            return "unripe", f"Top label '{top_label}' indicates unripe (confidence {round(top_conf,4)}).", {"top_label": top_label, "top_confidence": round(top_conf,4), "detections": detections}
        if any(w in top_label for w in ["rot", "rotten", "mold", "spoiled", "overripe", "bad"]):
            return "rotten", f"Top label '{top_label}' indicates rotting/poor quality (confidence {round(top_conf,4)}).", {"top_label": top_label, "top_confidence": round(top_conf,4), "detections": detections}
        if any(w in top_label for w in ["bruise", "blemish", "damage"]):
            return "damaged", f"Top label '{top_label}' indicates damage (confidence {round(top_conf,4)}).", {"top_label": top_label, "top_confidence": round(top_conf,4), "detections": detections}

    if probs:
        top = max(probs.items(), key=lambda x: x[1])
        label_name, label_conf = top[0], float(top[1])
        if label_conf >= 0.7:
            return "unknown", f"Detected '{label_name}' with high confidence {round(label_conf,4)}, but label doesn't map to ripe/unripe/rotten directly.", {"top_label": label_name, "top_confidence": round(label_conf,4), "detections": detections}
        else:
            return "unknown", f"Model detected '{label_name}' with low confidence ({round(label_conf,4)}). Unable to determine fruit quality.", {"top_label": label_name, "top_confidence": round(label_conf,4), "detections": detections}

    return "unknown", "No detection results to analyze.", {}

# -------------------------
# NLP: semantic examples and lazy model
INTENT_EXAMPLES = {
    "fruit_status": ["Is this fruit ripe?", "Is the fruit ready to eat?", "What's the quality of this fruit?"],
    "ripeness_recommendation": ["How should I store this to ripen?", "When will it ripen?"],
    "storage_advice": ["How should I store these fruits to last longer?", "Can I refrigerate bananas?"],
    "detect_damage": ["Is the fruit damaged or bruised?", "Are there signs of pest or physical damage?"],
    "shelf_life_estimate": ["How long will these fruits last in storage?"],
    "sorting_advice": ["Which fruits should I discard?", "How to sort fruits by quality?"],
    "post_harvest_handling": ["What post-harvest care is recommended?"],
    "visual_cues": ["What are visual signs of ripeness?", "What color indicates overripeness?"],
    "quality_thresholds": ["What confidence level is acceptable for quality detection?"],
    "request_id_lookup": ["Show request_id", "Show the audit for request_id"],
    "object_count": ["How many objects were detected?", "How many fruits detected?", "How many items are in the image?"],
    "help": ["What can you do?", "help", "commands"],
    "greeting": ["hello", "hi", "hey"]
}

_example_texts = []
_example_mapping = []
for k, exs in INTENT_EXAMPLES.items():
    for ex in exs:
        _example_texts.append(ex)
        _example_mapping.append(k)

_embedding_model = None
_example_emb = None

def ensure_embedding_model():
    global _embedding_model, _example_emb
    if not EMBEDDING_AVAILABLE:
        return False
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        _example_emb = _embedding_model.encode(_example_texts, convert_to_tensor=True)
    return True

def parse_intent_nlp(text):
    # try embedding-match first
    try:
        if ensure_embedding_model():
            emb = _embedding_model.encode(text, convert_to_tensor=True)
            hits = util.semantic_search(emb, _example_emb, top_k=1)[0]
            best = hits[0]
            best_idx = int(best["corpus_id"])
            score = float(best["score"])
            intent = _example_mapping[best_idx]
            if score >= 0.48:
                return {"intent": intent, "params": {}, "score": score}
    except Exception:
        pass
    # fallback rule-based (kept simple)
    t = (text or "").strip().lower()
    if t in ("hi", "hello", "hey", "hey bot"):
        return {"intent": "greeting", "params": {}}
    if re.search(r"\bhelp\b|\bcommands\b|\bwhat can you do\b", t):
        return {"intent": "help", "params": {}}
    if re.search(r"\bhealth\b|\bstatus\b|\bmodel\b|\bup\b|\bserve\b", t) and "fruit" not in t:
        return {"intent": "health", "params": {}}
    if re.search(r"\blast\b|\bmost recent\b|\brecent prediction\b|\blast prediction\b", t) and "fruit" not in t:
        return {"intent": "last_prediction", "params": {}}
    if re.search(r"\blogs\b|\brecords\b|\bhistory\b|\brequests\b", t):
        return {"intent": "list_logs", "params": {}}
    m = re.search(r"(show|get|explain)\s+(request_id|id|request)\s*[:= ]*\s*([0-9a-fA-F\-]{8,})", t)
    if m:
        return {"intent": "show_request", "params": {"request_id": m.group(3)}}
    if re.search(r"\b(fruit|apple|banana|mango|orange|guava|pineapple|papaya)\b", t) or re.search(r"\b(ripe|unripe|rotten|fresh|damaged|bruise|blemish|overripe|spoiled)\b", t):
        m2 = re.search(r"(request_id|id|request)\s*[:= ]*\s*([0-9a-fA-F\-]{8,})", t)
        params = {}
        if m2:
            params["request_id"] = m2.group(2)
        return {"intent": "fruit_status", "params": params}
    if re.search(r"\bpredict\b|\bupload\b|\banalyze image\b|\bdetect\b|\brun prediction\b", t):
        return {"intent": "predict", "params": {}}
    if re.search(r"\bhow many\b", t):
        return {"intent": "object_count", "params": {}}

    return {"intent": "unknown", "params": {}}

def format_audit_summary(aud):
    if not aud:
        return "No audit information available."
    try:
        if isinstance(aud, str):
            aud = json.loads(aud)
    except Exception:
        pass
    req_id = aud.get("request_id", "unknown")
    model_ver = aud.get("model_version", "unknown")
    detections = aud.get("detections", [])
    conf = aud.get("confidence", aud.get("calibrated_confidence", 0.0))
    t_ms = aud.get("processing_ms", aud.get("duration_ms", None))
    snippet = ", ".join([f'{d.get("label","?")}({d.get("confidence",0)})' for d in (detections or [])[:5]]) if detections else "none"
    parts = [f"request_id: {req_id}", f"model: {model_ver}", f"detections: {snippet}", f"confidence: {conf}"]
    if t_ms:
        parts.append(f"processing_ms: {t_ms}")
    return " | ".join(parts)

def answer_question_nlp(intent, params, audit=None, question_text=None):
    # return plain-English reply and structured data (not printed by UI by default)
    if audit:
        try:
            if isinstance(audit, str):
                aud_obj = json.loads(audit)
            else:
                aud_obj = audit
        except Exception:
            aud_obj = audit
    else:
        aud_obj = None

    if intent == "fruit_status":
        if aud_obj:
            status, explanation, structured = interpret_fruit_status(aud_obj)
            # friendly English replies
            if status == "ripe":
                return {"reply": "The fruit appears ripe. It looks ready to eat or sell now.", "data": {"status": status, "explanation": explanation, "structured": structured}}
            if status == "unripe":
                return {"reply": "The fruit appears unripe. Store at room temperature to ripen.", "data": {"status": status, "explanation": explanation, "structured": structured}}
            if status == "rotten":
                return {"reply": "The fruit appears spoiled/rotten — do not eat. Remove and discard affected fruit.", "data": {"status": status, "explanation": explanation, "structured": structured}}
            if status == "damaged":
                return {"reply": "The fruit has visible damage or bruising. Inspect and sort damaged pieces.", "data": {"status": status, "explanation": explanation, "structured": structured}}
            return {"reply": "I could not determine a clear quality from the image. The model saw: " + explanation, "data": {"status": status, "explanation": explanation, "structured": structured}}
        else:
            return {"reply": "I don't have a recent image to analyze. Upload an image or provide a request_id to analyze a prior prediction.", "data": {}}

    if intent == "ripeness_recommendation":
        if aud_obj:
            status, explanation, structured = interpret_fruit_status(aud_obj)
            if status == "unripe":
                return {"reply": "Store at room temperature (18–25°C) to ripen. Place near a ripe apple/banana to speed ripening.", "data": {}}
            if status == "ripe":
                return {"reply": "This fruit appears ripe — best to consume or sell now. Refrigerate if you want to extend shelf life.", "data": {}}
            if status == "rotten":
                return {"reply": "This fruit is rotten/spoiled — discard it and inspect the batch.", "data": {}}
            return {"reply": "Unable to provide a specific recommendation from the image; try a clearer photo.", "data": {}}
        else:
            return {"reply": "General ripening advice: keep fruits like bananas or mangoes at room temperature to ripen. Refrigerate ripe fruits to slow decay.", "data": {}}

    if intent == "storage_advice":
        return {"reply": "General storage tips: refrigerate apples, pears, and berries; keep bananas and mangoes at room temperature until ripe; keep fruits dry and avoid stacking to reduce bruising.", "data": {}}

    if intent == "detect_damage":
        if aud_obj:
            dets = aud_obj.get("detections") if isinstance(aud_obj, dict) else None
            if dets:
                damaged = [d for d in dets if any(w in (d.get("label","").lower()) for w in ["bruise","blemish","damage","worm","hole","scratch","bad"])]
                if damaged:
                    return {"reply": f"I see signs of damage or spoilage (example: {damaged[0]['label']} with confidence {damaged[0]['confidence']}). Sort or discard damaged items.", "data": {"damages": damaged}}
            return {"reply": "No obvious damage labels were detected in the latest audit, but inspect visually to be sure.", "data": {}}
        else:
            return {"reply": "To detect damage, provide a clear close-up image; look for dark patches, soft spots, holes, or mold.", "data": {}}

    if intent == "shelf_life_estimate":
        return {"reply": "Shelf life varies. Example: berries 2–7 days refrigerated; apples 1–2 months refrigerated; bananas 2–7 days depending on ripeness.", "data": {}}

    if intent == "sorting_advice":
        return {"reply": "Sort by quality: discard rotten, separate bruised/damaged, group ripe for quick sale and unripe to ripen further.", "data": {}}

    if intent == "post_harvest_handling":
        return {"reply": "Handle gently, sort quickly, dry wet fruit, cool rapidly if possible, and avoid heavy stacking to reduce bruising.", "data": {}}

    if intent == "visual_cues":
        return {"reply": "Visual cues: look for color change, softness near the stem, stronger aroma for ripeness; mold, dark wet spots, and collapse indicate spoilage.", "data": {}}

    if intent == "quality_thresholds":
        return {"reply": "Rule of thumb: confidence >= 0.7 is strong; 0.4–0.7 is uncertain and needs manual check; <0.4 is unreliable.", "data": {}}

    if intent == "request_id_lookup":
        rid = params.get("request_id")
        if not rid:
            return {"reply": "Please provide a request_id to look up (e.g., 'show request_id 1234abcd').", "data": {}}
        aud = AUDIT_STORE.get(rid)
        if not aud:
            try:
                db = SessionLocal()
                rec = db.query(Audit).filter(Audit.request_id == rid).first()
                db.close()
                if rec:
                    try:
                        aud = json.loads(rec.extra) if rec.extra else None
                    except Exception:
                        aud = {"request_id": rec.request_id, "response_snippet": rec.response_snippet}
            except Exception:
                aud = None
        if not aud:
            return {"reply": f"request_id {rid} not found.", "data": {}}
        # Return a short English summary
        status, explanation, _ = interpret_fruit_status(aud)
        return {"reply": f"Summary for {rid}: {status.upper()}. {explanation}", "data": aud}

    if intent == "help":
        return {"reply": "Ask about fruit quality, ripeness, storage, damage detection, or upload an image. Example: 'Is this fruit ripe?' or 'How should I store mangoes?'", "data": {}}

    if intent == "greeting":
        return {"reply": "Hello! Upload an image to analyze fruit quality or ask questions about ripeness and storage.", "data": {}}

    # return plain-English reply and structured data (not printed by UI by default)
    if audit:
        try:
            if isinstance(audit, str):
                aud_obj = json.loads(audit)
            else:
                aud_obj = audit
        except Exception:
            aud_obj = audit
    else:
        aud_obj = None

    if intent == "fruit_status":
        if aud_obj:
            status, explanation, structured = interpret_fruit_status(aud_obj)
            # friendly concise English replies (no long internal explanation in reply)
            if status == "ripe":
                return {"reply": "The fruit appears ripe. It looks ready to eat or sell now.", "data": {"status": status, "explanation": explanation, "structured": structured}}
            if status == "unripe":
                return {"reply": "The fruit appears unripe. Store at room temperature to ripen.", "data": {"status": status, "explanation": explanation, "structured": structured}}
            if status == "rotten":
                return {"reply": "The fruit appears spoiled/rotten — do not eat. Remove and discard affected fruit.", "data": {"status": status, "explanation": explanation, "structured": structured}}
            if status == "damaged":
                return {"reply": "The fruit has visible damage or bruising. Inspect and sort damaged pieces.", "data": {"status": status, "explanation": explanation, "structured": structured}}
            # fallback concise response if status unknown
            return {"reply": "I could not determine a clear quality from the image. Try a closer photo or provide the request_id.", "data": {"status": status, "explanation": explanation, "structured": structured}}
        else:
            return {"reply": "I don't have a recent image to analyze. Upload an image or provide a request_id to analyze a prior prediction.", "data": {}}
    if intent == "object_count":
        if aud_obj:
            dets = aud_obj.get("detections") or []
            if dets:
                top = max(dets, key=lambda d: d.get("confidence", 0))
                top_label = top.get("label", "unknown")
                top_conf = top.get("confidence", 0)
                return {"reply": f"I detected {len(dets)} object(s). Top detection: {top_label} (confidence {round(top_conf,3)}).", "data": {"count": len(dets), "top": {"label": top_label, "confidence": top_conf}}}
            else:
                return {"reply": "No objects were detected in the latest image.", "data": {"count": 0}}
        else:
            return {"reply": "No recent image found. Upload an image or include a request_id to check its detections.", "data": {}}


    return {"reply": "Sorry — I didn't fully understand. Try: 'Is this fruit ripe?' or 'Show request_id <id>'.", "data": {}}

# -------------------------
# Chat API (multipart for images + JSON for text)
@app.route("/api/v1/chat", methods=["POST"])
def chat_api():
    # image branch
    if request.content_type and request.content_type.startswith("multipart/form-data") and 'image' in request.files:
        file = request.files['image']
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        filename = unique_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)

        user_msg = (request.form.get("message") or "").strip()

        t0 = time.time()
        model = get_model()
        if model:
            results = model.predict(file_path, save=True)
        else:
            results = []
        processing_ms = int((time.time() - t0) * 1000)

        detections, probs = parse_results_boxes(results, filename)
        raw_conf = max(probs.values()) if probs else 0.0
        calibrated = calibrate_confidence(raw_conf)
        quality_score = compute_quality_score(probs)

        request_id = str(uuid.uuid4())
        heatmap_url = generate_heatmap_stub(file_path, HEATMAP_DIR)

        # find annotated image path (if saved)
        result_image_path = None
        try:
            output_dir = results[0].save_dir if (results and hasattr(results[0], "save_dir")) else OUTPUT_BASE
            candidate = os.path.join(output_dir, filename)
            if not os.path.exists(candidate):
                for root, _, files in os.walk(output_dir):
                    for f in files:
                        if f == filename:
                            candidate = os.path.join(root, f)
                            break
                    if os.path.exists(candidate):
                        break
            if os.path.exists(candidate):
                result_image_path = os.path.abspath(candidate)
        except Exception:
            result_image_path = None

        audit = {
            "request_id": request_id,
            "type": "chat_image_predict",
            "user_message": user_msg,
            "uploaded_filename": filename,
            "detections": detections,
            "probabilities": probs,
            "confidence": round(raw_conf,4),
            "calibrated_confidence": calibrated,
            "quality_score": quality_score,
            "model_version": MODEL_VERSION,
            "processing_ms": processing_ms,
            "timestamp": now_ts()
        }
        append_audit(audit)

        if detections:
            top = max(detections, key=lambda d: d.get("confidence",0))
            reply_text = f"I detected {len(detections)} object(s). Top detection: {top['label']} ({top['confidence']}). Request id: {request_id}"
        else:
            reply_text = f"No objects detected. Request id: {request_id}"

        data = {
            "request_id": request_id,
            "detections": detections,
            "probabilities": {k: round(v,4) for k,v in probs.items()},
            "confidence": round(raw_conf,4),
            "calibrated_confidence": calibrated,
            "quality_score": quality_score,
            "result_image": (result_image_path if result_image_path else None)
        }
        return jsonify({"reply": reply_text, "intent": "image_prediction", "data": data}), 200

    # text branch (JSON)
    body = request.get_json(silent=True) or {}
    text = body.get("message", "")
    if not text:
        return jsonify({"error": "No message provided."}), 400

    parsed = parse_intent_nlp(text)
    intent = parsed.get("intent")
    params = parsed.get("params", {})

    # intents handled by NLP answerer
    nlp_intents = {"fruit_status","ripeness_recommendation","storage_advice","detect_damage",
                   "shelf_life_estimate","sorting_advice","post_harvest_handling","visual_cues",
                   "quality_thresholds","request_id_lookup","help","greeting"}

    if intent in nlp_intents:
        # determine which audit to use: provided request_id -> last in AUDIT_STORE -> latest DB record
        rid = params.get("request_id")
        audit_for_answer = None

        if rid:
            audit_for_answer = AUDIT_STORE.get(rid)
            if not audit_for_answer:
                try:
                    db = SessionLocal()
                    rec = db.query(Audit).filter(Audit.request_id == rid).first()
                    db.close()
                    if rec:
                        try:
                            audit_for_answer = json.loads(rec.extra) if rec.extra else None
                        except Exception:
                            audit_for_answer = {"request_id": rec.request_id, "response_snippet": rec.response_snippet}
                except Exception:
                    audit_for_answer = None
        else:
            # no request_id provided -> try last in AUDIT_STORE or query DB
            try:
                if AUDIT_STORE:
                    audit_for_answer = list(AUDIT_STORE.values())[-1]
                else:
                    db = SessionLocal()
                    rec = db.query(Audit).order_by(Audit.id.desc()).first()
                    db.close()
                    if rec:
                        try:
                            audit_for_answer = json.loads(rec.extra) if rec.extra else None
                        except Exception:
                            audit_for_answer = {"request_id": rec.request_id, "response_snippet": rec.response_snippet}
            except Exception:
                audit_for_answer = None

        ans = answer_question_nlp(intent, params, audit=audit_for_answer, question_text=text)
        # return plain english reply and structured data (UI will show friendly text only)
        return jsonify({"reply": ans["reply"], "intent": intent, "data": ans.get("data", {})}), 200

    # fallback handlers (health, logs, show_request, last_prediction, predict)
    if intent == "health":
        health_resp = {"status": "ok", "model_version": MODEL_VERSION, "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else None}
        return jsonify({"reply": f"Model status: {health_resp['status']}. Version: {health_resp['model_version']}.", "intent": intent, "data": health_resp})

    if intent == "list_logs":
        try:
            db = SessionLocal()
            rows = db.query(Audit).order_by(Audit.id.desc()).limit(10).all()
            db.close()
            items = [{"request_id": r.request_id, "path": r.path, "duration_ms": r.duration_ms, "timestamp": r.timestamp.isoformat()} for r in rows]
            if not items:
                return jsonify({"reply":"No logs found.", "intent": intent, "data": []})
            reply = "Recent request_ids: " + ", ".join([i["request_id"] for i in items])
            return jsonify({"reply": reply, "intent": intent, "data": items})
        except Exception as e:
            app.logger.exception("list_logs error")
            return jsonify({"reply": f"Failed to fetch logs: {e}", "intent": intent}), 500

    if intent == "show_request":
        rid = params.get("request_id")
        if not rid:
            return jsonify({"reply":"Please provide a request_id.", "intent": intent}), 400
        try:
            aud = AUDIT_STORE.get(rid)
            if not aud:
                db = SessionLocal()
                rec = db.query(Audit).filter(Audit.request_id == rid).first()
                db.close()
                if rec:
                    try:
                        aud = json.loads(rec.extra) if rec.extra else None
                    except Exception:
                        aud = {"request_id": rec.request_id, "response_snippet": rec.response_snippet}
            if not aud:
                return jsonify({"reply": f"request_id {rid} not found.", "intent": intent}), 404
            # return human-friendly summary
            status, explanation, _ = interpret_fruit_status(aud)
            return jsonify({"reply": f"Summary for {rid}: {status.upper()}. {explanation}", "intent": intent, "data": aud})
        except Exception as e:
            app.logger.exception("show_request error")
            return jsonify({"reply": f"Error retrieving request: {e}", "intent": intent}), 500

    if intent == "last_prediction":
        try:
            db = SessionLocal()
            rec = db.query(Audit).order_by(Audit.id.desc()).first()
            db.close()
            aud = None
            if rec:
                try:
                    aud = json.loads(rec.extra) if rec.extra else None
                except Exception:
                    aud = {"request_id": rec.request_id, "response_snippet": rec.response_snippet}
            if not aud and AUDIT_STORE:
                aud = list(AUDIT_STORE.values())[-1]
            if not aud:
                return jsonify({"reply": "No prediction audits found yet.", "intent": intent}), 404
            status, explanation, _ = interpret_fruit_status(aud)
            return jsonify({"reply": f"Latest prediction — {status.upper()}. {explanation}", "intent": intent, "data": aud})
        except Exception as e:
            app.logger.exception("last_prediction error")
            return jsonify({"reply": f"Error retrieving last prediction: {e}", "intent": intent}), 500

    if intent == "predict":
        return jsonify({"reply":"To run a prediction, attach an image in the chat or use /api/v1/predict (multipart image).", "intent": intent})

    return jsonify({"reply":"Sorry — I didn't understand. Say 'help' for supported commands.", "intent":"unknown"})

# -------------------------
# Chat UI (updated to show friendly replies, not raw JSON)
@app.route("/chatbot")
def chatbot_page():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Model Assistant - Chat</title>
      <style>
        body { font-family: Arial, sans-serif; background:#f6f8fa; margin:0; padding:0; }
        .wrap { max-width:900px; margin:20px auto; background:white; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08); overflow:hidden; }
        header { padding:16px 20px; background:#24292f; color:white; }
        #chatbox { height:520px; overflow:auto; padding:20px; background: #fff; }
        .composer { display:flex; padding:12px; border-top:1px solid #eee; gap:8px; align-items:center; }
        .composer input[type="text"]{ flex:1; padding:10px 12px; border:1px solid #ddd; border-radius:6px; }
        .composer button { padding:10px 14px; border-radius:6px; border:none; background:#0366d6; color:white; cursor:pointer; }
        .composer label { background:#fff; border:1px dashed #ccc; padding:8px 10px; border-radius:6px; cursor:pointer; display:inline-block; }
        .msg { margin:8px 0; display:flex; }
        .msg.user { justify-content:flex-end; }
        .bubble { max-width:72%; padding:10px 12px; border-radius:12px; }
        .bubble.user { background:#daf1ff; color:#03396c; border-bottom-right-radius:4px; }
        .bubble.bot { background:#f1f3f5; color:#111; border-bottom-left-radius:4px; }
        .meta { font-size:11px; color:#666; margin-top:6px; }
        img.res { max-width:360px; border-radius:6px; display:block; margin-top:8px; }
        .small { font-size:13px; color:#333; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <header><strong>Model Assistant</strong> — Chat & Image upload</header>
        <div id="chatbox"></div>
        <div class="composer">
          <input id="textmsg" type="text" placeholder="Ask about fruit status, health, or attach image..." />
          <label>
            Attach
            <input id="fileinput" type="file" accept="image/*" style="display:none" />
          </label>
          <button id="sendbtn">Send</button>
        </div>
      </div>

      <script>
        const chatbox = document.getElementById('chatbox');
        const textmsg = document.getElementById('textmsg');
        const fileinput = document.getElementById('fileinput');
        const sendbtn = document.getElementById('sendbtn');

        function appendMessage(who, text, extraHtml) {
          const wrap = document.createElement('div');
          wrap.className = 'msg ' + (who === 'user' ? 'user' : 'bot');
          const bub = document.createElement('div');
          bub.className = 'bubble ' + (who === 'user' ? 'user' : 'bot');
          bub.textContent = text;
          wrap.appendChild(bub);
          if (extraHtml) {
            const div = document.createElement('div');
            div.innerHTML = extraHtml;
            wrap.appendChild(div);
          }
          chatbox.appendChild(wrap);
          chatbox.scrollTop = chatbox.scrollHeight;
        }

        async function sendText() {
          const text = textmsg.value.trim();
          if (!text) return;
          appendMessage('user', text);
          textmsg.value = '';
          appendMessage('bot', '...thinking');
          try {
            const res = await fetch('/api/v1/chat', {
              method: 'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify({message: text})
            });
            const j = await res.json();
            const last = chatbox.querySelector('.msg.bot:last-child');
            if (last && last.textContent.includes('...thinking')) last.remove();
            if (j.reply) {
              // friendly reply only
              let extraHtml = '';
              if (j.data) {
                if (j.data.request_id) extraHtml += '<div class="small"><strong>Request ID:</strong> ' + j.data.request_id + '</div>';
                if (j.data.detections && j.data.detections.length) {
                  extraHtml += '<div class="small"><strong>Detections:</strong></div><ul>';
                  j.data.detections.forEach(d => {
                    extraHtml += '<li>' + d.label + ' (' + d.confidence + ')</li>';
                  });
                  extraHtml += '</ul>';
                }
                if (j.data.result_image) {
                  extraHtml += '<div><img class="res" src="/result/' + encodeURIComponent(j.data.result_image) + '" /></div>';
                }
              }
              appendMessage('bot', j.reply, extraHtml);
            } else {
              appendMessage('bot', 'No response from server');
            }
          } catch (e) {
            appendMessage('bot', 'Request failed: ' + e.toString());
          }
        }

        async function sendImage(file) {
          appendMessage('user', 'Sent an image: ' + file.name);
          appendMessage('bot', 'Uploading and analyzing image...');
          const fd = new FormData();
          fd.append('image', file);
          const text = textmsg.value.trim();
          if (text) fd.append('message', text);
          textmsg.value = '';

          try {
            const res = await fetch('/api/v1/chat', {
              method: 'POST',
              body: fd
            });
            const j = await res.json();
            const last = chatbox.querySelector('.msg.bot:last-child');
            if (last && last.textContent.includes('Uploading')) last.remove();
            if (j.reply) {
              let extraHtml = '';
              if (j.data) {
                if (j.data.request_id) extraHtml += '<div class="small"><strong>Request ID:</strong> ' + (j.data.request_id||'') + '</div>';
                if (j.data.detections && j.data.detections.length) {
                  extraHtml += '<div class="small"><strong>Detections:</strong></div><ul>';
                  j.data.detections.forEach(d => {
                    extraHtml += '<li>' + d.label + ' (' + d.confidence + ')</li>';
                  });
                  extraHtml += '</ul>';
                } else {
                  extraHtml += '<div class="small">No detections</div>';
                }
                if (j.data.result_image) {
                  extraHtml += '<div><img class="res" src="/result/' + encodeURIComponent(j.data.result_image) + '" /></div>';
                }
              }
              appendMessage('bot', j.reply, extraHtml);
            } else {
              appendMessage('bot', 'No response for image upload');
            }
          } catch (e) {
            appendMessage('bot', 'Upload failed: ' + e.toString());
          }
        }

        sendbtn.addEventListener('click', async () => {
          if (fileinput.files && fileinput.files.length) {
            await sendImage(fileinput.files[0]);
            fileinput.value = '';
          } else {
            await sendText();
          }
        });

        textmsg.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            sendbtn.click();
          }
        });
      </script>
    </body>
    </html>
    """
    return html

# -------------------------
# Serve result image with safe path
@app.route("/result/<path:filename>")
def result_file(filename):
    safe_path = resolve_safe_path(filename)
    if not safe_path or not os.path.exists(safe_path):
        return "File not found or access denied", 404
    try:
        return send_file(safe_path)
    except Exception as e:
        app.logger.exception("Failed to send file: %s", e)
        return "Failed to send file", 500

# -------------------------
# Run server
if __name__ == "__main__":
    # For production use a WSGI server (gunicorn/uvicorn) and do not use debug=True
    app.run(host="0.0.0.0", port=5000, debug=False)
