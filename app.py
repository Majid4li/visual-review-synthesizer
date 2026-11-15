"""Visual Review Synthesizer: DIP (OpenCV) + NLP (Roman Urdu/English keywords) + Streamlit."""
import os
import re

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from skimage.color import rgb2lab

st.set_page_config(page_title="Visual Review Synthesizer", page_icon="🛍️", layout="wide")

# ============================== NLP ==============================
SIZE_SMALL = {"chota", "chhota", "choti", "chhoti", "small", "tight", "tang", "short"}
SIZE_BIG = {"bara", "bada", "barra", "badi", "large", "loose", "dheela", "dhila", "lamba", "baggy"}
TOPICS = {
    "Size": {"size", "fit", "fitting", "naap"} | SIZE_SMALL | SIZE_BIG,
    "Color": {"rang", "color", "colour", "shade", "fade", "faded", "dull", "pheeka"},
    "Stitching": {"stitching", "stitched", "silai", "dhaga", "thread", "seam", "ulta", "udhar", "udhri"},
    "Quality": {"quality", "kapra", "kapray", "fabric", "material", "cloth", "hole", "torn",
                "phata", "phati", "wrinkle", "wrinkled"},
    "Delivery": {"delivery", "late", "deri", "delay", "delayed", "packing", "parcel", "courier", "rider"},
}
NEG_CUES = SIZE_SMALL | SIZE_BIG | {
    "kharab", "kharaab", "bakwas", "ghatiya", "bekar", "faltu", "bad", "poor", "worst", "cheap",
    "fraud", "fake", "terrible", "horrible", "damaged", "torn", "phata", "phati", "hole", "fade",
    "faded", "dull", "pheeka", "alag", "different", "mismatch", "wrong", "galat", "late", "deri",
    "delay", "delayed", "disappointed", "wrinkled", "nikal", "nikla", "udhar", "udhri"}
POS_WORDS = {"acha", "achi", "achha", "achhi", "ache", "good", "great", "perfect", "best", "love",
             "nice", "awesome", "zabardast", "behtareen", "mast", "shandar", "amazing", "excellent",
             "pasand", "happy", "khubsurat"}
NEGATORS = {"nahi", "nahin", "nhi", "not", "no", "never"}

# --- test mein ghalat nikalne ke baad barhaye gaye alfaaz ---
TOPICS["Quality"] |= {"daagh", "dagh", "stain", "stained", "silwat", "silwatein", "patla", "patli", "thin", "rough"}
TOPICS["Stitching"] |= {"khul", "khula", "khuli", "udhad"}
TOPICS["Color"] |= {"utar", "utra", "utri"}
NEG_CUES |= {"daagh", "dagh", "stain", "stained", "silwat", "silwatein", "patla", "patli", "thin", "rough",
             "zaya", "gandi", "gandha", "khul", "khula", "khuli", "udhad", "kamzor", "kamzoor", "ulta",
             "ulti", "tedha", "tedhi", "utar", "utra", "utri"}
DELIVERY_NOUNS = {"delivery", "packing", "parcel", "courier", "rider", "dispatch", "shipping"}
QUALITY_NOUNS = {"quality", "kapra", "kapray", "fabric", "material", "cloth", "daagh", "dagh", "stain",
                 "stained", "silwat", "silwatein"}


def split_clauses(text):
    parts = re.split(r"[.,!?;\n]|\b(?:lekin|magar|par|but|aur|and)\b", str(text).lower())
    return [p for p in parts if p.strip()]


def tokens(clause):
    return re.findall(r"[a-z\u0600-\u06ff]+", clause)


def negated(tk, i):
    return any(t in NEGATORS for t in tk[max(0, i - 2):i] + tk[i + 1:i + 3])


def analyze_text(text):
    complaints, size_dirs, neg, pos = set(), [], 0, 0
    for clause in split_clauses(text):
        tk = tokens(clause)
        bad = False
        for i, t in enumerate(tk):
            n = negated(tk, i)
            if t in NEG_CUES:
                if n:
                    pos += 1
                else:
                    neg += 1
                    bad = True
            elif t in POS_WORDS:
                if n:
                    neg += 1
                    bad = True
                else:
                    pos += 1
        if bad:
            ts = set(tk)
            found = {cat for cat, words in TOPICS.items() if words & ts}
            # "packing phati thi" Delivery ki shikayat hai, kapray ki quality ki nahi
            if "Delivery" in found and "Quality" in found and (DELIVERY_NOUNS & ts) and not (QUALITY_NOUNS & ts):
                found.discard("Quality")
            complaints |= found
            if ts & SIZE_SMALL:
                size_dirs.append("Chota")
            elif ts & SIZE_BIG:
                size_dirs.append("Bara")
    sentiment = "Negative" if neg > pos else "Positive" if pos > neg else "Neutral"
    if sentiment == "Negative" and not complaints:
        complaints.add("Other")
    size_issue = None
    if "Size" in complaints:
        size_issue = size_dirs[0] if size_dirs else "Unclear"
    return {"sentiment": sentiment, "complaints": sorted(complaints), "size_issue": size_issue}


# ============================== DIP ==============================
W = np.array([0.5, 1.0, 1.0], dtype=np.float32)  # L ka weight kam: shadows se false alarms kam hon


def wdist(a, b):
    return np.linalg.norm((a - b) * W, axis=-1)


def load_img(data):
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def prep(img, max_side=512):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def to_lab(img):
    lab = rgb2lab(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).astype(np.float32)
    return cv2.GaussianBlur(lab, (0, 0), 4)


def product_mask(img, use_gc):
    h, w = img.shape[:2]
    if use_gc:
        try:
            mask = np.zeros((h, w), np.uint8)
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
            cv2.grabCut(img, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
            if m.sum() / 255 > 0.1 * h * w:
                return m
        except cv2.error:
            pass
    m = np.zeros((h, w), np.uint8)
    m[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)] = 255  # fallback: beech ka hissa
    return m


CLASSES = ["clean", "stain", "wrinkle"]  # model ke class names ka order


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defect_model.onnx")
    return cv2.dnn.readNetFromONNX(path) if os.path.exists(path) else None


def classify_defect(img, net, size=384):
    """Short side ko 384 karo, beech se crop (training/validation jaisa), phir ONNX model chalao."""
    h, w = img.shape[:2]
    sc = size / min(h, w)
    img = cv2.resize(img, (max(size, round(w * sc)), max(size, round(h * sc))))
    h, w = img.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    blob = cv2.dnn.blobFromImage(img[y0:y0 + size, x0:x0 + size], 1 / 255.0, (size, size), swapRB=True)
    net.setInput(blob)
    p = net.forward().flatten().astype(np.float64)
    if not 0.99 < p.sum() < 1.01:  # agar softmax pehle se nahi laga
        e = np.exp(p - p.max())
        p = e / e.sum()
    return dict(zip(CLASSES, p))


def analyze_image(bgr, ref_lab, use_gc, wr_thr, st_thr, col_thr, net=None, conf_thr=0.5):
    img = prep(bgr)
    mask = product_mask(img, use_gc)
    m = mask > 0
    area = max(int(m.sum()), 1)
    lab = to_lab(img)
    dom = np.median(lab[m], axis=0)  # dominant color (LAB)

    stain = (wdist(lab, dom) > 15) & m  # color se hatay huay hisse = stain/blotch
    stain_ratio = float(stain.sum()) / area

    # wrinkle = roshni/saaye ki naram tabdeeli (folds). Sirf sharp edges (Canny) naram folds nahi pakarta.
    L = lab[:, :, 0]
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    wrinkle = float(grad[inner].mean()) if inner.any() else 0.0

    col_de = float(wdist(dom, ref_lab)) if ref_lab is not None else None

    flags, pred, conf = [], None, None
    if net is not None:  # trained model
        probs = classify_defect(img, net)
        pred = max(probs, key=probs.get)
        conf = float(probs[pred])
        if pred in ("stain", "wrinkle") and conf >= conf_thr:
            flags.append(pred.capitalize())
    else:  # rule-based baseline
        if wrinkle > wr_thr:
            flags.append("Wrinkle")
        if stain_ratio > st_thr:
            flags.append("Stain/Blotch")
    if col_de is not None and col_de > col_thr:
        flags.append("Color mismatch")

    ov = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if net is None:
        ov[stain] = (0.5 * ov[stain] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
    return {"flags": flags, "wrinkle": round(wrinkle, 3), "stain_pct": round(stain_ratio * 100, 1),
            "color_de": None if col_de is None else round(col_de, 1), "overlay": ov,
            "pred": pred, "conf": None if conf is None else round(conf, 2)}


# ============================== Helpers ==============================
def find_col(df, names):
    low = {str(c).lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def get_image_bytes(ref, files):
    ref = "" if pd.isna(ref) else str(ref).strip()
    if not ref:
        return None
    if ref in files:
        return files[ref]
    if ref.startswith("http"):
        try:
            r = requests.get(ref, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.content
        except Exception:
            return None
    return None


def make_summary(res):
    n = len(res)
    lines = [f"Total reviews: {n}",
             f"- {(res.sentiment == 'Negative').mean() * 100:.0f}% reviews negative hain."]
    cats = res.complaints.explode().dropna().value_counts()
    for cat, cnt in cats.items():
        lines.append(f"- {cnt / n * 100:.0f}% reviews mein **{cat}** ki shikayat hai.")
    size = res.size_issue.dropna()
    if len(size):
        small = (size == "Chota").mean() * 100
        lines.append(f"- Size ki shikayat karne walon mein se {small:.0f}% ne size **chota** bataya.")
    with_img = int(res.has_image.sum())
    if with_img:
        bad = int((res.image_flags.apply(len) > 0).sum())
        fl = res.image_flags.explode().dropna().value_counts()
        detail = ", ".join(f"{k}: {v}" for k, v in fl.items()) or "koi nahi"
        lines.append(f"- Tasveer wale {with_img} reviews mein se {bad} ({bad / with_img * 100:.0f}%) "
                     f"mein visible kharabi mili ({detail}).")
    return "\n".join(lines)


DEMO = [
    "Size bohat chota hai, tight aa raha hai", "Stitching bakwas hai, dhaga nikal gaya",
    "Quality achi hai, bohat pasand aaya", "Rang fade ho gaya pehli wash mein",
    "Delivery late thi lekin product achha hai", "Color picture se bilkul alag hai",
    "Size perfect hai, fitting zabardast", "Kapra ghatiya hai, hole tha",
    "Too small, size chart is wrong", "Loose fitting hai, size bara hai",
    "Not good quality, fabric is cheap", "Best product, love the color",
    "Parcel damaged tha aur rider ne deri ki", "Size chota hai aur stitching kharab hai",
]

# ============================== UI ==============================
st.title("🛍️ Visual Review Synthesizer")
st.caption("Review ka text (NLP) + tasveer (DIP) mila kar seller ke liye automatic report")

with st.sidebar:
    st.header("Settings")
    ref_file = st.file_uploader("Product listing ki original photo (color compare, optional)",
                                type=["jpg", "jpeg", "png"])
    net = load_model()
    if net is None:
        st.warning("defect_model.onnx nahi mili: rule-based tareeqa chal raha hai.")
    method = st.radio("Defect detection", ["Trained model", "Rule-based (baseline)"],
                      index=0 if net is not None else 1)
    conf_thr = st.slider("Model confidence threshold", 0.40, 0.95, 0.65, 0.05)
    use_gc = st.checkbox("GrabCut (background hatao)", True)
    wr_thr = st.slider("Wrinkle threshold (fold/shading score)", 0.2, 6.0, 1.2, 0.05)
    st_thr = st.slider("Stain area threshold (%)", 1, 30, 5) / 100
    col_thr = st.slider("Color mismatch threshold (ΔE)", 5, 40, 15)

use_net = net if (net is not None and method == "Trained model") else None

ref_lab = None
if ref_file is not None:
    rb = load_img(ref_file.getvalue())
    if rb is not None:
        ri = prep(rb)
        ref_lab = np.median(to_lab(ri)[product_mask(ri, use_gc) > 0], axis=0)

tab1, tab2 = st.tabs(["📊 Reviews analysis", "🖼️ Quick image test"])

with tab1:
    st.subheader("1) Data")
    c1, c2 = st.columns(2)
    csv = c1.file_uploader("Reviews CSV (columns: review, image)", type="csv")
    imgs = c2.file_uploader("Review images (CSV ke filenames wali)", type=["jpg", "jpeg", "png"],
                            accept_multiple_files=True)
    if st.button("Demo data use karo"):
        st.session_state["demo"] = True

    df = None
    if csv is not None:
        df = pd.read_csv(csv, encoding="utf-8-sig")
    elif st.session_state.get("demo"):
        df = pd.DataFrame({"review": DEMO, "image": [""] * len(DEMO)})

    if df is not None:
        tcol = find_col(df, ["review", "text", "comment", "review_text"])
        icol = find_col(df, ["image", "image_url", "img", "photo"])
        if tcol is None:
            st.error("CSV mein 'review' naam ka column chahiye.")
            st.stop()
        st.dataframe(df.head(10))

        if st.button("Analyze karo", type="primary"):
            files = {f.name: f.getvalue() for f in imgs}
            rows, overlays = [], {}
            bar = st.progress(0.0)
            for i, (_, r) in enumerate(df.iterrows()):
                nlp = analyze_text(r[tcol])
                flags, w, s, c, has_img, pred, conf = [], None, None, None, False, None, None
                if icol:
                    bgr = load_img(get_image_bytes(r[icol], files))
                    if bgr is not None:
                        has_img = True
                        out = analyze_image(bgr, ref_lab, use_gc, wr_thr, st_thr, col_thr, use_net, conf_thr)
                        flags, w, s, c, pred, conf = (out["flags"], out["wrinkle"], out["stain_pct"], out["color_de"],
                                              out["pred"], out["conf"])
                        overlays[i] = out["overlay"]
                rows.append({"review": r[tcol], **nlp, "has_image": has_img, "image_flags": flags,
                             "pred": pred, "model_conf": conf, "wrinkle": w, "stain_pct": s, "color_dE": c})
                bar.progress((i + 1) / len(df))
            st.session_state["res"] = pd.DataFrame(rows)
            st.session_state["overlays"] = overlays

    res = st.session_state.get("res")
    if res is not None:
        st.subheader("2) Seller Report")
        n = len(res)
        a, b, c, d = st.columns(4)
        a.metric("Total reviews", n)
        b.metric("Negative", f"{(res.sentiment == 'Negative').mean() * 100:.0f}%")
        c.metric("Photo wale reviews", int(res.has_image.sum()))
        d.metric("Visible defects", int((res.image_flags.apply(len) > 0).sum()))

        summary = make_summary(res)
        st.markdown(summary)

        g1, g2 = st.columns(2)
        cats = res.complaints.explode().dropna().value_counts()
        if len(cats):
            g1.plotly_chart(px.bar(x=cats.index, y=cats.values, title="Text complaints (NLP)",
                                   labels={"x": "Category", "y": "Reviews"}))
        fl = res.image_flags.explode().dropna().value_counts()
        if len(fl):
            g2.plotly_chart(px.bar(x=fl.index, y=fl.values, title="Image defects (DIP)",
                                   labels={"x": "Defect", "y": "Images"}))

        ov = st.session_state.get("overlays", {})
        flagged = [i for i in ov if len(res.loc[i, "image_flags"]) > 0][:8]
        if flagged:
            st.markdown("**Flagged images** (red = stain/blotch, green = product area)")
            cols = st.columns(4)
            for k, i in enumerate(flagged):
                cols[k % 4].image(ov[i], caption=", ".join(res.loc[i, "image_flags"]), width=220)

        st.dataframe(res.drop(columns=[]).astype({"has_image": bool}))
        out_df = res.assign(complaints=res.complaints.apply(", ".join),
                            image_flags=res.image_flags.apply(", ".join))
        d1, d2 = st.columns(2)
        d1.download_button("Report CSV download", out_df.to_csv(index=False).encode("utf-8-sig"),
                           "review_report.csv", "text/csv")
        d2.download_button("Summary TXT download", summary.replace("**", "").encode("utf-8"),
                           "summary.txt", "text/plain")

with tab2:
    one = st.file_uploader("Ek kapray ki photo test karein", type=["jpg", "jpeg", "png"], key="one")
    if one is not None:
        bgr = load_img(one.getvalue())
        if bgr is None:
            st.error("Image nahi khul saki.")
        else:
            out = analyze_image(bgr, ref_lab, use_gc, wr_thr, st_thr, col_thr, use_net, conf_thr)
            x1, x2 = st.columns(2)
            x1.image(cv2.cvtColor(prep(bgr), cv2.COLOR_BGR2RGB), caption="Original", width=350)
            x2.image(out["overlay"], caption="Red = stain/blotch, Green = product area", width=350)
            st.write(f"**Wrinkle score:** {out['wrinkle']}  |  **Stain area:** {out['stain_pct']}%  |  "
                     f"**Color ΔE:** {out['color_de'] if out['color_de'] is not None else 'N/A (reference photo nahi di)'}")
            if out["pred"]:
                st.write(f"**Model ka andaza:** {out['pred']} ({out['conf'] * 100:.0f}% yaqeen)")
                if out["pred"] != "clean" and out["conf"] < conf_thr:
                    st.caption("Model ko yaqeen kam hai, is liye kharabi flag nahi ki.")
            st.write("**Flags:** " + (", ".join(out["flags"]) or "koi kharabi nahi mili"))
