import re
import json
from dataclasses import dataclass
from datetime import datetime
from dateutil import tz
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple

# =========================
# CONFIG
# =========================
DEFAULT_TZ = tz.gettz("America/Montevideo")
STATE_PATH = "brokeros_state.json"

ZONE_DICT = {
    "aidy grill": "Aidy Grill",
    "roosevelt": "Roosevelt",
    "peninsula": "Península",
    "penisnula": "Península",
    "mansa": "Mansa",
    "brava": "Brava",
    "la barra": "La Barra",
    "manantiales": "Manantiales",
    "jose ignacio": "José Ignacio",
    "punta ballena": "Punta Ballena",
    "pinares": "Pinares",
    "pastora": "La Pastora",
    "la pastora": "La Pastora",
    "chiverta": "Chiverta",
    "lugano": "Lugano",
    "cantegril": "Cantegril",
    "ocean park": "Ocean Park",
    "sauce de portezuelo": "Sauce de Portezuelo",
    "piriapolis": "Piriápolis",
    "rincon del indio": "Rincón del Indio",
}

PROPERTY_TYPE_HINTS = [
    ("terreno", "Terreno"),
    ("campo", "Campo"),
    ("local comercial", "Local/Oficina"),
    ("oficina", "Local/Oficina"),
    ("galpón", "Depósito/Galpón"),
    ("deposito", "Depósito/Galpón"),
    ("depósito", "Depósito/Galpón"),
    ("casa", "Casa"),
    ("ph", "PH"),
    ("apartamento", "Apartamento"),
    ("apto", "Apartamento"),
    ("depto", "Apartamento"),
    ("penthouse", "Penthouse"),
]

BUY_HINTS = ["en venta", "para compra", "compra", "vendo", "ofrezco", "tenemos", "en exclusividad", "venta"]
RENT_HINTS = ["alquiler", "alquilo", "anual", "invernal", "temporal", "quincena", "mensuales", "mes", "alquilo", "alquiler"]

HEADER_RE = re.compile(
    r"^(\d{1,2})/(\d{1,2})/(\d{4}),\s+(\d{1,2}):(\d{2}).*?-\s*(.*?):\s*(.*)$",
    re.IGNORECASE
)

PHONE_RE = re.compile(r"(\+598)\s?(\d{2})\s?(\d{3})\s?(\d{3})")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

USD_RE = re.compile(r"(u\$s|usd|u\$d|\$us|d[oó]lares)", re.IGNORECASE)
UYU_RE = re.compile(r"(uyu|pesos|\$)", re.IGNORECASE)

NUM_RE = re.compile(r"(\d{1,3}(?:[.\s]\d{3})+|\d{2,7})(?:\s*(k|mil))?", re.IGNORECASE)
RANGE_RE = re.compile(r"(\d[\d.\s]{1,10})\s*[-/]\s*(\d[\d.\s]{1,10})", re.IGNORECASE)

DORM_RE = re.compile(r"(\d)\s*(?:dorm(?:itorios)?|dor\b|dormi\.?)", re.IGNORECASE)
M2_RE = re.compile(r"(\d{3,5})\s*m2", re.IGNORECASE)

SYSTEM_HINTS = ["cifrados de extremo a extremo", "creó el grupo", "se te añadió al grupo", "obten"]

KEYWORD_BUILDINGS = [
    "torres del este", "gala", "gala puerto", "gala vista", "gala tower",
    "tiburón ii", "green park", "green life", "le parc", "tequendama",
    "tunquelen", "millenium", "trump", "venetian", "ancora", "malecon",
    "miami boulevard", "casino miguez", "marigot", "signature", "citrea",
    "brava 28", "villa brava", "espacio 1"
]

FUNNEL_STATES = ["nuevo", "contactado", "respondió", "agendado", "visita", "reserva", "cierre", "perdido"]

# =========================
# STATE (simple persistence)
# =========================
def load_state() -> Dict[str, Any]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"overrides": {}, "funnel": {}, "notes": {}, "events": []}

def save_state(state: Dict[str, Any]) -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # en cloud a veces no persiste; seguimos igual

# =========================
# PARSER / EXTRACTION
# =========================
@dataclass
class Message:
    ts: datetime
    sender_raw: str
    text: str

def normalize_phone(s: str) -> Optional[str]:
    m = PHONE_RE.search(s or "")
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}"

def parse_ts(d, mo, y, hh, mm) -> datetime:
    return datetime(int(y), int(mo), int(d), int(hh), int(mm), tzinfo=DEFAULT_TZ)

def has_media(text: str) -> bool:
    t = (text or "").lower()
    return "<multimedia omitido>" in t or "(archivo adjunto)" in t or "archivo adjunto" in t

def classify_message(text: str) -> str:
    t = (text or "").lower().strip()
    if any(h in t for h in SYSTEM_HINTS):
        return "SYSTEM"
    if has_media(t):
        return "MEDIA"
    if any(k in t for k in ["busco", "estoy buscando", "búsqueda", "necesito", "preciso"]):
        return "LEAD_REQUEST"
    if any(k in t for k in ["ofrezco", "tenemos", "comparto", "alquilo", "vendo"]) or URL_RE.search(t):
        return "LISTING"
    if any(k in t for k in ["reservad", "vendid", "seña"]):
        return "STATUS"
    return "OTHER"

def detect_operation(text: str) -> str:
    t = (text or "").lower()
    if any(h in t for h in RENT_HINTS):
        return "rent"
    if any(h in t for h in BUY_HINTS):
        return "buy"
    if "busco" in t:
        return "buy"
    return "unknown"

def detect_property_type(text: str) -> str:
    t = (text or "").lower()
    for k, v in PROPERTY_TYPE_HINTS:
        if k in t:
            return v
    return "Desconocido"

def normalize_zone_list(text: str) -> List[str]:
    t = (text or "").lower()
    zones = set()

    z = re.search(r"zonas?\s*:\s*(.+)", t, re.IGNORECASE)
    if z:
        raw = z.group(1)
        for part in re.split(r"[,\n;•]+", raw):
            p = part.strip(" .*-_").lower()
            if p:
                zones.add(ZONE_DICT.get(p, p.title()))

    for k, v in ZONE_DICT.items():
        if k in t:
            zones.add(v)
    return sorted(zones)

def extract_bedrooms(text: str) -> Tuple[Optional[int], Optional[int]]:
    t = (text or "").lower()
    m = DORM_RE.search(t)
    if m:
        return int(m.group(1)), None
    m2 = re.search(r"(\d)\s*o\s*(\d)\s*dor", t, re.IGNORECASE)
    if m2:
        return int(m2.group(1)), int(m2.group(2))
    if "no mono" in t:
        return 1, None
    return None, None

def _clean_num(s: str) -> int:
    return int(s.replace(".", "").replace(" ", ""))

def extract_money(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").lower()
    currency = None
    if USD_RE.search(t):
        currency = "USD"
    elif UYU_RE.search(t):
        currency = "UYU"

    rm = RANGE_RE.search(t)
    if rm:
        a = _clean_num(rm.group(1))
        b = _clean_num(rm.group(2))
        tail = t[rm.start(): rm.end()+25]
        if "mil" in tail or "k" in tail:
            if a < 1000: a *= 1000
            if b < 1000: b *= 1000
        return {"currency": currency or "USD", "min": min(a, b), "max": max(a, b), "confidence": 0.8}

    candidates = []
    for m in NUM_RE.finditer(t):
        raw = m.group(1)
        suffix = (m.group(2) or "").lower()
        n = _clean_num(raw)
        if suffix in ("k", "mil"):
            n *= 1000
        if n < 500:
            continue
        candidates.append(n)
    if not candidates:
        return None

    op = detect_operation(t)
    n = max(candidates) if op == "buy" else min(candidates)
    conf = 0.7 if currency else 0.4

    if "35mil" in t and not USD_RE.search(t):
        return {"currency": "UYU", "min": None, "max": 35000, "confidence": 0.35}

    return {"currency": currency or ("USD" if n >= 5000 else "UYU"), "min": None, "max": n, "confidence": conf}

def extract_m2(text: str) -> Optional[int]:
    m = M2_RE.search((text or "").lower())
    return int(m.group(1)) if m else None

def extract_keywords(text: str) -> List[str]:
    t = (text or "").lower()
    return [k for k in KEYWORD_BUILDINGS if k in t]

def parse_whatsapp_export(txt: str) -> pd.DataFrame:
    lines = txt.splitlines()
    messages = []
    current = None

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            if current:
                messages.append(current)
            d, mo, y, hh, mm, sender, text = m.groups()
            current = Message(ts=parse_ts(d, mo, y, hh, mm), sender_raw=sender, text=text)
        else:
            if current:
                current.text += "\n" + line
    if current:
        messages.append(current)

    rows = []
    for msg in messages:
        sender_phone = normalize_phone(msg.sender_raw) or normalize_phone(msg.text)
        msg_type = classify_message(msg.text)
        urls = URL_RE.findall(msg.text or "")
        rows.append({
            "ts": msg.ts,
            "sender_raw": msg.sender_raw,
            "sender_phone": sender_phone,
            "msg_type": msg_type,
            "has_media": has_media(msg.text),
            "urls": urls,
            "text": (msg.text or "").strip(),
        })
    return pd.DataFrame(rows)

def build_leads(df_msgs: pd.DataFrame) -> pd.DataFrame:
    leads = []
    for _, r in df_msgs[df_msgs["msg_type"] == "LEAD_REQUEST"].iterrows():
        text = r["text"]
        phone = r["sender_phone"]
        if not phone:
            continue

        name = r["sender_raw"] if (r["sender_raw"] and not str(r["sender_raw"]).startswith("+")) else ""
        tail = [x.strip() for x in str(text).splitlines() if x.strip()]
        if not name and tail:
            cand = tail[-1]
            if len(cand) <= 45 and not any(k in cand.lower() for k in ["usd", "u$s", "dorm", "zona", "ppto", "presupuesto"]):
                name = cand

        op = detect_operation(text)
        ptype = detect_property_type(text)
        bmin, bmax = extract_bedrooms(text)
        money = extract_money(text)
        zones = normalize_zone_list(text)
        kw = extract_keywords(text)
        min_m2 = extract_m2(text)
        pets_req = 1 if ("mascota" in text.lower() or "perro" in text.lower()) else 0

        t = text.lower()
        urgency = 0
        if any(x in t for x in ["urgente", "para ver mañana", "para ver hoy", "cerrar ya"]):
            urgency += 2
        if any(x in t for x in ["cliente concreto", "cliente activo"]):
            urgency += 2
        if "para ver" in t:
            urgency += 1

        leads.append({
            "lead_id": f"lead:{phone}",
            "ts": r["ts"],
            "phone": phone,
            "name": name,
            "operation": op,
            "property_type": ptype,
            "bedrooms_min": bmin,
            "bedrooms_max": bmax,
            "budget_currency": (money or {}).get("currency"),
            "budget_min": (money or {}).get("min"),
            "budget_max": (money or {}).get("max"),
            "zones": zones,
            "keywords": kw,
            "min_m2": min_m2,
            "pets_required": pets_req,
            "confidence": (money or {}).get("confidence", 0.25),
            "urgency": urgency,
            "raw_text": text[:1400],
            "urls": r["urls"],
        })

    if not leads:
        return pd.DataFrame()

    df = pd.DataFrame(leads)
    df = df.sort_values("ts", ascending=False).drop_duplicates(subset=["phone"], keep="first")
    return df

def build_listings(df_msgs: pd.DataFrame) -> pd.DataFrame:
    listings = []
    for _, r in df_msgs[df_msgs["msg_type"] == "LISTING"].iterrows():
        text = r["text"]
        phone = r["sender_phone"]
        name = r["sender_raw"] if (r["sender_raw"] and not str(r["sender_raw"]).startswith("+")) else ""

        op = detect_operation(text)
        ptype = detect_property_type(text)
        b, _ = extract_bedrooms(text)
        money = extract_money(text)
        zones = normalize_zone_list(text)
        zone_guess = zones[0] if zones else None
        kw = extract_keywords(text)

        # ID estable: por teléfono + ts
        listing_id = f"lst:{(phone or 'na')}:{int(pd.Timestamp(r['ts']).timestamp())}"

        listings.append({
            "listing_id": listing_id,
            "ts": r["ts"],
            "lister_phone": phone,
            "lister_name": name,
            "operation": op,
            "property_type": ptype,
            "zone_guess": zone_guess,
            "zones": zones,
            "bedrooms": b,
            "price_currency": (money or {}).get("currency"),
            "price": (money or {}).get("max"),
            "m2": extract_m2(text),
            "keywords": kw,
            "confidence": (money or {}).get("confidence", 0.25),
            "raw_text": text[:1400],
            "urls": r["urls"],
        })

    if not listings:
        return pd.DataFrame()

    return pd.DataFrame(listings).sort_values("ts", ascending=False)

# =========================
# Overrides + Funnel helpers
# =========================
def apply_overrides(record: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return record
    out = dict(record)
    for k, v in overrides.items():
        if v is None:
            continue
        out[k] = v
    return out

def get_funnel(state: Dict[str, Any], entity_id: str) -> str:
    return state.get("funnel", {}).get(entity_id, "nuevo")

def set_funnel(state: Dict[str, Any], entity_id: str, status: str):
    state.setdefault("funnel", {})[entity_id] = status

def get_note(state: Dict[str, Any], entity_id: str) -> str:
    return state.get("notes", {}).get(entity_id, "")

def set_note(state: Dict[str, Any], entity_id: str, note: str):
    state.setdefault("notes", {})[entity_id] = note

def set_override(state: Dict[str, Any], entity_id: str, patch: Dict[str, Any]):
    state.setdefault("overrides", {}).setdefault(entity_id, {})
    state["overrides"][entity_id].update(patch)

def get_override(state: Dict[str, Any], entity_id: str) -> Dict[str, Any]:
    return state.get("overrides", {}).get(entity_id, {})

# =========================
# MATCHING ENGINE v1
# =========================
def intersects(a: List[str], b: List[str]) -> bool:
    return bool(set(a or []) & set(b or []))

def money_compatible(lead: Dict[str, Any], listing: Dict[str, Any]) -> Tuple[bool, str]:
    # listing.price <= lead.budget_max (si ambos existen y moneda coincide)
    lc = lead.get("budget_currency")
    lm = lead.get("budget_max")
    pc = listing.get("price_currency")
    pr = listing.get("price")
    if not lm or not pr:
        return True, "monto faltante (no bloquea)"
    if lc and pc and lc != pc:
        return False, f"moneda distinta ({lc} vs {pc})"
    try:
        return (float(pr) <= float(lm)), "precio<=presupuesto" if float(pr) <= float(lm) else "precio>presupuesto"
    except Exception:
        return True, "monto ambiguo"

def dorm_compatible(lead: Dict[str, Any], listing: Dict[str, Any]) -> Tuple[bool, str]:
    bmin = lead.get("bedrooms_min")
    bmax = lead.get("bedrooms_max")
    bd = listing.get("bedrooms")
    if bd is None or bd == "" or pd.isna(bd):
        return True, "dorm faltante (no bloquea)"
    try:
        bd = int(bd)
    except Exception:
        return True, "dorm ambiguo"
    if bmin is None or bmin == "" or pd.isna(bmin):
        return True, "lead sin dorm (no bloquea)"
    try:
        bmin = int(bmin)
    except Exception:
        return True, "lead dorm ambiguo"
    if bmax is not None and bmax != "" and not pd.isna(bmax):
        try:
            bmax = int(bmax)
            return (bmin <= bd <= bmax), "dorm dentro de rango" if (bmin <= bd <= bmax) else "dorm fuera de rango"
        except Exception:
            return True, "rango dorm ambiguo"
    return (bd >= bmin), "dorm ok" if (bd >= bmin) else "dorm insuficiente"

def type_compatible(lead: Dict[str, Any], listing: Dict[str, Any]) -> Tuple[bool, str]:
    lt = (lead.get("property_type") or "").lower()
    pt = (listing.get("property_type") or "").lower()
    if not lt or lt == "desconocido" or not pt or pt == "desconocido":
        return True, "tipo faltante (no bloquea)"
    # si lead pide terreno, no le muestres apartamento, etc.
    if lt == pt:
        return True, "tipo coincide"
    # permisos blandos: "Apartamento" vs "Penthouse" (ambos apto)
    if ("apartamento" in lt and "penthouse" in pt) or ("penthouse" in lt and "apartamento" in pt):
        return True, "tipo similar"
    return False, "tipo no coincide"

def kw_overlap(lead: Dict[str, Any], listing: Dict[str, Any]) -> bool:
    return bool(set(lead.get("keywords") or []) & set(listing.get("keywords") or []))

def match_score(lead: Dict[str, Any], listing: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    # operación
    if lead.get("operation") != "unknown" and listing.get("operation") != "unknown":
        if lead["operation"] != listing["operation"]:
            return 0, ["operación distinta"]
        score += 10
        reasons.append("operación coincide")

    # zona
    lz = lead.get("zones") or []
    pz = listing.get("zones") or ([] if not listing.get("zone_guess") else [listing.get("zone_guess")])
    if intersects(lz, pz):
        score += 30
        reasons.append("zona coincide")
    else:
        # no bloquea totalmente, pero baja
        score -= 10
        reasons.append("zona no coincide")

    # presupuesto
    ok_money, why_money = money_compatible(lead, listing)
    if ok_money:
        score += 25
    else:
        score -= 30
    reasons.append(why_money)

    # dorm
    ok_dorm, why_dorm = dorm_compatible(lead, listing)
    if ok_dorm:
        score += 15
    else:
        score -= 20
    reasons.append(why_dorm)

    # tipo
    ok_type, why_type = type_compatible(lead, listing)
    if ok_type:
        score += 15
    else:
        score -= 25
    reasons.append(why_type)

    # keywords edificio
    if kw_overlap(lead, listing):
        score += 18
        reasons.append("keyword/edificio coincide")

    # urgencia
    urg = int(lead.get("urgency") or 0)
    score += min(urg, 4) * 3
    if urg >= 2:
        reasons.append("lead urgente")

    # confianza
    conf = float(lead.get("confidence") or 0.0)
    if conf < 0.35:
        score -= 8
        reasons.append("confianza baja")
    elif conf >= 0.6:
        score += 4
        reasons.append("confianza buena")

    return max(0, min(100, int(score))), reasons

def compute_matches_for_lead(lead: Dict[str, Any], listings: List[Dict[str, Any]], topn=15) -> List[Dict[str, Any]]:
    out = []
    for lst in listings:
        s, reasons = match_score(lead, lst)
        if s <= 0:
            continue
        out.append({"score": s, "reasons": reasons, "listing": lst})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:topn]

def compute_matches_for_listing(listing: Dict[str, Any], leads: List[Dict[str, Any]], topn=15) -> List[Dict[str, Any]]:
    out = []
    for ld in leads:
        s, reasons = match_score(ld, listing)
        if s <= 0:
            continue
        out.append({"score": s, "reasons": reasons, "lead": ld})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:topn]

# =========================
# UI helpers
# =========================
def chip_line(items: List[str]) -> str:
    if not items:
        return ""
    return "  ".join([f"`{x}`" for x in items])

def wa_link(phone: str) -> Optional[str]:
    if not phone:
        return None
    return f"https://wa.me/{phone.replace('+','')}"

def money_fmt(currency, amount):
    if not amount:
        return "—"
    cur = currency or ""
    try:
        a = int(float(amount))
        return f"{cur} {a:,}".replace(",", ".")
    except Exception:
        return f"{cur} {amount}"

def section_title(txt):
    st.markdown(f"### {txt}")

# =========================
# APP
# =========================
st.set_page_config(page_title="BrokerOS – WhatsApp", layout="wide")
st.title("BrokerOS – WhatsApp Deal Flow (v2)")
st.caption("Más orden • Matching bidireccional • Embudo • Overrides • Cards móviles")

# Load state
if "app_state" not in st.session_state:
    st.session_state["app_state"] = load_state()

state = st.session_state["app_state"]

with st.sidebar:
    st.header("Fuente")
    mode = st.radio("Cargar desde", ["Archivo en repo", "Pegar texto"], index=0)
    txt = ""

    if mode == "Archivo en repo":
        filename = st.text_input("Nombre del .txt", value="inmo_registradas.txt")
        try:
            with open(filename, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()
            st.success(f"OK: {filename}")
        except Exception:
            st.error("No pude abrir el archivo. Verificá el nombre y que esté subido al repo.")
    else:
        txt = st.text_area("Pegá el export acá", height=220)

    st.divider()
    nav = st.radio("Navegación", ["Prioridad HOY", "Leads", "Listings", "Matching", "Auditoría"], index=0)

    st.divider()
    st.subheader("Filtros")
    min_conf = st.slider("Confianza mínima", 0.0, 1.0, 0.30, 0.05)
    show_other = st.checkbox("Mostrar OTHER", value=False)
    show_media = st.checkbox("Incluir MEDIA en auditoría", value=True)

    st.divider()
    if st.button("💾 Guardar estado (overrides/embudo/notas)"):
        save_state(state)
        st.success("Guardado (si la plataforma lo permite).")

if not txt.strip():
    st.warning("Subí `inmo_registradas.txt` al repo (raíz) o pegá texto en la barra lateral.")
    st.stop()

df_msgs = parse_whatsapp_export(txt)
df_msgs = df_msgs[df_msgs["msg_type"] != "SYSTEM"].copy()
if not show_other:
    df_msgs = df_msgs[df_msgs["msg_type"] != "OTHER"].copy()

df_leads = build_leads(df_msgs)
df_listings = build_listings(df_msgs)

# Convert to dict lists + apply overrides
leads_list = []
if len(df_leads):
    for r in df_leads.to_dict(orient="records"):
        ov = get_override(state, r["lead_id"])
        leads_list.append(apply_overrides(r, ov))

listings_list = []
if len(df_listings):
    for r in df_listings.to_dict(orient="records"):
        ov = get_override(state, r["listing_id"])
        listings_list.append(apply_overrides(r, ov))

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Leads", int((df_msgs["msg_type"] == "LEAD_REQUEST").sum()))
k2.metric("Listings", int((df_msgs["msg_type"] == "LISTING").sum()))
k3.metric("Media", int((df_msgs["msg_type"] == "MEDIA").sum()))
k4.metric("Último", df_msgs["ts"].max().strftime("%d/%m %H:%M") if len(df_msgs) else "—")

st.divider()

# Selection keys (persist)
if "selected_lead_id" not in st.session_state:
    st.session_state["selected_lead_id"] = None
if "selected_listing_id" not in st.session_state:
    st.session_state["selected_listing_id"] = None

def pick_lead_id(lid): 
    st.session_state["selected_lead_id"] = lid
    st.session_state["selected_listing_id"] = None

def pick_listing_id(pid):
    st.session_state["selected_listing_id"] = pid
    st.session_state["selected_lead_id"] = None

def find_lead(lid):
    return next((x for x in leads_list if x["lead_id"] == lid), None)

def find_listing(pid):
    return next((x for x in listings_list if x["listing_id"] == pid), None)

# ===== UI: two panels =====
left, right = st.columns([1.25, 1])

# =========================
# LEFT PANEL (lists)
# =========================
with left:
    if nav == "Prioridad HOY":
        section_title("Prioridad HOY")
        view = [x for x in leads_list if float(x.get("confidence") or 0) >= min_conf]
        view.sort(key=lambda x: (int(x.get("urgency") or 0), x.get("ts")), reverse=True)
        view = view[:40]

        if not view:
            st.info("No hay leads con filtros actuales.")
        else:
            for r in view:
                zones_txt = ", ".join(r.get("zones") or [])
                money = money_fmt(r.get("budget_currency"), r.get("budget_max"))
                dorm = r.get("bedrooms_min") if r.get("bedrooms_min") is not None else "—"
                kw = r.get("keywords") or []
                status = get_funnel(state, r["lead_id"])

                tags = []
                if r.get("pets_required") == 1: tags.append("Mascota")
                if int(r.get("urgency") or 0) >= 2: tags.append("Urgente")
                if kw: tags.append("Edificio")
                tags.append(f"Estado:{status}")

                st.markdown(
                    f"""
<div style="border:1px solid #E5E7EB;border-radius:16px;padding:12px;margin-bottom:10px;background:white;">
  <div style="font-weight:800;font-size:15px;">LEAD • {r.get("property_type","")} • {money}</div>
  <div style="color:#6B7280;font-size:13px;margin-top:4px;">
    {r.get("phone","")} · {zones_txt or "Zonas: —"} · Dorm: {dorm}
  </div>
  <div style="margin-top:8px;">{chip_line(tags)}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button("Ver detalle", key=f"ld:{r['lead_id']}"):
                        pick_lead_id(r["lead_id"])
                with c2:
                    link = wa_link(r.get("phone"))
                    if link:
                        st.markdown(f"[WhatsApp]({link})")
                with c3:
                    if st.button("Contactado", key=f"ldc:{r['lead_id']}"):
                        set_funnel(state, r["lead_id"], "contactado")
                        save_state(state)
                        st.toast("Marcado contactado")

    elif nav == "Leads":
        section_title("Leads")
        view = [x for x in leads_list if float(x.get("confidence") or 0) >= min_conf]
        view.sort(key=lambda x: x.get("ts"), reverse=True)

        for r in view[:80]:
            zones_txt = ", ".join(r.get("zones") or [])
            money = money_fmt(r.get("budget_currency"), r.get("budget_max"))
            status = get_funnel(state, r["lead_id"])
            st.markdown(
                f"""
<div style="border:1px solid #E5E7EB;border-radius:16px;padding:12px;margin-bottom:10px;background:white;">
  <div style="font-weight:800;font-size:15px;">{money} • {r.get("property_type","")}</div>
  <div style="color:#6B7280;font-size:13px;margin-top:4px;">
    {r.get("phone","")} · {zones_txt or "Zonas: —"} · Estado: {status}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            if st.button("Ver detalle", key=f"ld2:{r['lead_id']}"):
                pick_lead_id(r["lead_id"])

    elif nav == "Listings":
        section_title("Listings")
        view = [x for x in listings_list if float(x.get("confidence") or 0) >= min_conf]
        for r in view[:80]:
            price = money_fmt(r.get("price_currency"), r.get("price"))
            zone = r.get("zone_guess") or (", ".join(r.get("zones") or []) or "Zona: —")
            status = get_funnel(state, r["listing_id"])
            st.markdown(
                f"""
<div style="border:1px solid #E5E7EB;border-radius:16px;padding:12px;margin-bottom:10px;background:white;">
  <div style="font-weight:800;font-size:15px;">{price} • {r.get("property_type","")}</div>
  <div style="color:#6B7280;font-size:13px;margin-top:4px;">
    {zone} · Dorm: {r.get("bedrooms") or "—"} · Estado: {status}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            if st.button("Ver detalle", key=f"ls:{r['listing_id']}"):
                pick_listing_id(r["listing_id"])

    elif nav == "Matching":
        section_title("Matching")
        mode2 = st.radio("Modo", ["Desde un LEAD", "Desde un LISTING"], horizontal=True)

        if mode2 == "Desde un LEAD":
            options = [(x["lead_id"], f'{x.get("phone")} • {x.get("property_type")} • {money_fmt(x.get("budget_currency"), x.get("budget_max"))}') for x in leads_list]
            if not options:
                st.info("No hay leads.")
            else:
                sel = st.selectbox("Elegí lead", options, format_func=lambda t: t[1])
                lid = sel[0]
                pick_lead_id(lid)

        else:
            options = [(x["listing_id"], f'{money_fmt(x.get("price_currency"), x.get("price"))} • {x.get("property_type")} • {(x.get("zone_guess") or "zona?")}') for x in listings_list]
            if not options:
                st.info("No hay listings.")
            else:
                sel = st.selectbox("Elegí listing", options, format_func=lambda t: t[1])
                pid = sel[0]
                pick_listing_id(pid)

    elif nav == "Auditoría":
        section_title("Auditoría / Inbox")
        show = df_msgs.copy()
        if not show_media:
            show = show[show["msg_type"] != "MEDIA"].copy()
        show["ts"] = show["ts"].dt.strftime("%d/%m %H:%M")
        st.dataframe(show[["ts","msg_type","sender_raw","sender_phone","has_media","text"]], use_container_width=True, hide_index=True)

# =========================
# RIGHT PANEL (detail + actions + matching)
# =========================
with right:
    sel_lead = find_lead(st.session_state["selected_lead_id"]) if st.session_state["selected_lead_id"] else None
    sel_lst = find_listing(st.session_state["selected_listing_id"]) if st.session_state["selected_listing_id"] else None

    if not sel_lead and not sel_lst:
        st.info("Seleccioná un Lead o un Listing para ver detalle + matching.")
        st.stop()

    if sel_lead:
        section_title("Detalle LEAD")
        status = get_funnel(state, sel_lead["lead_id"])
        zones_txt = ", ".join(sel_lead.get("zones") or [])
        money = money_fmt(sel_lead.get("budget_currency"), sel_lead.get("budget_max"))
        dorm = sel_lead.get("bedrooms_min") if sel_lead.get("bedrooms_min") is not None else "—"
        kw = ", ".join(sel_lead.get("keywords") or [])

        st.markdown(f"**Tel:** {sel_lead.get('phone')}  \n**Estado:** `{status}`")
        st.markdown(f"**Tipo:** {sel_lead.get('property_type')} · **Operación:** {sel_lead.get('operation')} · **Dorm:** {dorm}")
        st.markdown(f"**Presupuesto:** {money}  \n**Zonas:** {zones_txt or '—'}  \n**Keywords:** {kw or '—'}")

        # Funnel controls
        st.write("**Embudo**")
        new_status = st.selectbox("Cambiar estado", FUNNEL_STATES, index=FUNNEL_STATES.index(status) if status in FUNNEL_STATES else 0)
        if st.button("Guardar estado", key="save_lead_status"):
            set_funnel(state, sel_lead["lead_id"], new_status)
            save_state(state)
            st.success("Estado guardado")

        # Notes
        note = st.text_area("Notas", value=get_note(state, sel_lead["lead_id"]), height=90)
        if st.button("Guardar nota", key="save_lead_note"):
            set_note(state, sel_lead["lead_id"], note)
            save_state(state)
            st.success("Nota guardada")

        # WhatsApp template
        template = (
            f"Hola {sel_lead.get('name','') or ''}! Vi tu búsqueda de {sel_lead.get('property_type','propiedad')} "
            f"({dorm} dorm) en {zones_txt}. ¿Sigue vigente? Tengo opciones dentro de {money}. "
            "Si querés, te mando 2-3 alternativas y coordinamos visita."
        ).strip()
        st.text_area("Plantilla WhatsApp", value=template, height=120)
        link = wa_link(sel_lead.get("phone"))
        if link:
            st.markdown(f"💬 [Abrir WhatsApp]({link})")

        # Overrides
        st.divider()
        section_title("Corrección (override)")
        with st.expander("Editar datos detectados (mejora matching)", expanded=False):
            o_zone = st.text_input("Zonas (separadas por coma)", value=", ".join(sel_lead.get("zones") or []))
            o_curr = st.selectbox("Moneda presupuesto", ["", "USD", "UYU"], index=["", "USD", "UYU"].index(sel_lead.get("budget_currency") or ""))
            o_budget = st.text_input("Presupuesto max", value=str(sel_lead.get("budget_max") or ""))
            o_dorm = st.text_input("Dormitorios min", value=str(sel_lead.get("bedrooms_min") or ""))
            o_type = st.text_input("Tipo (ej: Apartamento/Casa/Terreno)", value=str(sel_lead.get("property_type") or ""))
            o_op = st.selectbox("Operación", ["buy", "rent", "unknown"], index=["buy", "rent", "unknown"].index(sel_lead.get("operation") or "unknown"))

            if st.button("Aplicar override", key="apply_lead_override"):
                patch = {
                    "zones": [z.strip() for z in o_zone.split(",") if z.strip()],
                    "budget_currency": o_curr or None,
                    "budget_max": int(o_budget) if o_budget.strip().isdigit() else (o_budget.strip() or None),
                    "bedrooms_min": int(o_dorm) if o_dorm.strip().isdigit() else (None if o_dorm.strip()=="" else o_dorm.strip()),
                    "property_type": o_type or None,
                    "operation": o_op or None,
                }
                set_override(state, sel_lead["lead_id"], patch)
                save_state(state)
                st.success("Override aplicado. Volvé a abrir el lead para ver el efecto.")

        st.divider()
        section_title("Top Listings compatibles")
        matches = compute_matches_for_lead(sel_lead, listings_list, topn=12)
        if not matches:
            st.warning("No encontré matches (con reglas actuales).")
        else:
            for m in matches:
                lst = m["listing"]
                price = money_fmt(lst.get("price_currency"), lst.get("price"))
                zone = lst.get("zone_guess") or (", ".join(lst.get("zones") or []) or "zona?")
                tags = [f"score:{m['score']}"]
                if kw_overlap(sel_lead, lst): tags.append("edificio")
                tags.append(f"estado:{get_funnel(state, lst['listing_id'])}")
                st.markdown(
                    f"""
<div style="border:1px solid #E5E7EB;border-radius:14px;padding:10px;margin-bottom:8px;background:white;">
  <div style="font-weight:800;">{price} • {lst.get("property_type","")} • {zone}</div>
  <div style="color:#6B7280;font-size:13px;">Razones: {", ".join(m["reasons"][:4])}</div>
  <div style="margin-top:6px;">{chip_line(tags)}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("Ver listing", key=f"view_lst_{lst['listing_id']}"):
                        pick_listing_id(lst["listing_id"])
                with c2:
                    if st.button("Marcar 'enviado'", key=f"sent_{lst['listing_id']}"):
                        # evento simple
                        state.setdefault("events", []).append({"ts": datetime.now().isoformat(), "lead": sel_lead["lead_id"], "listing": lst["listing_id"], "event": "enviado"})
                        save_state(state)
                        st.toast("Registrado: enviado")

        st.divider()
        st.write("**Mensaje original (recortado)**")
        st.code(sel_lead.get("raw_text",""), language="text")

    if sel_lst:
        section_title("Detalle LISTING")
        status = get_funnel(state, sel_lst["listing_id"])
        price = money_fmt(sel_lst.get("price_currency"), sel_lst.get("price"))
        zone = sel_lst.get("zone_guess") or (", ".join(sel_lst.get("zones") or []) or "zona?")
        kw = ", ".join(sel_lst.get("keywords") or [])

        st.markdown(f"**Zona:** {zone}  \n**Precio:** {price}  \n**Estado:** `{status}`")
        st.markdown(f"**Tipo:** {sel_lst.get('property_type')} · **Operación:** {sel_lst.get('operation')} · **Dorm:** {sel_lst.get('bedrooms') or '—'}")
        st.markdown(f"**Keywords:** {kw or '—'}")

        if sel_lst.get("urls"):
            st.write("**Links**")
            for u in sel_lst["urls"][:6]:
                st.markdown(f"- {u}")

        st.write("**Embudo**")
        new_status = st.selectbox("Cambiar estado", FUNNEL_STATES, index=FUNNEL_STATES.index(status) if status in FUNNEL_STATES else 0, key="lst_status")
        if st.button("Guardar estado", key="save_lst_status"):
            set_funnel(state, sel_lst["listing_id"], new_status)
            save_state(state)
            st.success("Estado guardado")

        note = st.text_area("Notas", value=get_note(state, sel_lst["listing_id"]), height=90, key="lst_note")
        if st.button("Guardar nota", key="save_lst_note"):
            set_note(state, sel_lst["listing_id"], note)
            save_state(state)
            st.success("Nota guardada")

        st.divider()
        section_title("Corrección (override)")
        with st.expander("Editar datos detectados", expanded=False):
            o_zone = st.text_input("Zona guess (texto)", value=str(sel_lst.get("zone_guess") or ""), key="o_zone_lst")
            o_curr = st.selectbox("Moneda precio", ["", "USD", "UYU"], index=["", "USD", "UYU"].index(sel_lst.get("price_currency") or ""), key="o_curr_lst")
            o_price = st.text_input("Precio", value=str(sel_lst.get("price") or ""), key="o_price_lst")
            o_dorm = st.text_input("Dormitorios", value=str(sel_lst.get("bedrooms") or ""), key="o_dorm_lst")
            o_type = st.text_input("Tipo", value=str(sel_lst.get("property_type") or ""), key="o_type_lst")
            o_op = st.selectbox("Operación", ["buy", "rent", "unknown"], index=["buy", "rent", "unknown"].index(sel_lst.get("operation") or "unknown"), key="o_op_lst")

            if st.button("Aplicar override listing", key="apply_lst_override"):
                patch = {
                    "zone_guess": o_zone or None,
                    "price_currency": o_curr or None,
                    "price": int(o_price) if o_price.strip().isdigit() else (o_price.strip() or None),
                    "bedrooms": int(o_dorm) if o_dorm.strip().isdigit() else (None if o_dorm.strip()=="" else o_dorm.strip()),
                    "property_type": o_type or None,
                    "operation": o_op or None,
                }
                set_override(state, sel_lst["listing_id"], patch)
                save_state(state)
                st.success("Override aplicado. Volvé a abrir el listing para ver el efecto.")

        st.divider()
        section_title("Top Leads compatibles")
        matches = compute_matches_for_listing(sel_lst, leads_list, topn=12)
        if not matches:
            st.warning("No encontré matches (con reglas actuales).")
        else:
            for m in matches:
                ld = m["lead"]
                money = money_fmt(ld.get("budget_currency"), ld.get("budget_max"))
                zones_txt = ", ".join(ld.get("zones") or [])
                tags = [f"score:{m['score']}", f"estado:{get_funnel(state, ld['lead_id'])}"]
                if kw_overlap(ld, sel_lst): tags.append("edificio")

                st.markdown(
                    f"""
<div style="border:1px solid #E5E7EB;border-radius:14px;padding:10px;margin-bottom:8px;background:white;">
  <div style="font-weight:800;">{money} • {ld.get("property_type","")} • {ld.get("phone","")}</div>
  <div style="color:#6B7280;font-size:13px;">Zonas: {zones_txt or "—"} · Razones: {", ".join(m["reasons"][:4])}</div>
  <div style="margin-top:6px;">{chip_line(tags)}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button("Ver lead", key=f"view_ld_{ld['lead_id']}"):
                        pick_lead_id(ld["lead_id"])
                with c2:
                    link = wa_link(ld.get("phone"))
                    if link:
                        st.markdown(f"[WhatsApp]({link})")
                with c3:
                    if st.button("Contactado", key=f"ld_contact_{ld['lead_id']}"):
                        set_funnel(state, ld["lead_id"], "contactado")
                        save_state(state)
                        st.toast("Marcado contactado")

        st.divider()
        st.write("**Mensaje original (recortado)**")
        st.code(sel_lst.get("raw_text",""), language="text")