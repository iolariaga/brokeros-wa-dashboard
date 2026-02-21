import re
import json
import sqlite3
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dateutil import tz

# =========================
# CONFIG
# =========================
APP_NAME = "BrokerOS • WhatsApp Deal Flow"
DEFAULT_TZ = tz.gettz("America/Montevideo")

DB_PATH = "brokeros.db"          # persistencia sólida
DEFAULT_TXT_FILENAME = "inmo_registradas.txt"

FUNNEL_STATES = ["nuevo", "contactado", "respondió", "agendado", "visita", "reserva", "cierre", "perdido"]

# Normalización de zonas (extendible)
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
    "balneario buenos aires": "Balneario Buenos Aires",
    "maldonado": "Maldonado",
}

PROPERTY_TYPE_HINTS = [
    ("terreno", "Terreno"),
    ("lote", "Terreno"),
    ("campo", "Campo"),
    ("local comercial", "Local/Oficina"),
    ("oficina", "Local/Oficina"),
    ("galpón", "Depósito/Galpón"),
    ("galpon", "Depósito/Galpón"),
    ("deposito", "Depósito/Galpón"),
    ("depósito", "Depósito/Galpón"),
    ("ph", "PH"),
    ("penthouse", "Penthouse"),
    ("casa", "Casa"),
    ("apartamento", "Apartamento"),
    ("apto", "Apartamento"),
    ("depto", "Apartamento"),
]

BUY_HINTS = ["en venta", "para compra", "compra", "vendo", "ofrezco", "tenemos", "en exclusividad", "venta"]
RENT_HINTS = ["alquiler", "alquilo", "anual", "invernal", "temporal", "quincena", "mensuales", "mes"]

KEYWORD_BUILDINGS = [
    "torres del este", "gala", "gala puerto", "gala vista", "gala tower",
    "tiburón ii", "tiburon ii", "green park", "green life", "le parc", "tequendama",
    "tunquelen", "millenium", "trump", "venetian", "ancora", "malecón", "malecon",
    "miami boulevard", "casino miguez", "marigot", "signature", "citrea",
    "brava 28", "villa brava", "espacio 1"
]

BROKER_WORDS = [
    "inmobiliaria", "bienes raices", "bienes raíces", "propiedades", "remax",
    "real estate", "negocios inmobiliarios", "broker", "agente"
]

SYSTEM_HINTS = ["cifrados de extremo a extremo", "creó el grupo", "se te añadió al grupo", "obten"]

# =========================
# REGEX
# =========================
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
M2_RE = re.compile(r"(\d{2,5})\s*m2", re.IGNORECASE)

# =========================
# DB
# =========================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def db_init(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS overrides(
        entity_id TEXT PRIMARY KEY,
        patch_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS funnel(
        entity_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS notes(
        entity_id TEXT PRIMARY KEY,
        note TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        event TEXT NOT NULL,
        lead_id TEXT,
        listing_id TEXT,
        payload_json TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS ingest_runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        source TEXT NOT NULL,
        msg_count INTEGER NOT NULL,
        lead_count INTEGER NOT NULL,
        listing_count INTEGER NOT NULL
    )
    """)
    conn.commit()

def now_iso() -> str:
    return datetime.now(tz=DEFAULT_TZ).isoformat()

def db_get_json(conn, table: str, entity_id: str) -> Dict[str, Any]:
    cur = conn.execute(f"SELECT patch_json FROM {table} WHERE entity_id=?", (entity_id,))
    row = cur.fetchone()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except Exception:
        return {}

def db_set_json(conn, table: str, entity_id: str, patch: Dict[str, Any]):
    conn.execute(
        f"INSERT INTO {table}(entity_id, patch_json, updated_at) VALUES(?,?,?) "
        f"ON CONFLICT(entity_id) DO UPDATE SET patch_json=excluded.patch_json, updated_at=excluded.updated_at",
        (entity_id, json.dumps(patch, ensure_ascii=False), now_iso())
    )
    conn.commit()

def db_get_funnel(conn, entity_id: str) -> str:
    cur = conn.execute("SELECT status FROM funnel WHERE entity_id=?", (entity_id,))
    row = cur.fetchone()
    return row[0] if row else "nuevo"

def db_set_funnel(conn, entity_id: str, status: str):
    conn.execute(
        "INSERT INTO funnel(entity_id, status, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(entity_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at",
        (entity_id, status, now_iso())
    )
    conn.commit()

def db_get_note(conn, entity_id: str) -> str:
    cur = conn.execute("SELECT note FROM notes WHERE entity_id=?", (entity_id,))
    row = cur.fetchone()
    return row[0] if row else ""

def db_set_note(conn, entity_id: str, note: str):
    conn.execute(
        "INSERT INTO notes(entity_id, note, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(entity_id) DO UPDATE SET note=excluded.note, updated_at=excluded.updated_at",
        (entity_id, note, now_iso())
    )
    conn.commit()

def db_add_event(conn, event: str, lead_id: Optional[str], listing_id: Optional[str], payload: Dict[str, Any] | None = None):
    conn.execute(
        "INSERT INTO events(ts, event, lead_id, listing_id, payload_json) VALUES(?,?,?,?,?)",
        (now_iso(), event, lead_id, listing_id, json.dumps(payload or {}, ensure_ascii=False))
    )
    conn.commit()

def db_add_ingest(conn, source: str, msg_count: int, lead_count: int, listing_count: int):
    conn.execute(
        "INSERT INTO ingest_runs(ts, source, msg_count, lead_count, listing_count) VALUES(?,?,?,?,?)",
        (now_iso(), source, msg_count, lead_count, listing_count)
    )
    conn.commit()

def db_last_ingest(conn) -> Optional[Dict[str, Any]]:
    cur = conn.execute("SELECT ts, source, msg_count, lead_count, listing_count FROM ingest_runs ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        return None
    return {"ts": row[0], "source": row[1], "msg_count": row[2], "lead_count": row[3], "listing_count": row[4]}

# =========================
# PARSER / EXTRACTION
# =========================
@dataclass
class Message:
    ts: datetime
    sender_raw: str
    text: str

def parse_ts(d, mo, y, hh, mm) -> datetime:
    return datetime(int(y), int(mo), int(d), int(hh), int(mm), tzinfo=DEFAULT_TZ)

def normalize_phone(s: str) -> Optional[str]:
    m = PHONE_RE.search(s or "")
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}"

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

    m2 = re.search(r"(\d)\s*o\s*(\d)\s*dor", t, re.IGNORECASE)
    if m2:
        return int(m2.group(1)), int(m2.group(2))

    m = DORM_RE.search(t)
    if m:
        return int(m.group(1)), None

    if "no mono" in t or "no monoamb" in t:
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

    # rango
    rm = RANGE_RE.search(t)
    if rm:
        a = _clean_num(rm.group(1))
        b = _clean_num(rm.group(2))
        tail = t[rm.start(): rm.end()+25]
        if "mil" in tail or "k" in tail:
            if a < 1000: a *= 1000
            if b < 1000: b *= 1000
        return {"currency": currency or "USD", "min": min(a, b), "max": max(a, b), "confidence": 0.8}

    # números sueltos
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

    # caso típico "hasta 35mil" (alquiler anual en pesos)
    if "35mil" in t and not USD_RE.search(t):
        return {"currency": "UYU", "min": None, "max": 35000, "confidence": 0.35}

    return {"currency": currency or ("USD" if n >= 5000 else "UYU"), "min": None, "max": n, "confidence": conf}

def extract_m2(text: str) -> Optional[int]:
    m = M2_RE.search((text or "").lower())
    return int(m.group(1)) if m else None

def extract_keywords(text: str) -> List[str]:
    t = (text or "").lower()
    return [k for k in KEYWORD_BUILDINGS if k in t]

def infer_actor(text: str, sender_raw: str) -> str:
    t = (text or "").lower()
    s = (sender_raw or "").lower()
    if any(w in t for w in ["dueño directo", "propietario", "dueño"]):
        return "owner"
    if any(w in t for w in BROKER_WORDS) or any(w in s for w in BROKER_WORDS):
        return "broker"
    # por defecto: broker (en grupos de inmobiliarias suele ser así)
    return "broker"

def parse_whatsapp_export(txt: str) -> pd.DataFrame:
    lines = txt.splitlines()
    messages: List[Message] = []
    current: Optional[Message] = None

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
    df = pd.DataFrame(rows)
    if len(df):
        df["ts"] = pd.to_datetime(df["ts"])
    return df

def build_leads(df_msgs: pd.DataFrame) -> pd.DataFrame:
    leads = []
    for _, r in df_msgs[df_msgs["msg_type"] == "LEAD_REQUEST"].iterrows():
        text = r["text"]
        phone = r["sender_phone"]
        if not phone:
            continue

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
        if any(x in t for x in ["cliente concreto", "cliente activo", "cliente en punta"]):
            urgency += 2
        if "para ver" in t:
            urgency += 1

        actor = infer_actor(text, r["sender_raw"])

        leads.append({
            "lead_id": f"lead:{phone}",
            "ts": r["ts"],
            "phone": phone,
            "sender_raw": r["sender_raw"],
            "actor": actor,  # broker/owner
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
            "confidence": float((money or {}).get("confidence", 0.25)),
            "urgency": urgency,
            "raw_text": text[:2000],
            "urls": r["urls"],
        })

    if not leads:
        return pd.DataFrame()

    df = pd.DataFrame(leads).sort_values("ts", ascending=False)
    # dedupe por phone: nos quedamos con lo último
    df = df.drop_duplicates(subset=["phone"], keep="first")
    return df

def build_listings(df_msgs: pd.DataFrame) -> pd.DataFrame:
    listings = []
    for _, r in df_msgs[df_msgs["msg_type"] == "LISTING"].iterrows():
        text = r["text"]
        phone = r["sender_phone"]

        op = detect_operation(text)
        ptype = detect_property_type(text)
        b, _ = extract_bedrooms(text)
        money = extract_money(text)
        zones = normalize_zone_list(text)
        zone_guess = zones[0] if zones else None
        kw = extract_keywords(text)
        actor = infer_actor(text, r["sender_raw"])

        # ID estable
        listing_id = f"lst:{(phone or 'na')}:{int(pd.Timestamp(r['ts']).timestamp())}"

        listings.append({
            "listing_id": listing_id,
            "ts": r["ts"],
            "lister_phone": phone,
            "sender_raw": r["sender_raw"],
            "actor": actor,
            "operation": op,
            "property_type": ptype,
            "zone": zone_guess,
            "zones": zones,
            "bedrooms": b,
            "price_currency": (money or {}).get("currency"),
            "price": (money or {}).get("max"),
            "m2": extract_m2(text),
            "keywords": kw,
            "confidence": float((money or {}).get("confidence", 0.25)),
            "raw_text": text[:2000],
            "urls": r["urls"],
        })

    if not listings:
        return pd.DataFrame()

    df = pd.DataFrame(listings).sort_values("ts", ascending=False)
    return df

# =========================
# Overrides apply
# =========================
def apply_overrides(rec: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    if not patch:
        return rec
    out = dict(rec)
    for k, v in patch.items():
        if v is None:
            continue
        out[k] = v
    return out

# =========================
# Helpers / formatting
# =========================
def money_fmt(currency, amount) -> str:
    if amount is None or amount == "" or (isinstance(amount, float) and pd.isna(amount)):
        return "—"
    cur = currency or ""
    try:
        a = int(float(amount))
        return f"{cur} {a:,}".replace(",", ".")
    except Exception:
        return f"{cur} {amount}".strip()

def wa_link(phone: str) -> Optional[str]:
    if not phone:
        return None
    return f"https://wa.me/{phone.replace('+','')}"

def tel_link(phone: str) -> Optional[str]:
    if not phone:
        return None
    return f"tel:{phone.replace(' ','')}"

def safe_int(x) -> Optional[int]:
    if x is None or x == "" or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return int(float(x))
    except Exception:
        return None

def list_intersects(a: List[str], b: List[str]) -> bool:
    return bool(set(a or []) & set(b or []))

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# =========================
# Dedup / repeated detection
# =========================
def listing_fingerprint(lst: Dict[str, Any]) -> str:
    # “huella” aproximada: zona + tipo + dorm + operación + precio redondeado + keywords
    zone = (lst.get("zone") or "").strip().lower()
    ptype = (lst.get("property_type") or "").strip().lower()
    op = (lst.get("operation") or "").strip().lower()
    dorm = str(lst.get("bedrooms") or "")
    price = safe_int(lst.get("price"))
    price_bucket = ""
    if price:
        # bucket por 10k para agrupar variantes
        price_bucket = str(int(round(price / 10000.0) * 10000))
    kw = ",".join(sorted(lst.get("keywords") or []))
    key = f"{zone}|{ptype}|{op}|{dorm}|{price_bucket}|{kw}"
    return sha1(key)

def compute_repeats(listings: List[Dict[str, Any]]) -> Dict[str, int]:
    fp_counts: Dict[str, int] = {}
    for lst in listings:
        fp = listing_fingerprint(lst)
        fp_counts[fp] = fp_counts.get(fp, 0) + 1
    return fp_counts

# =========================
# Zone average estimation (median)
# =========================
def compute_zone_stats(listings_df: pd.DataFrame) -> pd.DataFrame:
    # usamos USD solamente para “margen vs zona” (en UYU no tiene sentido mezclar)
    if listings_df.empty:
        return pd.DataFrame()

    df = listings_df.copy()
    df = df[(df["price_currency"] == "USD") & df["price"].notna()].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] > 1000].copy()

    # columnas clave
    df["zone"] = df["zone"].fillna("")
    df["property_type"] = df["property_type"].fillna("Desconocido")
    df["operation"] = df["operation"].fillna("unknown")
    df["bedrooms"] = df["bedrooms"].fillna(-1)

    # mediana por (zona, tipo, dorm, operación)
    stats = df.groupby(["zone", "property_type", "bedrooms", "operation"], dropna=False)["price"].median().reset_index()
    stats = stats.rename(columns={"price": "zone_median_price"})
    return stats

def attach_margin(listings_df: pd.DataFrame, zone_stats: pd.DataFrame) -> pd.DataFrame:
    if listings_df.empty:
        return listings_df

    df = listings_df.copy()
    df["price_num"] = pd.to_numeric(df["price"], errors="coerce")
    df["bedrooms_key"] = df["bedrooms"].fillna(-1)
    df["operation"] = df["operation"].fillna("unknown")
    df["zone"] = df["zone"].fillna("")
    df["property_type"] = df["property_type"].fillna("Desconocido")

    if zone_stats.empty:
        df["zone_median_price"] = None
        df["margin_pct"] = None
        return df

    merged = df.merge(
        zone_stats,
        left_on=["zone", "property_type", "bedrooms_key", "operation"],
        right_on=["zone", "property_type", "bedrooms", "operation"],
        how="left",
        suffixes=("", "_y")
    )
    merged["zone_median_price"] = merged["zone_median_price"]
    # margen = (mediana - precio) / mediana
    merged["margin_pct"] = None
    mask = (merged["price_currency"] == "USD") & merged["price_num"].notna() & merged["zone_median_price"].notna() & (merged["zone_median_price"] > 0)
    merged.loc[mask, "margin_pct"] = (merged.loc[mask, "zone_median_price"] - merged.loc[mask, "price_num"]) / merged.loc[mask, "zone_median_price"]
    return merged

# =========================
# MATCHING ENGINE (scoring + reasons)
# =========================
def money_compatible(lead: Dict[str, Any], listing: Dict[str, Any]) -> Tuple[bool, str]:
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

    bd = safe_int(bd)
    bmin = safe_int(bmin)
    bmax = safe_int(bmax)

    if bd is None:
        return True, "dorm faltante (no bloquea)"
    if bmin is None:
        return True, "lead sin dorm (no bloquea)"

    if bmax is not None:
        return (bmin <= bd <= bmax), "dorm dentro de rango" if (bmin <= bd <= bmax) else "dorm fuera de rango"
    return (bd >= bmin), "dorm ok" if (bd >= bmin) else "dorm insuficiente"

def type_compatible(lead: Dict[str, Any], listing: Dict[str, Any]) -> Tuple[bool, str]:
    lt = (lead.get("property_type") or "").lower()
    pt = (listing.get("property_type") or "").lower()
    if not lt or lt == "desconocido" or not pt or pt == "desconocido":
        return True, "tipo faltante (no bloquea)"
    if lt == pt:
        return True, "tipo coincide"
    if ("apartamento" in lt and "penthouse" in pt) or ("penthouse" in lt and "apartamento" in pt):
        return True, "tipo similar"
    return False, "tipo no coincide"

def kw_overlap(lead: Dict[str, Any], listing: Dict[str, Any]) -> bool:
    return bool(set(lead.get("keywords") or []) & set(listing.get("keywords") or []))

def match_score(lead: Dict[str, Any], listing: Dict[str, Any], repeats_count: int = 0) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []

    # operación
    lo = lead.get("operation")
    po = listing.get("operation")
    if lo != "unknown" and po != "unknown":
        if lo != po:
            return 0, ["operación distinta"]
        score += 10
        reasons.append("operación coincide")

    # zona
    lz = lead.get("zones") or []
    pz = listing.get("zones") or []
    if listing.get("zone"):
        pz = list(set(pz + [listing.get("zone")]))
    if list_intersects(lz, pz):
        score += 30
        reasons.append("zona coincide")
    else:
        score -= 10
        reasons.append("zona no coincide")

    # dinero
    ok_money, why_money = money_compatible(lead, listing)
    if ok_money:
        score += 25
    else:
        score -= 32
    reasons.append(why_money)

    # dorm
    ok_dorm, why_dorm = dorm_compatible(lead, listing)
    if ok_dorm:
        score += 15
    else:
        score -= 22
    reasons.append(why_dorm)

    # tipo
    ok_type, why_type = type_compatible(lead, listing)
    if ok_type:
        score += 15
    else:
        score -= 26
    reasons.append(why_type)

    # keywords edificio
    if kw_overlap(lead, listing):
        score += 18
        reasons.append("keyword/edificio coincide")

    # urgencia lead
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

    # “Repetidos” (si listing aparece repetido, puede ser negociable pero también ruido)
    if repeats_count >= 3:
        score += 6
        reasons.append(f"repetido x{repeats_count} (posible negociación)")

    return max(0, min(100, int(score))), reasons

def compute_matches_for_lead(lead: Dict[str, Any], listings: List[Dict[str, Any]], repeats_map: Dict[str, int], topn=15) -> List[Dict[str, Any]]:
    out = []
    for lst in listings:
        fp = listing_fingerprint(lst)
        rep = repeats_map.get(fp, 1)
        s, reasons = match_score(lead, lst, repeats_count=rep)
        if s <= 0:
            continue
        out.append({"score": s, "reasons": reasons, "repeats": rep, "listing": lst})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:topn]

def compute_matches_for_listing(listing: Dict[str, Any], leads: List[Dict[str, Any]], repeats_map: Dict[str, int], topn=15) -> List[Dict[str, Any]]:
    out = []
    fp = listing_fingerprint(listing)
    rep = repeats_map.get(fp, 1)
    for ld in leads:
        s, reasons = match_score(ld, listing, repeats_count=rep)
        if s <= 0:
            continue
        out.append({"score": s, "reasons": reasons, "repeats": rep, "lead": ld})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:topn]

# =========================
# PRIORITY SCORE (operativo)
# =========================
def lead_priority_score(lead: Dict[str, Any], conn: sqlite3.Connection) -> Tuple[int, List[str]]:
    """
    Score para “Prioridad HOY” (operativo):
    - urgencia (0..12)
    - sin contacto aún (bonus)
    - confianza (bonus)
    - dueño directo (bonus)
    - recencia (bonus)
    """
    s = 0
    reasons = []
    urg = int(lead.get("urgency") or 0)
    s += urg * 4
    if urg >= 2:
        reasons.append("urgente")

    conf = float(lead.get("confidence") or 0)
    if conf >= 0.6:
        s += 8
        reasons.append("confianza alta")
    elif conf < 0.35:
        s -= 5
        reasons.append("confianza baja")

    actor = lead.get("actor")
    if actor == "owner":
        s += 10
        reasons.append("dueño directo")

    status = db_get_funnel(conn, lead["lead_id"])
    if status in ["nuevo"]:
        s += 10
        reasons.append("sin contacto")
    elif status in ["contactado", "respondió"]:
        s += 4
        reasons.append("en seguimiento")
    elif status in ["visita", "reserva"]:
        s += 2
        reasons.append("pipeline avanzado")
    elif status in ["perdido", "cierre"]:
        s -= 50
        reasons.append("cerrado")

    # recencia
    ts = pd.Timestamp(lead.get("ts")).to_pydatetime() if lead.get("ts") is not None else None
    if ts:
        age_h = max(0.0, (datetime.now(tz=DEFAULT_TZ) - ts).total_seconds() / 3600.0)
        if age_h <= 24:
            s += 8
            reasons.append("últimas 24h")
        elif age_h <= 72:
            s += 4
            reasons.append("últimas 72h")

    return max(0, min(100, int(s))), reasons

# =========================
# UI building blocks
# =========================
def badge(text: str, kind: str = "info") -> str:
    # CSS badges
    colors = {
        "info": ("#E8F1FF", "#1B4D9B"),
        "ok": ("#EAFBF0", "#0F7A3E"),
        "warn": ("#FFF3DF", "#8A4B00"),
        "hot": ("#FFE8E8", "#9B1B1B"),
        "muted": ("#EEF0F4", "#394150"),
    }
    bg, fg = colors.get(kind, colors["info"])
    return f"<span style='display:inline-block;padding:3px 8px;border-radius:999px;background:{bg};color:{fg};font-size:12px;font-weight:700;margin-right:6px;'>{text}</span>"

def card_container_start():
    st.markdown("<div style='border:1px solid #E5E7EB;border-radius:16px;padding:12px;background:white;margin-bottom:10px;'>", unsafe_allow_html=True)

def card_container_end():
    st.markdown("</div>", unsafe_allow_html=True)

def topbar_css():
    st.markdown("""
<style>
/* General spacing */
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E5E7EB; padding: 14px; border-radius: 16px; }
</style>
""", unsafe_allow_html=True)

# =========================
# APP START
# =========================
st.set_page_config(page_title="BrokerOS", layout="wide")
topbar_css()

conn = db()
db_init(conn)

st.title(APP_NAME)
st.caption("Dashboard operativo estilo SaaS • Matching bidireccional • Oportunidades ocultas • Embudo • Ingesta")

# -------------------------
# SOURCE INPUT (sidebar)
# -------------------------
with st.sidebar:
    st.header("Fuente de datos")
    load_mode = st.radio("Cargar desde", ["Archivo en repo", "Pegar texto"], index=0)

    txt = ""
    source_label = ""
    if load_mode == "Archivo en repo":
        filename = st.text_input("Nombre del .txt", value=DEFAULT_TXT_FILENAME)
        try:
            with open(filename, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()
            st.success(f"OK: {filename}")
            source_label = f"file:{filename}"
        except Exception:
            st.error("No pude abrir el archivo. Subilo al repo y confirmá el nombre.")
    else:
        txt = st.text_area("Pegá el export de WhatsApp", height=220)
        source_label = "paste"

    st.divider()

    st.subheader("Filtros globales")
    only_today = st.checkbox("Fecha: Hoy", value=True)
    min_conf = st.slider("Confianza mínima", 0.0, 1.0, 0.30, 0.05)
    only_with_phone = st.checkbox("Solo con teléfono", value=True)
    only_owner = st.checkbox("Dueño directo", value=False)
    only_high_margin = st.checkbox("Solo alto margen (≥20%)", value=False)
    only_repeats = st.checkbox("Solo repetidos (3+)", value=False)

    st.divider()
    nav = st.radio("Secciones", ["Resumen", "Prioridad HOY", "Oportunidades ocultas", "Embudo", "Rendimiento", "Datos", "Monitoreo de Ingesta"], index=1)

if not txt.strip():
    st.warning("Subí el .txt al repo o pegá texto en el sidebar.")
    st.stop()

# -------------------------
# PARSE + BUILD
# -------------------------
df_msgs = parse_whatsapp_export(txt)
df_msgs = df_msgs[df_msgs["msg_type"] != "SYSTEM"].copy()

# filtro “hoy”
if only_today and len(df_msgs):
    today = datetime.now(tz=DEFAULT_TZ).date()
    df_msgs = df_msgs[df_msgs["ts"].dt.date == today].copy()

df_leads = build_leads(df_msgs)
df_listings = build_listings(df_msgs)

# persist ingest run
db_add_ingest(conn, source_label, int(len(df_msgs)), int(len(df_leads)), int(len(df_listings)))

# Apply overrides
leads_list: List[Dict[str, Any]] = []
for r in df_leads.to_dict(orient="records") if len(df_leads) else []:
    patch = db_get_json(conn, "overrides", r["lead_id"])
    leads_list.append(apply_overrides(r, patch))

listings_list: List[Dict[str, Any]] = []
for r in df_listings.to_dict(orient="records") if len(df_listings) else []:
    patch = db_get_json(conn, "overrides", r["listing_id"])
    listings_list.append(apply_overrides(r, patch))

# repeats
repeats_map = compute_repeats(listings_list)

# zone stats + margin
zone_stats = compute_zone_stats(pd.DataFrame(listings_list) if listings_list else pd.DataFrame())
df_listings_aug = attach_margin(pd.DataFrame(listings_list) if listings_list else pd.DataFrame(), zone_stats)
if not df_listings_aug.empty:
    # volcar de nuevo a dict con campos extra
    listings_aug = df_listings_aug.to_dict(orient="records")
else:
    listings_aug = listings_list

# -------------------------
# GLOBAL FILTERS APPLY
# -------------------------
def pass_common_filters_lead(ld: Dict[str, Any]) -> bool:
    if float(ld.get("confidence") or 0) < min_conf:
        return False
    if only_with_phone and not ld.get("phone"):
        return False
    if only_owner and ld.get("actor") != "owner":
        return False
    return True

def pass_common_filters_listing(lst: Dict[str, Any]) -> bool:
    if float(lst.get("confidence") or 0) < min_conf:
        return False
    if only_with_phone and not lst.get("lister_phone"):
        # listing puede venir sin phone; no siempre cortamos, pero el filtro lo pide
        return False
    if only_owner and lst.get("actor") != "owner":
        return False
    if only_high_margin:
        mp = lst.get("margin_pct")
        if mp is None or (isinstance(mp, float) and pd.isna(mp)) or mp < 0.20:
            return False
    if only_repeats:
        rep = repeats_map.get(listing_fingerprint(lst), 1)
        if rep < 3:
            return False
    return True

leads_view = [x for x in leads_list if pass_common_filters_lead(x)]
listings_view = [x for x in listings_aug if pass_common_filters_listing(x)]

# -------------------------
# KPIs (top)
# -------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Leads", int((df_msgs["msg_type"] == "LEAD_REQUEST").sum()))
k2.metric("Listings", int((df_msgs["msg_type"] == "LISTING").sum()))
k3.metric("Matches (estim)", "—" if not (leads_view and listings_view) else str(min(len(leads_view) * 3, 9999)))
high_margin_count = 0
if listings_aug:
    for x in listings_aug:
        mp = x.get("margin_pct")
        if mp is not None and not (isinstance(mp, float) and pd.isna(mp)) and mp >= 0.20 and x.get("price_currency") == "USD":
            high_margin_count += 1
k4.metric("Alto margen (≥20%)", high_margin_count)

st.divider()

# -------------------------
# Selection state
# -------------------------
if "selected_lead_id" not in st.session_state:
    st.session_state["selected_lead_id"] = None
if "selected_listing_id" not in st.session_state:
    st.session_state["selected_listing_id"] = None

def select_lead(lid: str):
    st.session_state["selected_lead_id"] = lid
    st.session_state["selected_listing_id"] = None

def select_listing(pid: str):
    st.session_state["selected_listing_id"] = pid
    st.session_state["selected_lead_id"] = None

def get_lead_by_id(lid: str) -> Optional[Dict[str, Any]]:
    return next((x for x in leads_list if x["lead_id"] == lid), None)

def get_listing_by_id(pid: str) -> Optional[Dict[str, Any]]:
    # ojo: listings_aug tiene extras. buscamos ahí primero
    return next((x for x in listings_aug if x["listing_id"] == pid), None)

# -------------------------
# Layout: 3-column (como tu mock)
# -------------------------
col_left, col_center, col_right = st.columns([0.95, 1.35, 0.85])

# =========================
# LEFT: Filters summary + quick funnel stats
# =========================
with col_left:
    st.subheader("🎛️ Filtros / Patrones")

    # mini resumen de funnel de leads visibles
    funnel_counts = {s: 0 for s in FUNNEL_STATES}
    for ld in leads_view:
        funnel_counts[db_get_funnel(conn, ld["lead_id"])] += 1

    card_container_start()
    st.markdown("**Embudo (leads filtrados)**", unsafe_allow_html=True)
    for s in ["nuevo", "contactado", "respondió", "visita", "cierre", "perdido"]:
        st.write(f"- **{s}**: {funnel_counts.get(s,0)}")
    card_container_end()

    # repeats snapshot
    rep_active = sum(1 for fp, c in repeats_map.items() if c >= 3)
    card_container_start()
    st.markdown("**Repetidos activos (3+)**", unsafe_allow_html=True)
    st.write(f"Huella con repeticiones: **{rep_active}**")
    card_container_end()

    # quick tips
    card_container_start()
    st.markdown("**Acciones rápidas sugeridas**")
    st.write("1) Contactar *Nuevos* + *Urgentes*")
    st.write("2) Revisar *Alto margen* en USD")
    st.write("3) Repetidos 3+ → posible negociación")
    card_container_end()

# =========================
# CENTER: Main (depends on nav) – list + detail
# =========================
with col_center:
    if nav == "Resumen":
        st.subheader("📌 Resumen ejecutivo")
        card_container_start()
        st.markdown(
            f"{badge('HOY', 'info')}"
            f"{badge(f'Leads: {len(leads_view)}', 'ok')}"
            f"{badge(f'Listings: {len(listings_view)}', 'muted')}"
            f"{badge(f'Alto margen: {high_margin_count}', 'warn')}",
            unsafe_allow_html=True
        )
        st.write("Este panel está optimizado para **priorización + seguimiento**. Usá “Prioridad HOY” para operar.")
        st.write("Usá “Oportunidades ocultas” para detectar **precios anómalos (≥20% debajo de mediana)**.")
        card_container_end()

        st.write("### Estado de datos")
        st.write(f"- Mensajes parseados: **{len(df_msgs)}**")
        st.write(f"- Leads extraídos: **{len(df_leads)}**")
        st.write(f"- Listings extraídos: **{len(df_listings)}**")
        st.write(f"- Zone stats (median): **{len(zone_stats)}** combinaciones")

    elif nav == "Prioridad HOY":
        st.subheader("🔥 Prioridad HOY")

        # rank leads
        ranked = []
        for ld in leads_view:
            score, reasons = lead_priority_score(ld, conn)
            if score <= 0:
                continue
            ranked.append((score, reasons, ld))
        ranked.sort(key=lambda x: x[0], reverse=True)

        left_list, right_detail = st.columns([1.0, 1.05])

        with left_list:
            st.markdown("**Lista operable**")
            if not ranked:
                st.info("No hay leads con estos filtros.")
            for score, reasons, ld in ranked[:40]:
                status = db_get_funnel(conn, ld["lead_id"])
                zones_txt = ", ".join(ld.get("zones") or []) or "—"
                money = money_fmt(ld.get("budget_currency"), ld.get("budget_max"))
                dorm = ld.get("bedrooms_min") if ld.get("bedrooms_min") is not None else "—"
                actor = ld.get("actor")

                card_container_start()
                st.markdown(
                    f"{badge('HOT','hot') if score>=80 else badge('OK','ok')}"
                    f"{badge(f'Score {score}','info')}"
                    f"{badge(status,'muted')}"
                    f"{badge('Dueño','ok') if actor=='owner' else ''}"
                    f"{badge('Urgente','warn') if int(ld.get('urgency') or 0)>=2 else ''}",
                    unsafe_allow_html=True
                )
                st.markdown(f"**{ld.get('property_type')}** • **{money}** • Dorm: **{dorm}**")
                st.caption(f"Zonas: {zones_txt}")
                st.caption(f"Razones: {', '.join(reasons[:4])}")
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button("Ver", key=f"pl_view_{ld['lead_id']}"):
                        select_lead(ld["lead_id"])
                with c2:
                    link = wa_link(ld.get("phone"))
                    if link:
                        st.markdown(f"[WhatsApp]({link})")
                with c3:
                    if st.button("Contactado", key=f"pl_contact_{ld['lead_id']}"):
                        db_set_funnel(conn, ld["lead_id"], "contactado")
                        db_add_event(conn, "status_contactado", ld["lead_id"], None, {})
                        st.toast("Marcado: contactado ✅")
                card_container_end()

        with right_detail:
            lid = st.session_state["selected_lead_id"]
            if not lid and ranked:
                lid = ranked[0][2]["lead_id"]
                select_lead(lid)

            lead = get_lead_by_id(lid) if lid else None
            if not lead:
                st.info("Seleccioná un lead para ver el detalle.")
            else:
                status = db_get_funnel(conn, lead["lead_id"])
                zones_txt = ", ".join(lead.get("zones") or []) or "—"
                money = money_fmt(lead.get("budget_currency"), lead.get("budget_max"))
                dorm = lead.get("bedrooms_min") if lead.get("bedrooms_min") is not None else "—"

                card_container_start()
                st.markdown("### Detalle del Lead")
                st.markdown(f"**Tel:** {lead.get('phone')}  \n**Estado:** `{status}`")
                st.markdown(f"**Tipo:** {lead.get('property_type')} • **Operación:** {lead.get('operation')} • **Dorm:** {dorm}")
                st.markdown(f"**Presupuesto:** {money}  \n**Zonas:** {zones_txt}")
                if lead.get("keywords"):
                    st.markdown(f"**Keywords:** {', '.join(lead['keywords'])}")
                card_container_end()

                # acciones
                c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
                with c1:
                    tlink = tel_link(lead.get("phone"))
                    if tlink:
                        st.markdown(f"[📞 Llamar]({tlink})")
                with c2:
                    wlink = wa_link(lead.get("phone"))
                    if wlink:
                        st.markdown(f"[💬 WhatsApp]({wlink})")
                with c3:
                    new_status = st.selectbox("Embudo", FUNNEL_STATES, index=FUNNEL_STATES.index(status) if status in FUNNEL_STATES else 0, key="lead_status")
                with c4:
                    if st.button("Guardar estado", key="lead_status_save"):
                        db_set_funnel(conn, lead["lead_id"], new_status)
                        db_add_event(conn, "status_change", lead["lead_id"], None, {"status": new_status})
                        st.success("OK")

                note = st.text_area("Notas", value=db_get_note(conn, lead["lead_id"]), height=90, key="lead_note")
                if st.button("Guardar nota", key="lead_note_save"):
                    db_set_note(conn, lead["lead_id"], note)
                    st.success("Nota guardada")

                # plantilla
                template = (
                    f"Hola! Vi tu búsqueda de {lead.get('property_type','propiedad')} "
                    f"({dorm} dorm) por {zones_txt}. ¿Sigue vigente? "
                    f"Tengo 2-3 opciones dentro de {money}. ¿Te envío ficha y coordinamos visita?"
                )
                st.text_area("Plantilla WhatsApp", value=template, height=100)

                # overrides
                with st.expander("⚙️ Corregir datos detectados (override)"):
                    oz = st.text_input("Zonas (coma)", value=", ".join(lead.get("zones") or []))
                    oc = st.selectbox("Moneda", ["", "USD", "UYU"], index=["", "USD", "UYU"].index(lead.get("budget_currency") or ""))
                    ob = st.text_input("Presupuesto max", value=str(lead.get("budget_max") or ""))
                    od = st.text_input("Dorm min", value=str(lead.get("bedrooms_min") or ""))
                    ot = st.text_input("Tipo", value=str(lead.get("property_type") or ""))
                    oop = st.selectbox("Operación", ["buy", "rent", "unknown"], index=["buy","rent","unknown"].index(lead.get("operation") or "unknown"))

                    if st.button("Aplicar override", key="apply_ov_lead"):
                        patch = {
                            "zones": [z.strip() for z in oz.split(",") if z.strip()],
                            "budget_currency": oc or None,
                            "budget_max": safe_int(ob) if safe_int(ob) is not None else (ob.strip() or None),
                            "bedrooms_min": safe_int(od),
                            "property_type": ot or None,
                            "operation": oop or None,
                        }
                        db_set_json(conn, "overrides", lead["lead_id"], patch)
                        st.success("Override guardado. Recargá/seleccioná de nuevo para ver cambios.")

                # matching top
                st.markdown("### 🔁 Top Listings compatibles")
                matches = compute_matches_for_lead(lead, listings_view, repeats_map, topn=10)
                if not matches:
                    st.warning("No encontré matches con reglas actuales.")
                for m in matches:
                    lst = m["listing"]
                    price = money_fmt(lst.get("price_currency"), lst.get("price"))
                    zone = lst.get("zone") or (", ".join(lst.get("zones") or []) or "zona?")
                    rep = m["repeats"]
                    margin = lst.get("margin_pct")
                    margin_txt = ""
                    if margin is not None and not (isinstance(margin, float) and pd.isna(margin)):
                        margin_txt = f" • margen vs zona: {int(margin*100)}%"

                    card_container_start()
                    st.markdown(
                        f"{badge(f'Score {m['score']}', 'info')}"
                        f"{badge(f'Rep x{rep}','warn') if rep>=3 else ''}"
                        f"{badge('Alto margen','hot') if (margin is not None and not pd.isna(margin) and margin>=0.2) else ''}",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**{price}** • **{lst.get('property_type')}** • {zone}{margin_txt}")
                    st.caption("Razones: " + ", ".join(m["reasons"][:5]))
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        if st.button("Ver listing", key=f"view_lst_{lst['listing_id']}"):
                            select_listing(lst["listing_id"])
                    with c2:
                        if st.button("Marcar enviado", key=f"sent_{lst['listing_id']}"):
                            db_add_event(conn, "sent_listing_to_lead", lead["lead_id"], lst["listing_id"], {"score": m["score"]})
                            st.toast("Registrado: enviado ✅")
                    card_container_end()

                st.markdown("**Mensaje original**")
                st.code(lead.get("raw_text", ""), language="text")

    elif nav == "Oportunidades ocultas":
        st.subheader("🧪 Oportunidades ocultas (alto margen)")
        st.caption("Detecta listings en **USD** con precio ≥20% por debajo de la mediana calculada por (zona, tipo, dorm, operación).")

        # ranking por margin_pct
        opps = []
        for lst in listings_view:
            mp = lst.get("margin_pct")
            if mp is None or (isinstance(mp, float) and pd.isna(mp)):
                continue
            if lst.get("price_currency") != "USD":
                continue
            opps.append(lst)
        opps.sort(key=lambda x: (x.get("margin_pct") or 0), reverse=True)

        if not opps:
            st.info("No hay oportunidades con datos suficientes. Tip: necesitás varios listings por zona para que la mediana tenga sentido.")
        else:
            left_list, right_detail = st.columns([1.0, 1.05])

            with left_list:
                for lst in opps[:50]:
                    mp = lst.get("margin_pct") or 0
                    if mp < 0.20:
                        continue
                    price = money_fmt(lst.get("price_currency"), lst.get("price"))
                    zone = lst.get("zone") or "—"
                    dorm = lst.get("bedrooms") or "—"
                    rep = repeats_map.get(listing_fingerprint(lst), 1)
                    actor = lst.get("actor")
                    status = db_get_funnel(conn, lst["listing_id"])

                    card_container_start()
                    st.markdown(
                        f"{badge(f'-{int(mp*100)}% vs zona','hot')}"
                        f"{badge(status,'muted')}"
                        f"{badge(f'Rep x{rep}','warn') if rep>=3 else ''}"
                        f"{badge('Dueño','ok') if actor=='owner' else ''}",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**{price}** • {lst.get('property_type')} • {zone} • dorm {dorm}")
                    st.caption("Keywords: " + (", ".join(lst.get("keywords") or []) or "—"))
                    c1, c2 = st.columns([1,1])
                    with c1:
                        if st.button("Ver", key=f"opp_view_{lst['listing_id']}"):
                            select_listing(lst["listing_id"])
                    with c2:
                        if lst.get("lister_phone"):
                            st.markdown(f"[WhatsApp]({wa_link(lst['lister_phone'])})")
                    card_container_end()

            with right_detail:
                pid = st.session_state["selected_listing_id"]
                if not pid and opps:
                    # primer opp real (>=20)
                    for o in opps:
                        if (o.get("margin_pct") or 0) >= 0.20:
                            pid = o["listing_id"]
                            select_listing(pid)
                            break

                listing = get_listing_by_id(pid) if pid else None
                if not listing:
                    st.info("Seleccioná un listing para ver detalle.")
                else:
                    mp = listing.get("margin_pct")
                    price = money_fmt(listing.get("price_currency"), listing.get("price"))
                    zone = listing.get("zone") or "—"
                    dorm = listing.get("bedrooms") or "—"
                    med = listing.get("zone_median_price")
                    med_txt = money_fmt("USD", med) if med is not None and not (isinstance(med, float) and pd.isna(med)) else "—"

                    card_container_start()
                    st.markdown("### Detalle oportunidad")
                    st.markdown(f"**{listing.get('property_type')}** • zona **{zone}** • dorm **{dorm}**")
                    st.markdown(f"**Precio:** {price}  \n**Mediana zona:** {med_txt}")
                    if mp is not None and not (isinstance(mp, float) and pd.isna(mp)):
                        st.markdown(f"**Descuento vs zona:** **-{int(mp*100)}%**")
                    rep = repeats_map.get(listing_fingerprint(listing), 1)
                    st.markdown(f"Repetición: **x{rep}**")
                    card_container_end()

                    # overrides
                    with st.expander("⚙️ Corregir datos (override listing)"):
                        oz = st.text_input("Zona", value=str(listing.get("zone") or ""))
                        oc = st.selectbox("Moneda", ["", "USD", "UYU"], index=["", "USD", "UYU"].index(listing.get("price_currency") or ""))
                        op = st.text_input("Precio", value=str(listing.get("price") or ""))
                        od = st.text_input("Dormitorios", value=str(listing.get("bedrooms") or ""))
                        ot = st.text_input("Tipo", value=str(listing.get("property_type") or ""))
                        oop = st.selectbox("Operación", ["buy", "rent", "unknown"], index=["buy","rent","unknown"].index(listing.get("operation") or "unknown"))

                        if st.button("Aplicar override", key="apply_ov_lst_opp"):
                            patch = {
                                "zone": oz or None,
                                "price_currency": oc or None,
                                "price": safe_int(op) if safe_int(op) is not None else (op.strip() or None),
                                "bedrooms": safe_int(od),
                                "property_type": ot or None,
                                "operation": oop or None,
                            }
                            db_set_json(conn, "overrides", listing["listing_id"], patch)
                            st.success("Override guardado. Recargá/seleccioná de nuevo para recalcular.")

                    # matching inverso
                    st.markdown("### 🎯 Leads compatibles con esta oportunidad")
                    matches = compute_matches_for_listing(listing, leads_view, repeats_map, topn=10)
                    if not matches:
                        st.warning("No hay leads compatibles con reglas actuales.")
                    for m in matches:
                        ld = m["lead"]
                        money = money_fmt(ld.get("budget_currency"), ld.get("budget_max"))
                        z = ", ".join(ld.get("zones") or []) or "—"
                        status = db_get_funnel(conn, ld["lead_id"])
                        card_container_start()
                        st.markdown(f"{badge(f'Score {m['score']}', 'info')}{badge(status,'muted')}", unsafe_allow_html=True)
                        st.markdown(f"**{money}** • {ld.get('property_type')} • {ld.get('phone')}")
                        st.caption(f"Zonas: {z}")
                        st.caption("Razones: " + ", ".join(m["reasons"][:5]))
                        c1, c2 = st.columns([1,1])
                        with c1:
                            if st.button("Ver lead", key=f"opp_ld_{ld['lead_id']}"):
                                select_lead(ld["lead_id"])
                        with c2:
                            st.markdown(f"[WhatsApp]({wa_link(ld['phone'])})")
                        card_container_end()

                    st.markdown("**Mensaje listing original**")
                    st.code(listing.get("raw_text",""), language="text")

    elif nav == "Embudo":
        st.subheader("🧱 Embudo (Kanban simple)")
        st.caption("Arrastrar no está en Streamlit nativo sin libs extra, pero esto ya sirve para operar: cambiá estado y hacé seguimiento.")

        cols = st.columns(len(["nuevo","contactado","respondió","visita","reserva","cierre","perdido"]))
        stages = ["nuevo","contactado","respondió","visita","reserva","cierre","perdido"]

        # separamos leads por estado
        by_stage: Dict[str, List[Dict[str, Any]]] = {s: [] for s in stages}
        for ld in leads_view:
            by_stage[db_get_funnel(conn, ld["lead_id"])].append(ld)

        for i, stg in enumerate(stages):
            with cols[i]:
                st.markdown(f"### {stg}")
                items = by_stage.get(stg, [])
                # orden por ts desc
                items.sort(key=lambda x: x.get("ts"), reverse=True)
                for ld in items[:20]:
                    money = money_fmt(ld.get("budget_currency"), ld.get("budget_max"))
                    card_container_start()
                    st.markdown(f"**{ld.get('property_type')}** • {money}")
                    st.caption(ld.get("phone"))
                    if st.button("Abrir", key=f"kb_{stg}_{ld['lead_id']}"):
                        select_lead(ld["lead_id"])
                    card_container_end()

        st.divider()
        st.subheader("Detalle rápido (selección)")
        lid = st.session_state["selected_lead_id"]
        if not lid:
            st.info("Elegí un lead en el tablero para editarlo.")
        else:
            ld = get_lead_by_id(lid)
            if ld:
                status = db_get_funnel(conn, ld["lead_id"])
                st.markdown(f"**{ld.get('phone')}** • `{status}`")
                new_status = st.selectbox("Mover a", FUNNEL_STATES, index=FUNNEL_STATES.index(status), key="kb_move")
                if st.button("Guardar", key="kb_save"):
                    db_set_funnel(conn, ld["lead_id"], new_status)
                    db_add_event(conn, "kanban_move", ld["lead_id"], None, {"to": new_status})
                    st.success("OK")

    elif nav == "Rendimiento":
        st.subheader("📈 Rendimiento")
        st.caption("Basado en eventos registrados (enviado, status_change, etc.). Esto se vuelve poderoso cuando lo uses 1-2 semanas.")

        # últimos eventos
        cur = conn.execute("SELECT ts, event, lead_id, listing_id, payload_json FROM events ORDER BY id DESC LIMIT 200")
        rows = cur.fetchall()
        if not rows:
            st.info("Todavía no hay eventos. Empezá marcando 'enviado', 'contactado', 'visita', etc.")
        else:
            df_ev = pd.DataFrame(rows, columns=["ts","event","lead_id","listing_id","payload"])
            df_ev["ts"] = pd.to_datetime(df_ev["ts"])
            st.dataframe(df_ev, use_container_width=True, hide_index=True)

            # métricas básicas
            sent = int((df_ev["event"] == "sent_listing_to_lead").sum())
            status_changes = int((df_ev["event"] == "status_change").sum())
            st.write(f"- Enviados: **{sent}**")
            st.write(f"- Cambios de estado: **{status_changes}**")

    elif nav == "Datos":
        st.subheader("🗃️ Datos (auditoría)")
        tabs = st.tabs(["Mensajes", "Leads", "Listings", "Zone Stats"])
        with tabs[0]:
            show = df_msgs.copy()
            show["ts"] = show["ts"].dt.strftime("%d/%m %H:%M")
            st.dataframe(show[["ts","msg_type","sender_raw","sender_phone","has_media","text"]], use_container_width=True, hide_index=True)
        with tabs[1]:
            st.dataframe(pd.DataFrame(leads_view), use_container_width=True, hide_index=True)
        with tabs[2]:
            st.dataframe(pd.DataFrame(listings_view), use_container_width=True, hide_index=True)
        with tabs[3]:
            st.dataframe(zone_stats, use_container_width=True, hide_index=True)

    elif nav == "Monitoreo de Ingesta":
        st.subheader("🛰️ Monitoreo de Ingesta")
        last = db_last_ingest(conn)
        if not last:
            st.info("Sin historial de ingesta todavía.")
        else:
            card_container_start()
            st.markdown(f"{badge('Sync automático (simulado)', 'ok')}{badge('Streamlit Cloud', 'muted')}", unsafe_allow_html=True)
            st.write(f"Última ingesta: **{last['ts']}**")
            st.write(f"Fuente: **{last['source']}**")
            st.write(f"Mensajes: **{last['msg_count']}** • Leads: **{last['lead_count']}** • Listings: **{last['listing_count']}**")
            card_container_end()

        st.write("### Salud del pipeline")
        # sin phone
        no_phone_msgs = int(df_msgs["sender_phone"].isna().sum())
        st.write(f"- Mensajes sin teléfono detectable: **{no_phone_msgs}**")
        # ambiguos moneda
        amb_leads = sum(1 for x in leads_list if (x.get("budget_currency") is None or x.get("confidence",0) < 0.35))
        st.write(f"- Leads con moneda/confianza baja: **{amb_leads}**")
        amb_list = sum(1 for x in listings_list if (x.get("price_currency") is None or x.get("confidence",0) < 0.35))
        st.write(f"- Listings con moneda/confianza baja: **{amb_list}**")

        st.write("### Qué sigue para 'sync real'")
        st.info(
            "Para ingestión continua real (sin export manual) necesitás un puente: "
            "WhatsApp Business API / un número con client autorizado / o integración por otro canal. "
            "Acá lo dejamos listo a nivel UI y almacenamiento; el conector lo enchufamos después."
        )

# =========================
# RIGHT: monitor + quick match panel
# =========================
with col_right:
    st.subheader("🧩 Match / Ingesta")

    # Ingest snapshot
    last = db_last_ingest(conn)
    card_container_start()
    st.markdown(f"{badge('Sync activo', 'ok')}{badge('Última corrida', 'muted')}", unsafe_allow_html=True)
    if last:
        st.write(f"⏱️ {last['ts']}")
        st.write(f"📩 {last['msg_count']} msgs • 🧑 {last['lead_count']} leads • 🏠 {last['listing_count']} listings")
    else:
        st.write("Sin datos de ingesta aún.")
    card_container_end()

    # Quick selection detail: if lead selected show its top 5, else if listing selected show its top 5
    lid = st.session_state["selected_lead_id"]
    pid = st.session_state["selected_listing_id"]

    if lid:
        lead = get_lead_by_id(lid)
        if lead:
            st.markdown("### Lead seleccionado")
            st.write(f"**{lead.get('phone')}** • {money_fmt(lead.get('budget_currency'), lead.get('budget_max'))}")
            matches = compute_matches_for_lead(lead, listings_view, repeats_map, topn=5)
            for m in matches:
                lst = m["listing"]
                price = money_fmt(lst.get("price_currency"), lst.get("price"))
                zone = lst.get("zone") or "—"
                mp = lst.get("margin_pct")
                mp_txt = f" (-{int(mp*100)}%)" if mp is not None and not (isinstance(mp, float) and pd.isna(mp)) else ""
                card_container_start()
                st.markdown(f"{badge(f'{m['score']}','info')} {price} • {zone}{mp_txt}", unsafe_allow_html=True)
                st.caption(", ".join(m["reasons"][:3]))
                if st.button("Abrir listing", key=f"rq_open_lst_{lst['listing_id']}"):
                    select_listing(lst["listing_id"])
                card_container_end()

    elif pid:
        listing = get_listing_by_id(pid)
        if listing:
            st.markdown("### Listing seleccionado")
            st.write(f"**{money_fmt(listing.get('price_currency'), listing.get('price'))}** • {listing.get('zone') or '—'}")
            matches = compute_matches_for_listing(listing, leads_view, repeats_map, topn=5)
            for m in matches:
                ld = m["lead"]
                card_container_start()
                st.markdown(f"{badge(f'{m['score']}','info')} {ld.get('phone')} • {money_fmt(ld.get('budget_currency'), ld.get('budget_max'))}", unsafe_allow_html=True)
                st.caption(", ".join(m["reasons"][:3]))
                if st.button("Abrir lead", key=f"rq_open_ld_{ld['lead_id']}"):
                    select_lead(ld["lead_id"])
                card_container_end()

    else:
        st.info("Seleccioná un lead o listing para ver matching rápido acá.")