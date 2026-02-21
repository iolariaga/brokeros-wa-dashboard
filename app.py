import re
import json
from dataclasses import dataclass
from datetime import datetime
from dateutil import tz
import pandas as pd
import streamlit as st

# ========= CONFIG =========
DEFAULT_TZ = tz.gettz("America/Montevideo")

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
    "rincón del indio": "Rincón del Indio",
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

BUY_HINTS = ["en venta", "para compra", "compra", "vendo", "ofrezco", "tenemos", "en exclusividad"]
RENT_HINTS = ["alquiler", "alquilo", "anual", "invernal", "temporal", "2da quincena", "quincena", "mensuales", "mes"]

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

SYSTEM_HINTS = ["cifrados de extremo a extremo", "creó el grupo", "se te añadió al grupo", "obten más información"]

KEYWORD_BUILDINGS = [
    "torres del este", "gala", "gala puerto", "gala vista", "gala tower",
    "tiburón ii", "green park", "green life", "le parc", "tequendama",
    "tunquelen", "millenium", "trump", "venetian", "ancora", "malecon",
    "miami boulevard", "casino miguez", "marigot", "signature", "citrea",
]

@dataclass
class Message:
    ts: datetime
    sender_raw: str
    text: str


def normalize_phone(s: str):
    m = PHONE_RE.search(s or "")
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}"


def parse_ts(d, m, y, hh, mm):
    # WhatsApp export: no confiamos en a.m./p.m. porque a veces viene con caracteres raros.
    # Tomamos hora tal cual. Si te queda corrido, luego afinamos con heurística.
    dt = datetime(int(y), int(m), int(d), int(hh), int(mm), tzinfo=DEFAULT_TZ)
    return dt


def has_media(text: str):
    t = (text or "").lower()
    return "<multimedia omitido>" in t or "(archivo adjunto)" in t or "archivo adjunto" in t


def classify_message(text: str):
    t = (text or "").lower().strip()
    if any(h in t for h in SYSTEM_HINTS):
        return "SYSTEM"
    if has_media(t):
        return "MEDIA"
    if ("busco" in t) or ("estoy buscando" in t) or ("búsqueda" in t) or ("necesito" in t) or ("preciso" in t):
        return "LEAD_REQUEST"
    if ("ofrezco" in t) or ("tenemos" in t) or ("comparto" in t) or ("alquilo" in t) or ("vendo" in t) or URL_RE.search(t):
        return "LISTING"
    if "reservad" in t or "vendid" in t or "seña" in t:
        return "STATUS"
    return "OTHER"


def detect_operation(text: str):
    t = (text or "").lower()
    if any(h in t for h in RENT_HINTS):
        return "rent"
    if any(h in t for h in BUY_HINTS):
        return "buy"
    if "busco" in t:
        return "buy"
    return "unknown"


def detect_property_type(text: str):
    t = (text or "").lower()
    for k, v in PROPERTY_TYPE_HINTS:
        if k in t:
            return v
    return "Desconocido"


def normalize_zone_list(text: str):
    t = (text or "").lower()
    zones = set()

    # "Zonas: A, B, C"
    z = re.search(r"zonas?\s*:\s*(.+)", t, re.IGNORECASE)
    if z:
        raw = z.group(1)
        for part in re.split(r"[,\n;•]+", raw):
            p = part.strip(" .*-_").lower()
            if not p:
                continue
            zones.add(ZONE_DICT.get(p, p.title()))

    # fallback: scan dict keys
    for k, v in ZONE_DICT.items():
        if k in t:
            zones.add(v)

    return sorted(zones)


def extract_bedrooms(text: str):
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


def extract_money(text: str):
    """
    Devuelve dict: {currency, min, max, confidence}
    """
    t = (text or "").lower()
    currency = None
    if USD_RE.search(t):
        currency = "USD"
    elif "usd" in t:
        currency = "USD"
    elif UYU_RE.search(t):
        currency = "UYU"

    # rangos: "300 - 500 mil"
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

    # Heurística: compra -> mayor número, alquiler -> menor número
    n = max(candidates) if op == "buy" else min(candidates)

    conf = 0.7 if currency else 0.4

    # caso típico: "hasta 35mil" alquiler anual (UYU) sin USD explícito
    if "35mil" in t and not USD_RE.search(t):
        return {"currency": "UYU", "min": None, "max": 35000, "confidence": 0.35}

    # si es alquiler y n está tipo 800/1200 y aparece "mensuales" => USD prob
    if op == "rent" and n <= 10000 and ("usd" in t or "u$s" in t or "dolares" in t or "dólares" in t):
        return {"currency": "USD", "min": None, "max": n, "confidence": max(conf, 0.7)}

    # default
    return {"currency": currency or ("USD" if n >= 5000 else "UYU"), "min": None, "max": n, "confidence": conf}


def extract_m2(text: str):
    m = M2_RE.search((text or "").lower())
    return int(m.group(1)) if m else None


def extract_keywords(text: str):
    t = (text or "").lower()
    kw = []
    for pat in KEYWORD_BUILDINGS:
        if pat in t:
            kw.append(pat)
    return kw


def parse_whatsapp_export(txt: str):
    lines = txt.splitlines()
    messages = []
    current = None

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            if current:
                messages.append(current)
            d, mo, y, hh, mm, sender, text = m.groups()
            ts = parse_ts(d, mo, y, hh, mm)
            current = Message(ts=ts, sender_raw=sender, text=text)
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


def build_leads(df_msgs: pd.DataFrame):
    leads = []
    for _, r in df_msgs[df_msgs["msg_type"] == "LEAD_REQUEST"].iterrows():
        text = r["text"]
        phone = r["sender_phone"]
        if not phone:
            continue

        name = r["sender_raw"] if (r["sender_raw"] and not str(r["sender_raw"]).startswith("+")) else ""
        # firma: última línea no vacía (si parece nombre)
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

        leads.append({
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
            "confidence": (money or {}).get("confidence", 0.35 if money else 0.25),
            "raw_text": text[:800],
        })

    if not leads:
        return pd.DataFrame(columns=[])

    df = pd.DataFrame(leads)
    # última búsqueda por teléfono
    df = df.sort_values("ts", ascending=False).drop_duplicates(subset=["phone"], keep="first")
    return df


def build_listings(df_msgs: pd.DataFrame):
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

        listings.append({
            "ts": r["ts"],
            "lister_phone": phone,
            "lister_name": name,
            "operation": op,
            "property_type": ptype,
            "zone_guess": zone_guess,
            "bedrooms": b,
            "price_currency": (money or {}).get("currency"),
            "price": (money or {}).get("max"),
            "urls": r["urls"],
            "confidence": (money or {}).get("confidence", 0.35 if money else 0.25),
            "raw_text": text[:800],
        })

    if not listings:
        return pd.DataFrame(columns=[])

    df = pd.DataFrame(listings).sort_values("ts", ascending=False)
    return df


# ========= UI =========
st.set_page_config(page_title="BrokerOS – WhatsApp Deal Flow", layout="wide")

st.title("BrokerOS – WhatsApp Deal Flow (MVP)")
st.caption("Ingesta → extracción → leads/listings + detalle + plantillas WhatsApp. (Iteración rápida desde el celular)")

# Fuente TXT
with st.sidebar:
    st.header("Fuente")
    mode = st.radio("¿De dónde cargar?", ["Archivo en repo", "Pegar texto"], index=0)

    txt = ""
    if mode == "Archivo en repo":
        filename = st.text_input("Nombre del .txt", value="inmo_registradas.txt")
        try:
            with open(filename, "r", encoding="utf-8") as f:
                txt = f.read()
            st.success(f"OK: {filename} cargado")
        except Exception as e:
            st.error(f"No pude abrir {filename}. Subilo al repo o corregí el nombre.")
    else:
        txt = st.text_area("Pegá el export acá", height=240)

    st.divider()
    st.header("Filtros rápidos")
    min_conf = st.slider("Confianza mínima", 0.0, 1.0, 0.35, 0.05)
    show_system = st.checkbox("Mostrar SYSTEM", value=False)

if not txt.strip():
    st.info("Subí el .txt al repo (ej: inmo_registradas.txt) o pegá texto en la barra lateral.")
    st.stop()

df_msgs = parse_whatsapp_export(txt)
if not show_system:
    df_msgs = df_msgs[df_msgs["msg_type"] != "SYSTEM"].copy()

df_leads = build_leads(df_msgs)
df_listings = build_listings(df_msgs)

# KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mensajes", len(df_msgs))
c2.metric("Leads", int((df_msgs["msg_type"] == "LEAD_REQUEST").sum()))
c3.metric("Listings", int((df_msgs["msg_type"] == "LISTING").sum()))
c4.metric("Media", int((df_msgs["msg_type"] == "MEDIA").sum()))
c5.metric("Último msg", df_msgs["ts"].max().strftime("%Y-%m-%d %H:%M") if len(df_msgs) else "—")

st.divider()

tab1, tab2, tab3 = st.tabs(["Leads", "Listings", "Ingesta (raw)"])

with tab1:
    st.subheader("Leads detectados (último criterio por teléfono)")
    if len(df_leads) == 0:
        st.warning("No se detectaron leads todavía.")
    else:
        view = df_leads.copy()
        view["zones"] = view["zones"].apply(lambda z: ", ".join(z) if isinstance(z, list) else "")
        view["keywords"] = view["keywords"].apply(lambda z: ", ".join(z) if isinstance(z, list) else "")
        view = view[view["confidence"] >= min_conf].copy()

        st.dataframe(
            view[["ts","phone","name","operation","property_type","bedrooms_min","bedrooms_max","budget_currency","budget_max","zones","keywords","pets_required","confidence"]],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Detalle + Plantilla WhatsApp")
        phones = view["phone"].tolist()
        sel = st.selectbox("Elegí un lead por teléfono", phones)
        row = df_leads[df_leads["phone"] == sel].iloc[0].to_dict()

        zones_txt = ", ".join(row["zones"]) if isinstance(row["zones"], list) else ""
        dorm_txt = str(row["bedrooms_min"]) + (f"–{row['bedrooms_max']}" if row.get("bedrooms_max") else "")
        budget_txt = f"{row.get('budget_currency','')} {row.get('budget_max','')}"
        ptype = row.get("property_type","propiedad")

        colA, colB = st.columns([1.2, 1])
        with colA:
            st.write("**Mensaje (recortado)**")
            st.code(row.get("raw_text",""), language="text")

        with colB:
            st.write("**Acciones**")
            st.write(f"📞 **Tel:** {row['phone']}")
            wa_link = f"https://wa.me/{row['phone'].replace('+','')}"
            st.markdown(f"💬 WhatsApp: {wa_link}")
            st.write("---")
            template = (
                f"Hola {row.get('name','') or ''}! Vi tu búsqueda de {ptype} "
                f"({dorm_txt} dorm) en {zones_txt}. "
                f"¿Sigue vigente? Tengo opciones dentro de {budget_txt}. "
                f"¿Querés que te mande 2-3 alternativas por acá?"
            ).strip()
            st.text_area("Plantilla WhatsApp", value=template, height=140)

with tab2:
    st.subheader("Listings detectados")
    if len(df_listings) == 0:
        st.warning("No se detectaron listings todavía.")
    else:
        view = df_listings.copy()
        view["urls_count"] = view["urls"].apply(lambda u: len(u) if isinstance(u, list) else 0)
        view = view[view["confidence"] >= min_conf].copy()

        st.dataframe(
            view[["ts","lister_phone","lister_name","operation","property_type","zone_guess","bedrooms","price_currency","price","urls_count","confidence"]],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Detalle")
        idx = st.selectbox("Elegí un listing (por timestamp)", view["ts"].astype(str).tolist())
        row = view[view["ts"].astype(str) == idx].iloc[0].to_dict()

        colA, colB = st.columns([1.2, 1])
        with colA:
            st.write("**Mensaje (recortado)**")
            st.code(row.get("raw_text",""), language="text")

        with colB:
            st.write("**Links**")
            for u in (row.get("urls") or []):
                st.markdown(f"- {u}")
            st.write("---")
            st.write("**Siguiente paso:** acá vamos a conectar el botón *Buscar leads compatibles* (matching).")

with tab3:
    st.subheader("Ingesta raw (para auditar parsing)")
    df_show = df_msgs.copy()
    df_show["ts"] = df_show["ts"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        df_show[["ts","msg_type","sender_raw","sender_phone","has_media","text"]],
        use_container_width=True,
        hide_index=True
                 )
