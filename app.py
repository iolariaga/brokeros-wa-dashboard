import re
from dataclasses import dataclass
from datetime import datetime
from dateutil import tz

import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
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
RENT_HINTS = ["alquiler", "alquilo", "anual", "invernal", "temporal", "quincena", "mensuales", "mes"]

# Formato típico export WhatsApp Android:
# 06/02/2026, 08:31 - Nombre: mensaje
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
]


# =========================
# Helpers
# =========================
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


def parse_ts(d, mo, y, hh, mm):
    return datetime(int(y), int(mo), int(d), int(hh), int(mm), tzinfo=DEFAULT_TZ)


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
        tail = t[rm.start(): rm.end() + 25]
        if "mil" in tail or "k" in tail:
            if a < 1000:
                a *= 1000
            if b < 1000:
                b *= 1000
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


def extract_m2(text: str):
    m = M2_RE.search((text or "").lower())
    return int(m.group(1)) if m else None


def extract_keywords(text: str):
    t = (text or "").lower()
    return [k for k in KEYWORD_BUILDINGS if k in t]


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


def build_leads(df_msgs: pd.DataFrame):
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
        if "urgente" in t or "para ver mañana" in t or "para ver hoy" in t or "cerrar ya" in t:
            urgency += 2
        if "cliente concreto" in t or "cliente activo" in t:
            urgency += 2
        if "para ver" in t:
            urgency += 1

        leads.append({
            "ts": r["ts"],
            "phone": phone,
            "name": name,
            "operation": op,
            "property_type": ptype,
            "bedrooms_min": bmin,
            "bedrooms_max": bmax,
            "budget_currency": (money or {}).get("currency"),
            "budget_max": (money or {}).get("max"),
            "zones": zones,
            "keywords": kw,
            "min_m2": min_m2,
            "pets_required": pets_req,
            "confidence": (money or {}).get("confidence", 0.25),
            "urgency": urgency,
            "raw_text": text[:900],
        })

    if not leads:
        return pd.DataFrame()

    df = pd.DataFrame(leads)
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
        kw = extract_keywords(text)

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
            "keywords": kw,
            "confidence": (money or {}).get("confidence", 0.25),
            "raw_text": text[:900],
        })

    if not listings:
        return pd.DataFrame()

    return pd.DataFrame(listings).sort_values("ts", ascending=False)


def apply_search(df: pd.DataFrame, cols: list[str], query: str):
    if df is None or len(df) == 0 or not query.strip():
        return df
    qq = query.strip().lower()
    mask = False
    for c in cols:
        if c in df.columns:
            mask = mask | df[c].astype(str).str.lower().str.contains(qq, na=False)
    return df[mask].copy()


def wa_link(phone: str | None):
    if not phone:
        return None
    return f"https://wa.me/{phone.replace('+','')}"


def chips_md(items):
    if not items:
        return ""
    safe = []
    for x in items:
        safe.append(str(x).replace("<", "&lt;").replace(">", "&gt;"))
    return "".join([f'<span class="chip">{x}</span>' for x in safe])


def card_box(title, subtitle, body_html, actions_html="", chips_html=""):
    # basic sanitize for title/subtitle
    title = str(title).replace("<", "&lt;").replace(">", "&gt;")
    subtitle = str(subtitle).replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f"""
<div class="bx">
  <div class="bx-title">{title}</div>
  <div class="bx-sub">{subtitle}</div>
  <div class="bx-body">{body_html}</div>
  <div>{chips_html}</div>
  <div class="actionrow">{actions_html}</div>
</div>
""", unsafe_allow_html=True)


# =========================
# UI
# =========================
st.set_page_config(page_title="BrokerOS – WhatsApp", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2.5rem; }
h1 { font-size: 1.55rem !important; }
h2 { font-size: 1.15rem !important; }
h3 { font-size: 1.0rem !important; }
small, .stCaption { color: #9CA3AF; }

.bx {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 14px;
  margin: 10px 0;
  background: rgba(255,255,255,0.03);
}
.bx-title { font-weight: 800; font-size: 1.0rem; }
.bx-sub { color: #9CA3AF; font-size: 0.85rem; margin-top: 3px; }
.bx-body { margin-top: 10px; font-size: 0.95rem; line-height: 1.35; }

.chip { display:inline-block; padding:4px 10px; border-radius:999px;
  border:1px solid rgba(255,255,255,0.10); margin-right:6px; margin-top:6px;
  font-size:0.80rem; color:#E5E7EB; background: rgba(255,255,255,0.02);
}

.actionrow { margin-top: 10px; font-size: 0.95rem; }
hr { border-color: rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

# Sidebar + session state
if "txt" not in st.session_state:
    st.session_state.txt = ""

with st.sidebar:
    st.header("BrokerOS")
    st.caption("Ingesta → extracción → acción rápida")

    mode = st.radio("Fuente de datos", ["Archivo en repo", "Subir archivo", "Pegar texto"], index=0)

    txt_in = ""
    if mode == "Archivo en repo":
        filename = st.text_input("Nombre del .txt", value="inmo_registradas.txt")
        try:
            with open(filename, "r", encoding="utf-8", errors="replace") as f:
                txt_in = f.read()
            st.success(f"OK: {filename}")
        except Exception as e:
            st.error("No pude abrir el archivo.")
            st.caption(str(e))

    elif mode == "Subir archivo":
        up = st.file_uploader("Export de WhatsApp (.txt)", type=["txt"])
        if up:
            txt_in = up.read().decode("utf-8", errors="replace")
            st.success(f"OK: {up.name}")

    else:
        txt_in = st.text_area("Pegá el export", height=180, placeholder="Pegá el texto exportado…")

    st.divider()
    st.subheader("Filtros")
    q = st.text_input("Buscar (tel / nombre / zona)", value="")
    min_conf = st.slider("Confianza mínima", 0.0, 1.0, 0.30, 0.05)
    show_system = st.checkbox("Mostrar SYSTEM", value=False)
    show_other = st.checkbox("Mostrar OTHER", value=False)

    st.divider()
    process = st.button("Procesar", type="primary", use_container_width=True)
    clear = st.button("Limpiar", use_container_width=True)

if clear:
    st.session_state.txt = ""
    st.rerun()

if process and txt_in.strip():
    st.session_state.txt = txt_in

txt = st.session_state.txt

st.title("BrokerOS – WhatsApp Deal Flow (MVP)")
st.caption("Ingesta → extracción → leads/listings + detalle + plantillas WhatsApp (iteración rápida desde el celular)")

if not txt.strip():
    st.info("📌 Elegí una fuente en la barra lateral y tocá **Procesar**.")
    st.stop()

if not HEADER_RE.search(txt):
    st.error("No reconozco el formato del export. Probá exportar el chat como texto (sin multimedia).")
    with st.expander("Ver 25 primeras líneas (debug)"):
        st.code("\n".join(txt.splitlines()[:25]), language="text")
    st.stop()

# Parse + build
df_msgs = parse_whatsapp_export(txt)

if not show_system:
    df_msgs = df_msgs[df_msgs["msg_type"] != "SYSTEM"].copy()
if not show_other:
    df_msgs = df_msgs[df_msgs["msg_type"] != "OTHER"].copy()

df_leads = build_leads(df_msgs)
df_listings = build_listings(df_msgs)

# KPIs + downloads
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mensajes", len(df_msgs))
c2.metric("Leads", int((df_msgs["msg_type"] == "LEAD_REQUEST").sum()))
c3.metric("Listings", int((df_msgs["msg_type"] == "LISTING").sum()))
c4.metric("Media", int((df_msgs["msg_type"] == "MEDIA").sum()))

if len(df_msgs) > 0:
    st.caption(
        f"Rango: {df_msgs['ts'].min().strftime('%d/%m %H:%M')} → {df_msgs['ts'].max().strftime('%d/%m %H:%M')}"
    )

colA, colB = st.columns(2)
with colA:
    st.download_button(
        "⬇️ leads.csv",
        data=(df_leads.to_csv(index=False).encode("utf-8") if len(df_leads) else "phone\n".encode("utf-8")),
        file_name="leads.csv",
        mime="text/csv",
        use_container_width=True
    )
with colB:
    st.download_button(
        "⬇️ listings.csv",
        data=(df_listings.to_csv(index=False).encode("utf-8") if len(df_listings) else "lister_phone\n".encode("utf-8")),
        file_name="listings.csv",
        mime="text/csv",
        use_container_width=True
    )

tabs = st.tabs(["Hoy", "Leads", "Listings", "Raw"])

# =========================
# Tab: Hoy
# =========================
with tabs[0]:
    st.subheader("Prioridad HOY (táctico)")
    if len(df_leads) == 0:
        st.info("No hay leads detectados.")
    else:
        view = df_leads[df_leads["confidence"] >= min_conf].copy()
        view = apply_search(view, ["phone", "name", "raw_text"], q)
        view = view.sort_values(["urgency", "ts"], ascending=[False, False]).head(30)

        for _, r in view.iterrows():
            zones = ", ".join(r["zones"]) if isinstance(r["zones"], list) else ""
            kw = ", ".join(r["keywords"]) if isinstance(r["keywords"], list) else ""
            dorm = r["bedrooms_min"] if pd.notna(r["bedrooms_min"]) else "—"
            money = f"{r.get('budget_currency') or ''} {r.get('budget_max') or ''}".strip()

            tags = []
            if r.get("pets_required") == 1:
                tags.append("Mascota")
            if r.get("urgency", 0) >= 2:
                tags.append("Urgente")
            if kw:
                tags.append("Edificio")

            link = wa_link(r["phone"])
            actions = ""
            if link:
                actions += f'💬 <a href="{link}" target="_blank">Abrir WhatsApp</a> &nbsp; | &nbsp; '
            actions += "📋 Plantilla abajo (tab Leads)"

            body = (
                f"<b>Tipo:</b> {r.get('property_type','—')}<br>"
                f"<b>Dorm:</b> {dorm}<br>"
                f"<b>Zonas:</b> {zones or '—'}<br>"
                f"<b>Keywords:</b> {kw or '—'}"
            )

            card_box(
                title=f"LEAD • {money or '—'}",
                subtitle=f"{r['phone']} · {r.get('name','') or ''}",
                body_html=body,
                chips_html=chips_md(tags),
                actions_html=actions,
            )

# =========================
# Tab: Leads
# =========================
with tabs[1]:
    st.subheader("Leads")
    if len(df_leads) == 0:
        st.info("No hay leads.")
    else:
        view = df_leads[df_leads["confidence"] >= min_conf].copy()
        view = apply_search(view, ["phone", "name", "raw_text"], q)
        view = view.sort_values("ts", ascending=False)

        st.markdown("### Detalle + Plantilla WhatsApp")
        sel = st.selectbox("Elegí un lead", view["phone"].tolist())
        row = view[view["phone"] == sel].iloc[0].to_dict()

        zones = ", ".join(row["zones"]) if isinstance(row["zones"], list) else ""
        dorm = row["bedrooms_min"] if pd.notna(row["bedrooms_min"]) else "—"
        money = f"{row.get('budget_currency') or ''} {row.get('budget_max') or ''}".strip()

        st.code(row.get("raw_text", ""), language="text")

        plantilla = (
            f"Hola {row.get('name','') or ''}! Vi tu búsqueda de {row.get('property_type','propiedad')} "
            f"({dorm} dorm) en {zones}. ¿Sigue vigente? Tengo opciones dentro de {money}. "
            "¿Querés que te mande 2-3 alternativas por acá?"
        ).strip()

        st.text_area("Plantilla", value=plantilla, height=140)

        st.markdown("---")

        for _, r in view.head(50).iterrows():
            zones = ", ".join(r["zones"]) if isinstance(r["zones"], list) else ""
            dorm = r["bedrooms_min"] if pd.notna(r["bedrooms_min"]) else "—"
            money = f"{r.get('budget_currency') or ''} {r.get('budget_max') or ''}".strip()

            link = wa_link(r["phone"])
            body = (
                f"<b>Tipo:</b> {r.get('property_type','—')}<br>"
                f"<b>Dorm:</b> {dorm}<br>"
                f"<b>Zonas:</b> {zones or '—'}<br>"
                f"<b>Conf:</b> {r.get('confidence',0):.2f}"
            )

            card_box(
                title=f"{money or '—'} • {r.get('property_type','—')}",
                subtitle=f"{r['phone']} · {r.get('name','') or ''}",
                body_html=body,
                actions_html=(f'💬 <a href="{link}" target="_blank">WhatsApp</a>' if link else ""),
            )

# =========================
# Tab: Listings
# =========================
with tabs[2]:
    st.subheader("Listings")
    if len(df_listings) == 0:
        st.info("No hay listings.")
    else:
        view = df_listings[df_listings["confidence"] >= min_conf].copy()
        view = apply_search(view, ["lister_phone", "lister_name", "raw_text"], q)
        view = view.sort_values("ts", ascending=False)

        for _, r in view.iterrows():
            price = f"{r.get('price_currency') or ''} {r.get('price') or ''}".strip()
            zone = r.get("zone_guess") or "Zona ?"
            dorm = r.get("bedrooms") if pd.notna(r.get("bedrooms")) else "—"
            urls = r.get("urls") or []
            kw = ", ".join(r.get("keywords") or [])

            actions = ""
            if urls:
                actions = "<br>".join([f'🔗 <a href="{u}" target="_blank">Abrir link</a>' for u in urls[:3]])

            body = (
                f"<b>Zona:</b> {zone}<br>"
                f"<b>Dorm:</b> {dorm}<br>"
                f"<b>Keywords:</b> {kw or '—'}<br>"
                f"<b>Conf:</b> {r.get('confidence',0):.2f}"
            )

            card_box(
                title=f"{price or '—'} • {r.get('property_type','—')}",
                subtitle=f"Links: {len(urls)} · {r.get('lister_name','') or ''}",
                body_html=body,
                actions_html=actions,
            )

# =========================
# Tab: Raw
# =========================
with tabs[3]:
    st.subheader("Ingesta (raw)")
    show = df_msgs.copy()
    if len(show):
        show["ts"] = show["ts"].dt.strftime("%d/%m %H:%M")
    st.dataframe(
        show[["ts", "msg_type", "sender_raw", "sender_phone", "has_media", "text"]],
        use_container_width=True,
        hide_index=True
    )

    with st.expander("Ver 25 primeras líneas del texto (debug)"):
        st.code("\n".join(txt.splitlines()[:25]), language="text")