import os, json
import gradio as gr
import pandas as pd
import torch
import spaces
import threading
import re, time, html, math
from datetime import datetime
from zoneinfo import ZoneInfo
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from tavily import TavilyClient
from concurrent.futures import ThreadPoolExecutor

REPO = "skrsathish/skr-rescue-rater"   # use your exact username
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

nemotron = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/",
                  api_key=os.environ["NEBIUS_API_KEY"], timeout=20, max_retries=1)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
MAX_MESSAGES = 10   # protects API credits

@spaces.GPU
def predict(texts):
    enc = tok(texts, padding=True, truncation=True, max_length=96, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=-1)[:, 1].tolist()

def band(p):
    if p >= 0.7: return "URGENT"
    if p >= 0.1: return "NEEDS REVIEW"
    return "ROUTINE"

SYSTEM = ("You read disaster help messages. Reply with JSON only: "
          '{"urgent": true or false, "reason": "trapped|medical|missing|collapse|none", '
          '"location": "town or place, or unknown", '
          '"address": "street address exactly as written in the message, or empty", '
          '"people": "number of people at risk as an integer, or null if not stated", '
          '"vulnerable": "short list such as children, elderly, cannot walk, or empty", '
          '"explanation": "one short sentence", '
          '"reply": "one short text to send back to the sender. If it is an emergency, ask only '
          'for missing facts such as street address or number of people. If it is not an '
          'emergency, say politely that this line is for emergencies. Never ask for a phone '
          'number. Use the same language as the sender. Empty if nothing is needed."}')

def ask_nemotron(message):
    r = nemotron.chat.completions.create(
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": message}])
    text = r.choices[0].message.content
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return {"urgent": True, "reason": "unknown", "location": "unknown",
                "explanation": "Could not read the reply, so a human should check."}

QUERIES = {"trapped":  "flood warning {p} North Carolina",
           "medical":  "shelter or hospital open {p} North Carolina Helene",
           "missing":  "missing persons reunification {p} North Carolina Helene",
           "collapse": "landslide road closure {p} North Carolina"}

def official_alert(place, reason):
    if place == "unknown":
        return "no location given"
    town = place.split(",")[-1].strip()
    q = QUERIES.get(reason, QUERIES["trapped"]).format(p=town)
    searches = [(q, ["weather.gov", "fema.gov", "ncdps.gov", "readync.gov", "buncombecounty.org"]),
                ("Helene flood western North Carolina September 2024", ["weather.gov"])]
    for query, domains in searches:
        res = tavily.search(query=query, include_domains=domains,
                            start_date="2024-09-25", end_date="2024-10-05", max_results=5)
        for hit in res["results"]:
            text = (hit["title"] + " " + hit.get("content", "")).lower()
            if "/gsp/" in hit["url"] or town.lower() in text:
                return f"{hit['title']} ({hit['url']})"
    return "no matching official report found"

def safe_nemotron(message):
    try:
        return ask_nemotron(message)
    except Exception:
        return {"urgent": None, "reason": "unavailable", "location": "unknown",
                "explanation": "Second opinion unavailable, so a human should check."}

def safe_alert(place, reason):
    try:
        return official_alert(place, reason)
    except Exception:
        return "official source unavailable right now"

def second_opinion(text):
    n = safe_nemotron(text)
    u = n.get("urgent")
    if isinstance(u, str):
        n["urgent"] = {"true": True, "false": False}.get(u.strip().lower())
    if n.get("urgent") is False and str(n.get("reason") or "none") == "none":
        n["source"] = "not needed (not an emergency)"
    else:
        n["source"] = safe_alert(str(n.get("location") or "unknown"),
                                 str(n.get("reason") or "none"))
    return n

DAILY_LIMIT = 500          # messages per day for the public demo
_usage = {"day": None, "n": 0}
_lock = threading.Lock()

def take_budget(k):
    today = datetime.now(ZoneInfo("America/New_York")).date()
    with _lock:
        if _usage["day"] != today:
            _usage["day"], _usage["n"] = today, 0
        if _usage["n"] + k > DAILY_LIMIT:
            return False
        _usage["n"] += k
        return True

def triage(raw):
    texts = [t.strip() for t in raw.split("\n") if t.strip()][:MAX_MESSAGES]
    if not texts:
        return pd.DataFrame()
    if not take_budget(len(texts)):
        raise gr.Error("Daily demo limit reached. Please try again tomorrow.")
    probs = predict(texts)
    with ThreadPoolExecutor(max_workers=5) as pool:
        opinions = list(pool.map(second_opinion, texts))

    rows = []
    for text, p, n in zip(texts, probs, opinions):
        failed = n["urgent"] is None
        agree = (not failed) and (p > 0.5) == n["urgent"]
        b = band(p)
        if not agree and not failed and b != "NEEDS REVIEW":
            b = "NEEDS REVIEW"
        if failed and b == "ROUTINE":
            b = "NEEDS REVIEW"
        models = ("agree" if agree else
                  "second opinion unavailable" if failed else
                  "DISAGREE: human review")
        town = str(n.get("location") or "unknown").split(",")[-1].strip()
        rows.append({"Band": b, "Rater": round(p, 2), "Reason": str(n.get("reason") or "none"),
                     "Location": town,
                     "Address": find_address(text, n.get("address")),
                     "People": find_people(text, n.get("people")),
                     "Vulnerable": n.get("vulnerable") or "",
                     "Why": str(n.get("explanation") or ""), "Official source": n["source"],
                     "Models": models, "Nemotron": n["urgent"],
                     "Reply": str(n.get("reply") or ""), "Message": text})
    order = {"URGENT": 0, "NEEDS REVIEW": 1, "ROUTINE": 2}
    df = pd.DataFrame(rows)
    df["n_first"] = (df.Nemotron == True).astype(int)
    df = df.assign(o=df.Band.map(order)).sort_values(
        ["o", "n_first", "Rater"], ascending=[True, False, False])
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.drop(columns=["o", "n_first"])

COLORS = {"URGENT": "#d62828", "NEEDS REVIEW": "#f77f00", "ROUTINE": "#2a9d8f"}
STREET = re.compile(r"\b\d{1,5}\s+(?:\w+\s+){1,3}"
                    r"(?:st|street|rd|road|dr|drive|ave|avenue|ln|lane|ct|court|way|blvd|hwy)\b",
                    re.I)

def source_html(s):
    if s.endswith(")") and " (http" in s:
        title, url = s[:-1].split(" (http", 1)
        return f'<a href="http{html.escape(url)}" target="_blank" rel="noopener">{html.escape(title)}</a>'
    return html.escape(s)

TOWNS = {"swannanoa": (35.60, -82.40), "asheville": (35.60, -82.56),
         "black mountain": (35.62, -82.32), "chimney rock": (35.44, -82.25),
         "lake lure": (35.42, -82.20), "marshall": (35.80, -82.68),
         "spruce pine": (35.91, -82.06), "burnsville": (35.92, -82.30),
         "hendersonville": (35.32, -82.46), "boone": (36.21, -81.67)}
COUNTY = {"swannanoa": "Buncombe", "asheville": "Buncombe", "black mountain": "Buncombe",
          "chimney rock": "Rutherford", "lake lure": "Rutherford", "marshall": "Madison",
          "spruce pine": "Mitchell", "burnsville": "Yancey",
          "hendersonville": "Henderson", "boone": "Watauga"}
AREA_RISK = {"Buncombe": "Relatively Moderate · flood 91.7/100"}
# FEMA National Risk Index v1.20, via riskbycounty.com/north-carolina/buncombe-county,
# checked Oct 29, 2026.
TEAMS = {"Team A (Asheville)": (35.60, -82.56),
         "Team B (Spruce Pine)": (35.91, -82.06),
         "Team C (Hendersonville)": (35.32, -82.46)}

def km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))

def nearest_team(town):
    spot = TOWNS.get(town.lower())
    if not spot:
        return "location needed"
    name = min(TEAMS, key=lambda t: km(TEAMS[t], spot))
    d = km(TEAMS[name], spot)
    return f"{name}, " + ("same town" if d < 1 else f"{d:.0f} km")

def checked_address(addr, message):
    addr = str(addr or "").strip()
    if addr and addr.split(",")[0].lower() in message.lower():
        return addr
    return ""

def find_address(message, model_addr):
    m = STREET.search(message)
    if m:
        return m.group(0)
    return checked_address(model_addr, message)

WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
NUM = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
PEOPLE = re.compile(NUM + r"\s+(?:people|ppl|persons|kids|children|adults|of us)\b", re.I)
FAMILY = re.compile(r"\bfamily of\s+" + NUM + r"\b", re.I)

def find_people(message, model_people):
    m = PEOPLE.search(message) or FAMILY.search(message)
    if m:
        w = m.group(1).lower()
        return str(int(w) if w.isdigit() else WORDNUM[w])
    try:
        n = int(model_people)
        return str(n) if n > 0 else ""
    except (TypeError, ValueError):
        return ""

def tile(label, value, alert=False):
    color = "#b00020" if alert else "#111"
    return (f'<div style="background:#f4f5f7; border-radius:8px; padding:8px 10px">'
            f'<div style="font-size:11px; color:#666; letter-spacing:.05em">{label}</div>'
            f'<div style="font-size:15px; font-weight:700; color:{color}">{value}</div></div>')

def card(r):
    color = COLORS[r["Band"]]
    town = r["Location"]
    reason = "" if r["Reason"] == "none" else html.escape(r["Reason"].upper())

    head = (f'<div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap">'
            f'<span style="background:{color}; color:#fff; font-weight:700; padding:3px 12px;'
            f' border-radius:999px; font-size:14px">#{r["Rank"]} {r["Band"]}</span>')
    if reason:
        head += (f'<span style="border:1px solid {color}; color:{color}; font-weight:700;'
                 f' padding:2px 10px; border-radius:999px; font-size:13px">{reason}</span>')
    head += (f'<span style="margin-left:auto; color:#555; font-size:14px">'
             f'📍 {html.escape(town)}</span></div>')

    msg = (f'<div style="font-size:20px; font-weight:600; line-height:1.35; margin:10px 0">'
           f'"{html.escape(r["Message"])}"</div>')

    if r["Address"]:
        addr = tile("ADDRESS", html.escape(r["Address"]))
    elif r["Band"] == "ROUTINE":
        addr = tile("ADDRESS", "not needed")
    else:
        addr = tile("ADDRESS", "MISSING: ask sender", alert=True)
    ppl = r["People"] or "not stated"
    vul = f' · {html.escape(str(r["Vulnerable"]))}' if r["Vulnerable"] else ""
    county = COUNTY.get(town.lower(), "")
    if county in AREA_RISK:
        risk = f'{county} Co. · {AREA_RISK[county]}'
    elif county:
        risk = f'{county} Co. · not in demo data'
    else:
        risk = "unknown area"
    tiles = (f'<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));'
             f' gap:8px">'
             + addr
             + tile("PEOPLE", f"{html.escape(str(ppl))}{vul}")
             + tile("AREA RISK (FEMA NRI)", risk)
             + tile("NEAREST TEAM (demo)", nearest_team(town))
             + "</div>")

    if r["Models"] == "agree":
        check = f'✓ Both models agree (Rater {r["Rater"]}, Nemotron urgent: {html.escape(str(r["Nemotron"]))})'
    else:
        label = "DISPUTED: human review" if r["Models"].startswith("DISAGREE") else r["Models"]
        check = (f'<b style="color:#b00020">⚠ {label}</b> '
                 f'(Rater {r["Rater"]}, Nemotron urgent: {html.escape(str(r["Nemotron"]))})')
    small = (f'<div style="font-size:14px; color:#555; margin-top:10px; line-height:1.6">'
             f'Why: {html.escape(r["Why"])}<br>{check}<br>'
             f'Official report (Helene, 2024): {source_html(r["Official source"])}</div>')

    reply = ""
    if r["Reply"]:
        reply = (f'<div style="margin-top:12px; padding:10px 14px; background:#e8f1ff;'
                 f' border:1px solid #9cc2ff; border-radius:14px 14px 14px 2px; max-width:80%">'
                 f'<div style="font-size:12px; font-weight:700; color:#1d4ed8; margin-bottom:4px">'
                 f'💬 DRAFT REPLY TO SENDER (a human checks before sending)</div>'
                 f'<div style="font-style:italic; color:#1e293b">{html.escape(r["Reply"])}</div></div>')

    return (f'<div style="border-left:8px solid {color}; padding:14px 16px; margin:14px 0;'
            f' background:#fff; color:#222; border-radius:8px;'
            f' box-shadow:0 1px 3px rgba(0,0,0,.08)">'
            f'{head}{msg}{tiles}{small}{reply}</div>')

def kpis(df, seconds):
    counts = df.Band.value_counts()
    flags = int((df.Models != "agree").sum())
    now = datetime.now(ZoneInfo("America/New_York")).strftime("%I:%M %p ET")
    def kpi_box(label, n, color):
        return (f'<div style="flex:1; min-width:140px; background:#fff; border-top:6px solid {color};'
                f' border-radius:10px; padding:12px 14px; box-shadow:0 1px 3px rgba(0,0,0,.08)">'
                f'<div style="font-size:34px; font-weight:800; color:{color}">{n}</div>'
                f'<div style="font-size:13px; color:#555; font-weight:700">{label}</div></div>')
    return ('<div style="display:flex; gap:10px; flex-wrap:wrap">'
            + kpi_box("URGENT", counts.get("URGENT", 0), COLORS["URGENT"])
            + kpi_box("NEEDS REVIEW", counts.get("NEEDS REVIEW", 0), COLORS["NEEDS REVIEW"])
            + kpi_box("ROUTINE", counts.get("ROUTINE", 0), COLORS["ROUTINE"])
            + kpi_box("DISPUTED", flags, "#6b21a8")
            + '</div>'
            + f'<div style="font-size:13px; color:#666; margin-top:6px">'
              f'Ranked {len(df)} messages in {seconds:.0f} s, at {now}.</div>')

def cards_html(df, choice):
    if choice == "DISPUTED":
        df = df[df.Models != "agree"]
    elif choice != "All":
        df = df[df.Band == choice]
    if df.empty:
        return "<p>No messages in this group.</p>"
    return "".join(card(r) for _, r in df.iterrows())

def dispatch_text(df):
    lines = [f"#{r['Rank']} {r['Band']} | {r['Reason']} | {r['Address'] or 'NO ADDRESS'}, "
             f"{r['Location']} | people: {r['People'] or '?'} | {r['Message']}"
             for _, r in df.iterrows() if r["Band"] != "ROUTINE"]
    return "\n".join(lines)

def run(raw):
    start = time.time()
    df = triage(raw)
    if df.empty:
        return "Paste at least one message.", "", "", df, "All"
    return kpis(df, time.time() - start), cards_html(df, "All"), dispatch_text(df), df, "All"

def refilter(df, choice):
    if df is None or len(df) == 0:
        return ""
    return cards_html(df, choice)

SAMPLES = ("Water is at the second step. Two kids here, no car. 41 Laurel Creek Dr, Swannanoa.\n"
           "Grandma has not taken her insulin since yesterday. 112 Pine Ridge Ln, Black Mountain.\n"
           "Water is in the house at 214 Riverside Dr. Elderly man cannot walk. Swannanoa.\n"
           "URGENT please call me back about my car insurance claim. 9 Hilltop Ct, Asheville.\n"
           "Power is out but we are safe and dry. 305 Cedar Bend Rd, Asheville.\n"
           "Necesito ayuda, el agua sube y mi mamá no puede caminar. 77 Maple Hollow Rd, Swannanoa.")

TICKER = ("Hurricane Helene demo window: Sept 25 to Oct 5, 2024, western North Carolina"
          " · Example messages, not real people"
          " · Two models check each message: SKR Rater (DistilBERT) and NVIDIA Nemotron on Nebius Token Factory"
          " · Official reports found live by Tavily"
          " · FEMA area risk: Buncombe County only"
          " · Rescue teams are demo locations"
          " · A human reads every message and makes every decision")

TICKER_HTML = f"""
<style>
.rr-ticker {{display:flex; align-items:center; background:#111827; color:#fff;
             border-radius:8px; overflow:hidden; font-weight:600; font-size:15px}}
.rr-badge  {{background:#d62828; color:#fff !important; padding:8px 12px; flex:none; letter-spacing:.05em}}
.rr-track  {{overflow:hidden; white-space:nowrap; flex:1}}
.rr-track span {{display:inline-block; padding-left:100%; color:#fff !important;
                 animation:rr-scroll 45s linear infinite}}
.rr-ticker:hover span {{animation-play-state:paused}}
@keyframes rr-scroll {{from {{transform:translateX(0)}} to {{transform:translateX(-100%)}}}}
@media (prefers-reduced-motion: reduce) {{.rr-track span {{animation:none; padding-left:12px}}}}
</style>
<div class="rr-ticker"><div class="rr-badge">DEMO</div>
  <div class="rr-track"><span>{TICKER}</span></div></div>
"""
def step(n, title, sub):
    return (f'<div style="flex:1; min-width:150px; background:#f4f5f7; border-radius:8px; padding:10px">'
            f'<div style="font-size:12px; color:#666 !important">STEP {n}</div>'
            f'<div style="font-weight:700; color:#111 !important">{title}</div>'
            f'<div style="font-size:13px; color:#444 !important">{sub}</div></div>')

ARROW = '<div style="align-self:center; font-size:20px; color:#999 !important">→</div>'
HOW_HTML = ('<div style="display:flex; gap:6px; flex-wrap:wrap; margin:6px 0">'
            + step(1, "SKR Rater", "DistilBERT scores every message, fast")
            + ARROW + step(2, "NVIDIA Nemotron", "on Nebius Token Factory: second opinion, facts, draft reply")
            + ARROW + step(3, "Tavily", "finds a related official report")
            + ARROW + step(4, "Ranked cards", "a human reads all; disputes are flagged")
            + '</div>')
STORY_HTML = """
<div style="background:#111827; border-radius:12px; padding:22px 24px">
  <div style="color:#f77f00 !important; font-weight:700; letter-spacing:.08em; font-size:13px">
    SEPTEMBER 2024 · HURRICANE HELENE · WESTERN NORTH CAROLINA</div>
  <div style="color:#fff !important; font-size:26px; font-weight:800; margin:8px 0 10px">
    A family on a roof should not wait behind a power-cut text.</div>
  <div style="color:#d1d5db !important; font-size:16px; line-height:1.6">
    In a disaster surge, help messages arrive faster than people can read them.
    <b style="color:#fff !important">Rescue Rater</b> helps dispatchers choose which message to read first.
    <b style="color:#fff !important">A human reads every message.</b>
    Two AI models suggest the order, and flag every message where they disagree.</div>
  <div style="color:#fca5a5 !important; font-size:14px; font-weight:700; margin-top:12px">
    Decision support only. Not connected to 911. Example messages only.</div>
  <div style="color:#9ca3af !important; font-size:13px; margin-top:6px">
    Built with NVIDIA Nemotron on Nebius Token Factory, Tavily, and a fine-tuned DistilBERT.</div>
</div>"""

with gr.Blocks(title="Rescue Rater") as demo:
    with gr.Column(visible=True) as intro:
        gr.HTML(STORY_HTML)
        start = gr.Button("Start the demo  →", variant="primary", size="lg")
    gr.Markdown("# Rescue Rater\n"
                "**Decision support only. A human reads every message and makes every decision. "
                "In a life threat, call 911.**")
    gr.HTML(TICKER_HTML)
    gr.HTML(HOW_HTML)
    gr.Markdown("Paste help messages, one per line (max 10). The most urgent come first.")
    box = gr.Textbox(lines=6, label="Help messages", value=SAMPLES)
    btn = gr.Button("Rank messages", variant="primary")
    kpi = gr.HTML()
    pick = gr.Radio(["All", "URGENT", "NEEDS REVIEW", "ROUTINE", "DISPUTED"],
                    value="All", label="Show")
    out = gr.HTML()
    dispatch = gr.Textbox(label="Copy for dispatch", lines=5)
    state = gr.State()
    btn.click(run, inputs=box, outputs=[kpi, out, dispatch, state, pick], concurrency_limit=2)
    start.click(lambda: gr.update(visible=False), outputs=intro)
    pick.change(refilter, inputs=[state, pick], outputs=out)

demo.launch()
