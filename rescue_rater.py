# Rescue Rater - IEEE AI Hack for Humanity 2026
# Each CELL block is one Google Colab cell. Run in order with a T4 GPU.

# ============================================================
# CELL 1: Build shuffled Helene message dataset with held-out templates
# ============================================================
import random, pandas as pd
random.seed(42)

URGENT = {
 "trapped": ["We are trapped on the roof, water still rising",
             "Stuck upstairs, water is up to the stairs now",
             "Cannot get out of the house, flooding fast",
             "My car is stuck in water and it is coming in",
             "Family of {n} trapped in attic, need boat",
             "Water at our chest, cannot leave the house"],
 "medical": ["Mom is diabetic, insulin ran out yesterday",
             "My father needs oxygen and the power is out",
             "Neighbor is bleeding badly, roads are closed",
             "Baby has high fever and we cannot drive out",
             "Grandma fell and cannot stand, no way to reach hospital",
             "Dialysis patient here, missed treatment for {n} days"],
 "missing": ["Cannot reach my elderly parents for {n} days",
             "My brother went to check the creek and has not come back",
             "Lost contact with my aunt, her house is by the river",
             "Kids were at a friend's house, now we cannot reach them"],
 "collapse": ["Tree fell on our house, someone is under it",
              "Part of the roof collapsed, my son is hurt",
              "Mudslide hit the house next door, people inside",
              "House shifted off foundation, we are inside"],
}
NOT_URGENT = {
 "power": ["Power is out but we are okay",
           "No electricity since last night, everyone is fine",
           "Lost power, we have a generator for now",
           "When will power come back in our area?"],
 "info": ["Where is the nearest shelter?",
          "Is the main road open yet?",
          "How do I apply for FEMA help?",
          "Are schools closed tomorrow?"],
 "property": ["Basement has some water, we are safe upstairs",
              "Fence blew down in the yard",
              "Some shingles came off the roof, no leaks",
              "Car got a dent from a branch, nobody hurt"],
 "supplies": ["We have food for a few days, need more water soon",
              "Running low on dog food, not urgent",
              "Could use some batteries when someone can bring them",
              "We are fine but phone battery is low"],
}
PREFIX = ["", "Please help. ", "Hi, ", "Urgent: ", "Not sure who to ask, ", "Help, "]
SUFFIX = ["", " Please send someone.", " Thanks.", " Any update?", ""]

def build(groups, label):
    rows = []
    for cat, temps in groups.items():
        for t_id, t in enumerate(temps):
            for _ in range(12):
                msg = random.choice(PREFIX) + t.format(n=random.randint(2,5)) + random.choice(SUFFIX)
                rows.append({"text": msg, "label": label,
                             "category": cat, "template": f"{cat}_{t_id}"})
    return rows

df = pd.DataFrame(build(URGENT, 1) + build(NOT_URGENT, 0))

# Hold out one whole template per category for testing (no memorising)
test_templates = df.groupby("category")["template"].unique().apply(lambda x: random.choice(list(x)))
df["split"] = df["template"].apply(lambda t: "test" if t in set(test_templates) else "train")

# Shuffle so urgent and non-urgent are mixed
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train, test = df[df.split=="train"], df[df.split=="test"]
print(len(train), "train |", len(test), "test")
print(train["label"].value_counts())
print(df.head(10))

# ============================================================
# CELL 2: Baseline, TF-IDF + logistic regression
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

vec = TfidfVectorizer(ngram_range=(1, 2))
Xtr, Xte = vec.fit_transform(train.text), vec.transform(test.text)
base = LogisticRegression(max_iter=1000).fit(Xtr, train.label)
base_pred = base.predict(Xte)
print(classification_report(test.label, base_pred, target_names=["not urgent", "urgent"]))

# ============================================================
# CELL 3: Fine-tune DistilBERT as the binary rater
# ============================================================
!pip -q install transformers
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

def batches(d, bs=16, seed=0):
    d = d.sample(frac=1, random_state=seed)  # reshuffle every epoch
    for i in range(0, len(d), bs):
        b = d.iloc[i:i+bs]
        enc = tok(list(b.text), padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        yield enc, torch.tensor(b.label.values).to(device)

model.train()
for epoch in range(3):
    for enc, y in batches(train, seed=epoch):
        loss = model(**enc, labels=y).loss
        loss.backward(); opt.step(); opt.zero_grad()
    print("epoch", epoch + 1, "loss", round(loss.item(), 3))

def predict(texts):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i+32], padding=True, truncation=True,
                      max_length=64, return_tensors="pt").to(device)
            out += torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
    return np.array(out)

rater_pred = (predict(list(test.text)) > 0.5).astype(int)
print(classification_report(test.label, rater_pred, target_names=["not urgent", "urgent"]))
y = test.label.values
total, calm = (y == 1).sum(), (y == 0).sum()
print(f"Rescue Rater caught {((rater_pred==1)&(y==1)).sum()} of {total} urgent | false alarms {((rater_pred==1)&(y==0)).sum()} of {calm}")
print(f"Baseline     caught {((base_pred==1)&(y==1)).sum()} of {total} urgent | false alarms {((base_pred==1)&(y==0)).sum()} of {calm}")

# ============================================================
# CELL 4: Zip risk + priority + hot zones + Uber-style dispatch
# ============================================================
import math
# Flood scores are MOCK values for the demo. In production: NC FRIS + FIMAN data.
ZIPS = pd.DataFrame([
 ("28778", "Swannanoa",     10, 35.60, -82.40),
 ("28720", "Chimney Rock",  10, 35.44, -82.25),
 ("28746", "Lake Lure",      9, 35.42, -82.20),
 ("28753", "Marshall",       9, 35.80, -82.68),
 ("28777", "Spruce Pine",    8, 35.91, -82.06),
 ("28801", "Asheville",      7, 35.60, -82.56),
 ("28711", "Black Mountain", 7, 35.62, -82.32),
 ("28714", "Burnsville",     6, 35.92, -82.30),
 ("28792", "Hendersonville", 4, 35.32, -82.46),
 ("28607", "Boone",          3, 36.21, -81.67),
], columns=["zip", "town", "flood_risk", "lat", "lon"])
TEAMS = {"Team A": "28801", "Team B": "28777", "Team C": "28792"}
Z = ZIPS.set_index("zip")

def km(z1, z2):
    a, b = Z.loc[z1], Z.loc[z2]
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    h = (math.sin((p2-p1)/2)**2 +
         math.cos(p1)*math.cos(p2)*math.sin(math.radians(b.lon-a.lon)/2)**2)
    return round(2 * 6371 * math.asin(math.sqrt(h)), 1)

# 30 "live" messages: unseen wordings, zip assigned at random for the demo (not in the text)
live = test.sample(30, random_state=7).copy()
random.seed(7)
live["zip"] = random.choices(list(ZIPS.zip), weights=list(ZIPS.flood_risk), k=30)
live = live.merge(ZIPS[["zip", "town", "flood_risk"]], on="zip")
live["urgent_prob"] = predict(list(live.text)).round(2)
live["urgent"] = (live.urgent_prob > 0.5).astype(int)
live["priority"] = (live.flood_risk + 10 * live.urgent_prob).round(1)

print("=== PRIORITY LIST ===")
print(live.sort_values("priority", ascending=False)
      [["priority", "town", "flood_risk", "urgent_prob", "text"]].to_string(index=False))

print("\n=== HOT ZONES ===")
hz = (live[live.urgent == 1].groupby(["town", "flood_risk"])
      .size().reset_index(name="urgent_msgs"))
hz["hot_score"] = hz.urgent_msgs * hz.flood_risk
print(hz.sort_values("hot_score", ascending=False).to_string(index=False))

print("\n=== DISPATCH (priority first, distance breaks ties) ===")
urgent = live[live.urgent == 1].copy()
urgent["team"] = urgent.zip.apply(lambda z: min(TEAMS, key=lambda t: km(TEAMS[t], z)))
urgent["km"] = urgent.apply(lambda r: km(TEAMS[r.team], r.zip), axis=1)
for team in TEAMS:
    lst = urgent[urgent.team == team].sort_values(["priority", "km"], ascending=[False, True])
    print(f"\n{team} (based in {Z.loc[TEAMS[team]].town}): {len(lst)} urgent cases")
    print(lst[["priority", "km", "town", "category", "text"]]
          .rename(columns={"category": "reason"}).to_string(index=False))

# ============================================================
# CELL 5: Hot zone alert + drone scout (SIMULATED for demo)
# ============================================================
ALERT_THRESHOLD = 15          # hot_score that triggers a drone
DRONE_BASES = {"Drone 1": "28801", "Drone 2": "28714"}

def drone_scout(row):
    # Demo only: high-risk zips are more likely to be confirmed.
    # In production: live drone camera + computer vision.
    random.seed(int(row.zip))
    confirmed = random.random() < row.flood_risk / 10
    return "CONFIRMED: people need rescue" if confirmed else "NOT CONFIRMED: send to human review"

print("=== DRONE ALERTS ===")
hz = hz.merge(ZIPS[["town", "zip"]], on="town")
for _, r in hz.sort_values("hot_score", ascending=False).iterrows():
    if r.hot_score >= ALERT_THRESHOLD:
        drone = min(DRONE_BASES, key=lambda d: km(DRONE_BASES[d], r.zip))
        print(f"ALERT: {r.town} ({r.zip}) hot score {r.hot_score}")
        print(f"   {drone} launched, {km(DRONE_BASES[drone], r.zip)} km away")
        print(f"   Drone report: {drone_scout(r)}\n")

# ============================================================
# CELL 6: Three-band triage + reason column
# ============================================================
# Lean toward safety: anything with even a small chance of urgency gets a human look.
# The 0.1 threshold was set after seeing demo results; a real system sets it on a validation set.
# The reason column comes from synthetic labels; a real system needs a second model to predict it.
def band(p):
    if p >= 0.7: return "URGENT"
    if p >= 0.1: return "NEEDS REVIEW"
    return "ROUTINE"

live["band"] = live.urgent_prob.apply(band)
order = {"URGENT": 0, "NEEDS REVIEW": 1, "ROUTINE": 2}
view = (live.assign(o=live.band.map(order))
            .sort_values(["o", "priority"], ascending=[True, False]))

print("=== THREE-BAND TRIAGE ===")
print(live.band.value_counts().to_string())
print()
print(view[["band", "priority", "town", "category", "urgent_prob", "text"]]
      .rename(columns={"category": "reason"}).to_string(index=False))

missed = view[(view.band == "ROUTINE") & (view.label == 1)]
print(f"\nUrgent messages left in ROUTINE: {len(missed)} of {(live.label == 1).sum()}")
