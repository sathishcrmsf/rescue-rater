# Rescue Rater

**IEEE AI Hack for Humanity 2026 | Track: Geospatial Intelligence for Action**

Rescue Rater sorts disaster help messages so rescue teams reach the most urgent people first.

## The problem

In September 2024, Hurricane Helene flooded western North Carolina. More than 100 people died in the state. 911 lines were overloaded. Thousands of people sent help messages by text and social media. Responders could not tell which messages were life or death.

**Current experience:** messages arrive in time order. A family on a roof can wait behind many messages about power cuts. Teams choose areas by phone calls and guesswork.

**With Rescue Rater:** each message gets a priority score when it arrives. The most urgent messages from the worst-hit areas go to the top. Each team sees its own list, nearest critical cases first.

## How it works

```
Help message --> SKR Binary Rater --> urgent probability --+
Zip code --> Flood risk score (1-10) ----------------------+
                                                           v
                  Priority = risk + 10 x urgent probability
                                                           v
         Three-band triage --> Hot zones --> Drone alert --> Team dispatch
```

1. **Risk layer.** Each zip code has a flood risk score from 1 to 10.
2. **Message layer.** SKR Binary Rater, a fine-tuned DistilBERT model, gives the probability that a message is urgent.
3. **Priority layer.** Priority = zip flood risk + (10 x urgent probability).
4. **Three-band triage.** URGENT (0.7 or more), NEEDS REVIEW (0.1 to 0.7), ROUTINE (less than 0.1). Anything that might be urgent gets a human look.
5. **Hot zones.** Urgent messages grouped by zip. Hot score = urgent count x flood risk.
6. **Drone alert.** A hot zone above the alert level sends the nearest drone to confirm the need (simulated).
7. **Dispatch.** Each urgent message goes to the nearest rescue team. Lists sort by priority first, then distance.

## Results (held-out test set, 96 messages)

| | Rescue Rater | Baseline (TF-IDF + logistic regression) |
|---|---|---|
| Accuracy | **84%** | 59% |
| Urgent caught | 39 of 48 | 48 of 48 |
| False alarms | **6 of 48** | 39 of 48 |

The baseline flags almost everything as urgent. Rescue Rater cuts false alarms from 39 to 6. With the NEEDS REVIEW band, no urgent message stays in the ROUTINE pile in the demo run.

These results are early evidence only. The test set is small and synthetic.

## Data quality rules

- Urgent and non-urgent messages are shuffled together. They are never grouped.
- Each category has many different wordings.
- Whole wordings are held out for testing, so the model is tested on phrasing it never saw.
- Calm but urgent messages are included (for example, "insulin ran out yesterday").
- Zip codes are not in the message text, so the model judges words, not locations.

## How to run

1. Open `Rescue_Rater.ipynb` in Google Colab.
2. Set Runtime > Change runtime type > T4 GPU.
3. Run all cells. Training takes about 2 to 5 minutes.

`rescue_rater.py` has the same code as plain Python, split into cells.

## Limits and risks

- The model missed most "mudslide hit the house next door" messages, a wording it never saw. It can miss new phrasing. The NEEDS REVIEW band reduces this risk.
- The 0.1 review threshold was set after looking at results. A real system sets it on a separate validation set.
- The reason column (trapped, medical, missing, collapse) comes from synthetic labels. A real system needs a second model to predict it.
- In the demo, zip codes are assigned at random. Distances are straight-line, not road distance.
- Fake or spam messages could push an area up the list.

## Trust and responsible design

- Rescue Rater only sorts. A human reads every message and makes every rescue decision. A low score never means ignored.
- A message with no location goes to NEEDS REVIEW with a "location unknown" flag. It is never dropped.
- A real system would encrypt messages, limit access to verified responders, rate-limit and flag duplicates, and delete data after the event.

## Scalability

- DistilBERT is small and fast. It can run on serverless cloud services that scale up during a disaster.
- Swap the risk layer to support fire, hurricane, or earthquake. Load new zip scores to use it in any region.

## Next steps

1. **Real flood data:** NC FRIS for flood risk and FIMAN for live river gauges, replacing mock scores.
2. **Location:** extract addresses from messages with a geocoder, use carrier location from texts to 911, and auto-reply to ask for an address when it is missing.
3. **More varied data:** 10,000+ real labelled messages from past disasters, with messy, short, and misspelled messages and other languages.
4. **Road-closure routing:** replace straight-line distance with real road access.
5. **Alert loop:** recommend Wireless Emergency Alerts for hot zones. An official approves each alert.
6. **Reason model:** a second small model that predicts trapped, medical, missing, or collapse.
7. **Drone checks:** live camera feeds with computer vision.
8. **Pilot:** a read-only sorting aid that runs next to a county 911 dispatch system.

## Third-party and pre-existing components

- DistilBERT (`distilbert-base-uncased`) from Hugging Face.
- Python, pandas, scikit-learn, PyTorch, Hugging Face Transformers, Google Colab.
- Training method reused from the author's earlier SKR Binary Rater project (ticket triage).
- All messages are synthetic. Zip flood scores are mock values. Drone results are simulated. Town names and zip codes are real; coordinates are approximate.
- Planned data sources: NC FRIS, NC FIMAN, NWS Southeast River Forecast Center.

## AI disclosure

Claude (Anthropic), an AI assistant, helped draft code and documentation. The author is responsible for the correctness, safety, and originality of the submission.

## Author

Sathish, IEEE member.
