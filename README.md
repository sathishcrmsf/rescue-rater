# Rescue Rater v2

**Nebius x NVIDIA Global AI Hackathon | Track: Best Apps and Agents | Also: Best Use of Tavily**

- Live demo: https://huggingface.co/spaces/skrsathish/rescue-rater
- Video: coming soon
- Model: https://huggingface.co/skrsathish/skr-rescue-rater

Rescue Rater helps dispatchers choose which disaster help message to read first.
A human reads every message. Two AI models suggest the order, and flag every message
where they disagree.

> Decision support only. Not connected to 911. Example messages only.

## The problem

In late September 2024, Hurricane Helene caused catastrophic flooding in western North
Carolina. In a disaster surge, help messages arrive faster than people can read them.
A family on a roof should not wait behind a power-cut text.

## How it works

```
Help messages (max 10 per click)
  -> SKR Rater (fine-tuned DistilBERT): urgent probability, fast, on ZeroGPU
  -> In parallel for each message (5 workers):
       NVIDIA Nemotron 3 Nano on Nebius Token Factory:
         urgent, reason, location, address, people, vulnerable, explanation, draft reply
       Tavily: related official report (weather.gov, fema.gov, ncdps.gov, readync.gov,
         buncombecounty.org), Helene date window, town relevance check
  -> Rules: URGENT / NEEDS REVIEW / ROUTINE; any disagreement goes to NEEDS REVIEW (DISPUTED)
  -> Dashboard tiles, filter, one card per message, copy-for-dispatch text
```

Bands from the Rater probability `p`:

| Band | Rule |
|---|---|
| URGENT | p >= 0.7 |
| NEEDS REVIEW | 0.1 <= p < 0.7 |
| ROUTINE | p < 0.1 |

If the Rater (`p > 0.5`) and Nemotron disagree, the message goes to NEEDS REVIEW and is
marked DISPUTED. The second model can move a message up, never down. Inside a band,
messages that Nemotron calls urgent come first.

## How Nemotron and Token Factory are used

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, called at runtime through the
  OpenAI-compatible Token Factory API (`https://api.tokenfactory.nebius.com/v1/`).
- Nemotron is the second opinion. It reads the whole sentence and other languages.
  The small Rater only knows English.
- Nemotron returns JSON: urgent, reason, location, address, people, vulnerable groups,
  a one-line explanation, and a draft reply in the sender's language.
- Why Nano: one call per message must be fast and cheap. Calls run in parallel, so a
  batch of 6 messages ranks in about 6 to 12 seconds.
- Fail-safe: 20 second timeout, one retry. If Nemotron fails, the card says
  "second opinion unavailable" and a human checks it.

## How Tavily is used

- The query depends on the need: trapped finds flood warnings, medical finds shelters,
  missing finds reunification, collapse finds road closures.
- Official domains only, and the Helene date window (Sept 25 to Oct 5, 2024).
- A result is kept only if it comes from the NWS Greer office (covers western North
  Carolina) or names the town. If the first search finds nothing, a second search looks
  for the regional NWS Helene report. Otherwise the app says
  "no matching official report found". Honest beats wrong.
- Messages that both models call non-emergency skip the search.
- Lesson: we tried Tavily for the FEMA county risk score. It found the official page but
  not the number (it is inside a map app). So we use fixed FEMA data. Live search for live
  facts, fixed data for fixed facts.

## Safety design

- Decision support only. A human reads every message and makes every decision.
- The second model can move a message up, never down.
- Address and people count come from fixed patterns first. A model address is shown only
  if it appears in the message.
- Draft replies are drafts. A human checks before sending.
- All text from senders and models is HTML-escaped.
- Public demo limits: 10 messages per click, 2 clicks at a time, 500 messages per day.

## Results (held-out test set, synthetic)

Caught all 48 urgent messages in every run. Flagged 0 to 8 of 48 calm messages by mistake
(median 4), across 5 training seeds. The keyword baseline (TF-IDF + logistic regression)
flagged 39 of 48. Test wordings were never seen in training.

Example: *"Necesito ayuda, el agua sube y mi mamá no puede caminar."* The English-only
Rater scored it 0.08. Nemotron said urgent. They disagreed, so the app sent it to human
review, and the draft reply came back in Spanish.

## What's new in v2 (v1: IEEE AI Hack for Humanity, Oct 24, 2026)

| v1 | v2 |
|---|---|
| One model | Two models; disagreement goes to a human |
| Reason from synthetic labels | Nemotron predicts the reason |
| No location extraction | Street pattern + Nemotron, checked against the message |
| No official data | Tavily official reports; FEMA National Risk Index |
| No reply to sender | Draft reply in the sender's language |
| Results changed every run | PyTorch seed + 5-seed range |
| Notebook only | Hosted app with dashboard, filter, cards |

## What is real and what is demo

- Real: both models, live Nemotron and Tavily calls, ranking rules, address and people
  extraction, the safety rules.
- Demo: example messages and made-up street names; the Helene date window; FEMA risk for
  Buncombe County only (FEMA National Risk Index v1.20); rescue teams and straight-line
  distances.

## Known limits

- Training data is synthetic. Real messages are messier.
- The Rater reacts to text it never saw in training (adding a street address moved one
  score from 0.02 to 0.45).
- "3 kids on the way" can look like an address, because "way" is a street word.
- Nemotron Nano can write text before the JSON. If parsing fails, the message is marked
  urgent (the safe side).
- The daily limit resets when the Space restarts.

## Run it yourself

1. Duplicate the Hugging Face Space, or run `app.py` locally:
   `pip install gradio transformers torch openai tavily-python pandas spaces`
2. Set the environment variables `NEBIUS_API_KEY` and `TAVILY_API_KEY`
   (Space secrets on Hugging Face). Never put keys in code.
3. Training: `Rescue_Rater.ipynb` in Google Colab with a T4 GPU (`rescue_rater.py` has
   the same code as plain Python).

## Files

- `app.py`: the Gradio app (Hugging Face Space)
- `Rescue_Rater.ipynb`, `rescue_rater.py`: data, training, baseline, v1 triage
- `LICENSE`: Apache License 2.0

## Next steps

Real labeled disaster messages; voice and photo messages (Nemotron VL); road-aware
routing; assign and handled buttons; carrier location from 911 texts; a read-only pilot
next to a county dispatch system.

## Third-party components and data

- DistilBERT (`distilbert-base-uncased`), Hugging Face Transformers, PyTorch, Gradio,
  pandas, scikit-learn.
- NVIDIA Nemotron 3 Nano via Nebius Token Factory; Tavily Search API.
- FEMA National Risk Index v1.20 (Buncombe County). Town coordinates are approximate.

## AI disclosure

Claude (Anthropic) helped draft code and documentation. The author is responsible for the
correctness, safety, and originality of the submission.

## License

Apache License 2.0. See `LICENSE`.

## Author

Sathish, IEEE member.
