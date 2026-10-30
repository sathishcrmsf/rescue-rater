---
license: apache-2.0
language:
- en
base_model: distilbert-base-uncased
pipeline_tag: text-classification
tags:
- disaster-response
- triage
- hurricane-helene
- distilbert
---

# SKR Rescue Rater

A small DistilBERT model that gives the probability that a disaster help message is urgent.
It is the fast first model in **Rescue Rater**, a decision support app that helps dispatchers
choose which message to read first. A human reads every message and makes every decision.

- App (Hugging Face Space): https://huggingface.co/spaces/skrsathish/rescue-rater
- Code: https://github.com/sathishcrmsf/rescue-rater

## Labels

| Label | Meaning |
|---|---|
| 0 | Not urgent (power cut, information request, minor damage, supplies) |
| 1 | Urgent (trapped, medical, missing person, collapse) |

## How to use

```python
from transformers import pipeline
rater = pipeline("text-classification", model="skrsathish/skr-rescue-rater")
rater("Water is at the second step. Two kids here, no car.")
```

In the app, the urgent probability sets a band: URGENT (0.7 or more), NEEDS REVIEW
(0.1 to 0.7), ROUTINE (less than 0.1). A second model (NVIDIA Nemotron on Nebius Token
Factory) checks every message. If the two models disagree, the message goes to human review.

## Training

- Base model: `distilbert-base-uncased`, fine-tuned for 2 classes.
- Data: synthetic help messages in the style of Hurricane Helene (western North Carolina,
  2024). Urgent and not-urgent messages are shuffled together, with many wordings per
  category.
- Held-out test: one whole wording per category is kept out of training, so the test uses
  phrasing the model never saw.
- 3 epochs, AdamW, learning rate 3e-5, batch size 16, max length 64, seed 42.

## Evaluation (held-out synthetic test set, 48 urgent and 48 calm messages)

| | Urgent caught | Calm flagged by mistake |
|---|---|---|
| This model, 5 training seeds | 48 of 48 in every run | 0 to 8 of 48 (median 4) |
| This model, seed 42 (uploaded) | 48 of 48 | 4 of 48 |
| Keyword baseline (TF-IDF + logistic regression) | 48 of 48 | 39 of 48 |

## Limits

- Trained on synthetic data. Real messages are messier, shorter, and misspelled.
- English only. Example: a Spanish urgent message scored 0.08. In the app, the second model
  caught it.
- Text it never saw can move the score. Adding a street address moved one calm message from
  0.02 to 0.45.
- Not a 911 system. Use only as decision support with a human in charge.

## License

Apache License 2.0, the same as the base model, DistilBERT.

## Author

Sathish, IEEE member. Built for the Nebius x NVIDIA Global AI Hackathon (2026). Claude
(Anthropic) helped draft code and documentation.
