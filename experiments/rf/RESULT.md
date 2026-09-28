# Roboflow detector vs deployed pipeline - result (2026-09-27)

**Verdict: Roboflow shows no advantage; keep the existing model.** At matched selectivity the deployed
pipeline gives as many correct answers with far fewer wrong labels, and Roboflow cannot give MORE
answers than today even at its most permissive setting. Protocol: PROTOCOL.md (fixed before any
Roboflow output). Numbers: comparison.json. All photos below are clean for both models; they are
development evidence, not an independent or clinical validation.

## Run
3,510 distinct clean photos (the RIT manifest repeats 82 byte-identical Lower Limb photos; each is
counted once here), 0 errors, round trip 0.205 s median / 0.311 s p95 from this Mac to Roboflow
serverless (deployed pipeline: 7.0 ms median locally; neither is a Vercel measurement).
Dev (thresholds) = 199 clean val photos; evaluation = 3,311 clean test + RIT photos.

## Protocol result (section 3)
- Roboflow "selected" threshold (val accuracy >= deployed's 97.8%): 0.82 -> it answers almost nothing
  (4 right / 3 wrong of 148 wound photos). Verdict rule: "no demonstrated advantage / too small to tell".
- The deployed val accuracy is inflated (val was its early-stopping set), which made this bar strict.
- Matched coverage in the protocol's direction was unreachable: at the workflow's fixed 0.40 floor
  Roboflow answers fewer clean val wound photos than the deployed pipeline at 0.60.

## Follow-up head-to-head (added after seeing the above; labelled FOLLOW-UP in compare.py)
Roboflow at 0.40 (its most permissive setting) vs the deployed pipeline at 0.735, the threshold at
which it answers as many clean val wound photos (chosen on val):

| Clean evaluation photos | Roboflow @0.40 | Deployed @0.735 | Roboflow - deployed, paired 95% CI |
|---|---|---|---|
| Wound photos (148): right / wrong / unanswered | 39 / 12 / 97 | 46 / 1 / 101 | right -23..+11; wrong **+4..+18** |
| Burns recognised (119) | 32 | 40 | -24..+8 |
| Other non-wounds labelled (439) | 118 | 33 | **+63..+107** |
| Healthy feet labelled (1,508) | 60 | 64 | -30..+25 |
| Chronic wounds labelled (1,216) | 347 | 48 | **+255..+349** |
Deployed today (0.60): 60 right / 4 wrong / 84 unanswered; burns 53/119; other non-wounds 60/439;
feet 170/1,508; chronic 122/1,216.
Too few to judge per category: abrasion 1, bruise 11, cut 17 evaluation photos.
Combining ("answer only when both agree") answers 1 of 148 wound photos.

## Limitations
The workflow hides detections below 0.40; our val/test copies are 224 px while Roboflow trained at
640 px (a possible Roboflow disadvantage, not corrected); none of the 3,510 photos resembles any of
the 1,353 photos Roboflow was trained or selected on.
