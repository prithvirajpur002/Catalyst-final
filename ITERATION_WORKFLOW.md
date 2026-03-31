# Catalyst RVC — Iteration Workflow

## Core Principles

1. **One variable per experiment** — Change EXACTLY ONE thing. No mixing.
2. **Fixed test set** — All experiments evaluate on identical inputs.
3. **Explicit decisions** — Record which experiment won and why.
4. **Full traceability** — Every run is immutable and fully logged.

---

## The Iteration Loop

### Step 1: Define Experiment

Use the web UI to define a new experiment:

```
Experiment ID:  exp_001
Dataset:        clean
Config:         baseline
Epochs:         200
Batch Size:     6
```

**Rule:** If exp_001 already ran, create exp_002. Never overwrite.

### Step 2: Create Experiment Structure

The system creates:

```
experiments/
  exp_001/
    config.json          ← immutable spec (what ran)
    model/               ← trained model
    samples/             ← output samples
    logs/                ← training logs
```

### Step 3: Run Training

```bash
python rdp/experiment_runner.py run exp_001
```

System will:
- Load config from `config.json`
- Train for exactly 200 epochs
- Save model to `experiments/exp_001/model/`
- Log all output to `experiments/exp_001/logs/run.log`

### Step 4: Evaluate

System evaluates on CANONICAL TEST SET:

```
test_inputs/
  neutral.wav      (5s @ 440Hz)
  sustained.wav    (8s @ 440Hz)
  short.wav        (3s @ 440Hz)
```

All experiments use identical inputs.

**Scores produced:**
- `naturalness` (0.0–1.0)
- `clarity` (0.0–1.0)
- `identity` (0.0–1.0)

**Composite score:**
```
score = naturalness * 0.45 + clarity * 0.35 + identity * 0.20
```

### Step 5: Compare & Decide

Example results:

```
exp_001 (clean, baseline, 200e)     → composite: 0.782
exp_002 (clean, baseline, 300e)     → composite: 0.791
```

**Decision:** exp_002 won (+0.009 improvement)

**Record in UI:**

```
Winner:      exp_002
Loser:       exp_001
Reason:      "More epochs (300 vs 200) improved clarity slightly"
Next Exp:    exp_003
Rationale:   "Try different dataset (natural) with winning config"
```

### Step 6: Next Experiment

MUST change exactly one variable from winner:

```
exp_003:  Change DATASET from clean → natural
          Keep: baseline config, 300 epochs

exp_004:  Change EPOCHS from 300 → 400
          Keep: natural dataset, baseline config

exp_005:  Change CONFIG from baseline → high_quality
          Keep: natural dataset, 300 epochs
```

**NOT allowed:**
```
exp_007:  clean → natural AND 200 → 400 epochs   ❌ Two changes
exp_008:  baseline → high_quality AND 6 → 8 batch ❌ Two changes
```

---

## Command Reference

### UI (Recommended)

Go to web app → Experiment Control

- Define new experiment with dropdowns
- Click "Create Experiment"
- System creates immutable structure
- Check status in "Experiment Runs" table
- Record decisions in "Decisions" table

### CLI

```bash
# Create exp_001
python rdp/experiment_runner.py create exp_001 clean baseline 200 6

# Run it
python rdp/experiment_runner.py run exp_001

# Get info
python rdp/experiment_runner.py info exp_001

# List all
python rdp/experiment_runner.py list
```

---

## Database Tables

### experiment_runs

Immutable log of every run:

| Field | Type | Purpose |
|-------|------|---------|
| exp_id | text | unique ID (exp_001, exp_002, ...) |
| dataset | text | clean, natural, or raw |
| config | text | baseline or high_quality |
| epochs | int | training epochs |
| batch_size | int | batch size |
| status | text | pending, running, complete, failed |
| scores | jsonb | {naturalness, clarity, identity} when complete |
| created_at | timestamp | when experiment was created |
| completed_at | timestamp | when training finished |

### decisions

Iteration memory (human decisions):

| Field | Type | Purpose |
|-------|------|---------|
| winner_exp_id | text | which experiment performed best |
| loser_exp_id | text | (optional) what it beat |
| reason_summary | text | why it won (e.g., "better clarity") |
| next_planned_exp_id | text | (optional) next to run |
| rationale | text | why that next step |
| created_at | timestamp | when decision was made |

---

## What Success Looks Like

**Week 1:**

```
exp_001 (clean, baseline, 200e)      → 0.751
exp_002 (clean, baseline, 300e)      → 0.782  ← better
exp_003 (natural, baseline, 300e)    → 0.795  ← better
```

**Decision:** Dataset matters. Natural > clean with same config.

**Week 2:**

```
exp_004 (natural, high_quality, 300e) → 0.798  ← small gain
exp_005 (natural, baseline, 400e)     → 0.801  ← bigger gain (epochs)
exp_006 (natural, baseline, 500e)     → 0.799  ← diminishing returns
```

**Decision:** Sweet spot found. 400 epochs, natural, baseline is best.

**Week 3:**

Now run production:

```
exp_007 (natural, baseline, 400e) FINAL
  → deploy model/final.pth
  → use for inference
```

---

## Anti-Patterns (Don't Do)

❌ **Change multiple variables at once**
- Makes results uninterpretable

❌ **Overwrite experiment**
- Lose what actually ran

❌ **Use different test sets for different experiments**
- Comparison becomes meaningless

❌ **Skip decision log**
- Lose iteration memory

❌ **Run 10 experiments then compare**
- You'll forget which change helped

✓ **Run 1, evaluate, decide, run 2**
- Builds understanding step by step

---

## FAQ

**Q: Can I run exp_003 before exp_002 finishes?**
A: Yes. You can parallelize. But only evaluate and compare AFTER both complete.

**Q: What if I run exp_004 but decide it's worse?**
A: That's OK. Go back to exp_003 config for exp_005. Decision log captures this.

**Q: Can I delete an experiment?**
A: No. Everything is immutable. If you made a mistake, create exp_next with correct config.

**Q: How do I know when experiment is done?**
A: Status in experiment_runs table changes to "complete". Scores populated.

**Q: What if test results don't match expectations?**
A: Document in decision log. This is how you learn. Maybe:
- Dataset quality issues
- Model needs more data
- Config not right for task

---

## Your Next 10 Steps

1. Create exp_001 (clean, baseline, 200e)
2. Wait for it to complete
3. Review scores in table
4. Create exp_002 (natural, baseline, 200e) ← ONLY dataset changes
5. Wait for completion
6. Compare: which dataset better?
7. Record decision in "Decisions" table
8. Create exp_003 based on decision
9. Repeat steps 2–7
10. After 3–5 experiments, you'll see clear pattern

**Then you'll know what actually matters for YOUR voice data.**
