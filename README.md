# Parameter Golf Experiments

Training the smallest language model that fits in 16MB — [OpenAI's Parameter Golf challenge](https://github.com/openai/parameter-golf).

## Submissions (8xH100, official leaderboard)

| Version | val_bpb | Artifact Size | Key Ideas | PR |
|---|---|---|---|---|
| **v2: ContextFuse-BigramSmear** | **1.1537** | 15.3 MB | BigramHash, SmearGate, int6 export, SWA, Muon | [#174](https://github.com/openai/parameter-golf/pull/174) |
| v1: ContextFuse-2048 | 1.1779 | 15.9 MB | 9L/512d, GQA, RoPE, seq2048, FP16 tied embed | [#143](https://github.com/openai/parameter-golf/pull/143) |

## Budget-Constrained Experiments (consumer GPUs)

Running the same architectures on cheap hardware to study training efficiency under extreme budget constraints.

| Experiment | GPU | Cost | val_bpb | Notes |
|---|---|---|---|---|
| A5000 port (baseline) | RTX A5000 | ~$0.77 | 1.4972 (stop-time) | Single-GPU port, found EMA export bug |

## Repo Structure

```
submissions/           # Official leaderboard submissions (8xH100)
  v1-contextfuse-2048/
  v2-contextfuse-bigramsmear/
experiments/           # Custom ports and research
  a5000-port/          # Single-GPU RTX A5000 port
records/               # Full experiment history with logs
  track_10min_16mb/    # 22 ablation runs on the official track
  track_non_record_16mb/  # Non-record experiments and ports
```

## Highlights

- **22 ablation runs** covering architecture search, quantization strategies, optimizer tuning, and evaluation fixes
- **Single-GPU port** proving the architecture works on a $0.25/hr RTX A5000
- **EMA export bug discovery** — stop-time weights (1.4972) beat EMA (1.6124) on A5000; raw averaging controls added
- Total budget for consumer GPU experiments: **< $1.00**

## Architecture Evolution

`NaiveBaseline` → `FP16Embed` → `Seq2048` → `SlidingWindow` → `MixedPrecision` → `SmearGate` → `BigramHash` → `XSA` → `EMA+GPTQ-lite` → `LeakyReLU+LegalTTT+ParallelMuon`

Each step documented with training logs, READMEs, and submission.json in the records directory.

## Links

- [Parameter Golf challenge](https://github.com/openai/parameter-golf)
- [PR #143 (v1)](https://github.com/openai/parameter-golf/pull/143)
- [PR #174 (v2)](https://github.com/openai/parameter-golf/pull/174)
