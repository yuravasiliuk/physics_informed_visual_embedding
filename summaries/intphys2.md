# IntPhys 2 — Benchmarking Intuitive Physics in Complex Synthetic Environments

Bordes, Garrido, Kao, Williams, Rabbat, Dupoux. FAIR Meta. arXiv 2506.09849, June 2025. [PDF](../papers/IntPhys2.pdf).

## TL;DR
A photorealistic Unreal-Engine video benchmark for the four canonical infant-cognition principles — **Permanence, Immutability, Spatio-Temporal Continuity, Solidity** — built on the violation-of-expectation paradigm (quadruplets: 2 possible + 2 impossible videos that share content but flip plausibility). 1,416 videos. Human accuracy 96 %; best AI model (Gemini-2.5 Flash) 56 %; predictive models (V-JEPA 2, Cosmos-Predict-4B, VideoMAEv2) all sit at ~50–58 % on the main set and ~49–55 % on the held-out set.

## Why it's in the reading pile
- Likely **target evaluation benchmark** if the brainstorm wants to claim "physics-informed embeddings."
- Diagnoses the failure modes the brainstorm is trying to fix (occlusion handling, short-term memory across moving cameras, distinguishing visually-similar but physically-different events).

## Benchmark design
- **VoE quadruplets**: each scene rendered twice — once with an obstacle, once without — so the same scene becomes possible / impossible depending on whether occluded events are consistent with what should have happened. Eliminates low-level pixel shortcuts.
- **Splits**:
  - **Debug** (60 videos, 5 scenes): bright assets, static cameras, plus near-identical variations for noise sensitivity testing.
  - **Main** (1,012 videos, 253 scenes): easy / medium / hard sub-splits; static and moving cameras.
  - **Held-out** (344 videos): no metadata released; the "real" test set.
- **Innovations over IntPhys 1**: focused on occlusions; dynamic camera movements; photorealistic Unreal scenes; multiple scenes per principle; higher short-term memory demand (longer occlusion durations, up to ~7 s).

## Evaluation protocols
- **Predictive models (V-JEPA, Cosmos, VideoMAE)**: compute per-window surprise = prediction error in pixel or latent space; classify pair as correct if mean-surprise of possible video < impossible video. Two protocols: pairwise (default) and single-video.
- **MLLMs (GPT-4o, Gemini, Qwen-VL)**: two prompt styles (open yes/no, structured 0/1), tested across 10/30/60/120-frame inputs.
- Best-hyperparameter accuracy is reported (slightly inflates models vs. humans).

## Headline results
| Model               | Type        | Easy | Med  | Hard | Overall | Held-out |
|---------------------|-------------|------|------|------|---------|----------|
| Human               | —           | —    | —    | —    | **96.44**| 92.44   |
| Gemini-2.5 Flash    | MLLM        | 64.4 | 56.8 | 54.5 | 55.6    | 56.1     |
| GPT-4o              | MLLM        | 57.7 | 54.8 | 54.2 | 53.8    | 53.2     |
| V-JEPA 2-h          | Predictive  | 54.0 | 58.5 | 59.4 | 57.5    | 56.4     |
| VideoMAEv2-g        | Predictive  | 46.0 | 58.5 | 52.7 | 53.5    | 53.5     |
| Cosmos-Predict-4B   | Predictive  | 46.0 | 52.0 | 48.0 | 49.4    | 48.8     |

By category, the **permanence** condition is easiest for all models (objects not moving on their own). The biggest gap to humans is on **immutability** and **solidity** in moving-camera settings.

## What it tells us for the brainstorm
1. **Strong vision-only world models still miss core physics** — this matches the Opinion paper's argument. Any physics-grounded supervision will be tested first here.
2. **Memory & occlusion** are the weak spots. Designing the brainstorm's supervision around object-permanence-preserving quantities (e.g. tracking CoM through occlusion) would target the largest measurable gap.
3. **Camera motion hurts** — fixed vs. moving cameras shifts results several points. The brainstorm should ensure sim training mixes fixed and dynamic cameras (Unreal/Isaac Sim can do this trivially).
4. The benchmark is *out-of-distribution* for any model trained on it; predicted-future-surprise scoring is what matters, not classification. Good fit for evaluating a JEPA-style embedding + predictor.

## Practical
- Code: https://github.com/facebookresearch/IntPhys2
- Blogpost: https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks
- Suggested companion read: the parallel benchmark **InfLevel** (Weihs et al., 2022) for additional held-out scenarios.

## Caveats from authors
- Unreal-only — sim2real not addressed.
- Frame-count subsampling distorts model performance; humans naturally aggregate full video.
- Memory window of current models < scenario length → expect harder splits to remain near chance until architecture changes.
