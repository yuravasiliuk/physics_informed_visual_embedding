# RT-H — Action Hierarchies Using Language

Belkhale, Ding, Xiao, Sermanet, Vuong, Tompson, Chebotar, Dwibedi, Sadigh. Google DeepMind / Stanford. arXiv 2403.01823v2, Jun 2024. [PDF](../papers/2403.01823v2.pdf).

> **Note**: only abstract / first page examined.

## TL;DR
A two-level **language-conditioned robot policy**:
1. From `(task, image)` predict **language motions** (short phrases like "move arm forward," "rotate arm right," "close gripper") using a VLM.
2. Conditioned on the predicted language motion *and* the task, predict the low-level robot action.

The language-motion intermediate layer (a) shares structure across semantically diverse tasks (so the policy gets more out of multi-task data), and (b) provides a natural intervention point — humans can correct the policy in language, and those corrections can be used as new supervision.

## Why it's in the reading pile
Tangential to the brainstorm. RT-H is about *output* hierarchies for robot policies, not about *input* representations or simulator supervision. Relevance is limited to:
- An example of how **intermediate symbolic / language-grounded variables** between observation and action can improve generalization across tasks (parallel to using physical quantities as auxiliary heads).
- If the brainstorm's encoder ever feeds a VLA model, RT-H is a candidate downstream consumer.

## Not deeply read
Action representation, training datasets (RT-X family), intervention-learning protocol, and ablations are not summarized here.

Website: https://rt-hierarchy.github.io/
