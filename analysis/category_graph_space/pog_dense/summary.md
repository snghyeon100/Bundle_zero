# Category Graph Space Analysis

## Graph Statistics
- pog_dense count train: edges=766, density=0.346, largest_component=67/67
- pog_dense binary train: edges=766, density=0.346, largest_component=67/67

## Completion Summary
### pog_dense
- set_direct_confidence: hit@1=0.465, hit@3=0.787, hit@5=0.875, MRR=0.643, coverage=0.997
- graph_conditional_count: hit@1=0.450, hit@3=0.767, hit@5=0.851, MRR=0.627, coverage=1.000
- graph_conditional_binary: hit@1=0.450, hit@3=0.767, hit@5=0.851, MRR=0.627, coverage=1.000
- graph_raw_binary: hit@1=0.427, hit@3=0.732, hit@5=0.836, MRR=0.605, coverage=1.000
- graph_raw_count: hit@1=0.427, hit@3=0.732, hit@5=0.836, MRR=0.605, coverage=1.000
- graph_ppmi_count: hit@1=0.097, hit@3=0.293, hit@5=0.435, MRR=0.267, coverage=1.000
- graph_ppmi_binary: hit@1=0.097, hit@3=0.293, hit@5=0.435, MRR=0.269, coverage=1.000
- lightgcn_bi_category_similarity: hit@1=0.068, hit@3=0.193, hit@5=0.308, MRR=0.197, coverage=1.000

## Text vs Graph Hit@3 Quadrants
- pog_dense both_hit: 29 (0.005)
- pog_dense text_only: 19 (0.003)
- pog_dense graph_only: 4523 (0.762)
- pog_dense both_miss: 1367 (0.230)

Interpretation: text centroid similarity measures semantic category similarity, while category graph profiles measure bundle complementarity. A large graph-only quadrant supports the claim that category completion is structural rather than semantic similarity.
