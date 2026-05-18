# Category Graph Space Analysis

## Graph Statistics
- pog count train: edges=1059, density=0.414, largest_component=72/72
- pog binary train: edges=1059, density=0.414, largest_component=72/72

## Completion Summary
### pog
- set_direct_confidence: hit@1=0.455, hit@3=0.765, hit@5=0.878, MRR=0.634, coverage=0.999
- graph_conditional_count: hit@1=0.461, hit@3=0.760, hit@5=0.868, MRR=0.635, coverage=1.000
- graph_conditional_binary: hit@1=0.461, hit@3=0.760, hit@5=0.869, MRR=0.634, coverage=1.000
- graph_raw_count: hit@1=0.453, hit@3=0.750, hit@5=0.864, MRR=0.628, coverage=1.000
- graph_raw_binary: hit@1=0.453, hit@3=0.750, hit@5=0.865, MRR=0.628, coverage=1.000
- graph_ppmi_count: hit@1=0.101, hit@3=0.310, hit@5=0.483, MRR=0.279, coverage=1.000
- graph_ppmi_binary: hit@1=0.101, hit@3=0.310, hit@5=0.484, MRR=0.280, coverage=1.000
- text_centroid_similarity: hit@1=0.008, hit@3=0.029, hit@5=0.087, MRR=0.093, coverage=1.000

## Text vs Graph Hit@3 Quadrants
- pog both_hit: 87 (0.022)
- pog text_only: 28 (0.007)
- pog graph_only: 2953 (0.738)
- pog both_miss: 932 (0.233)

Interpretation: text centroid similarity measures semantic category similarity, while category graph profiles measure bundle complementarity. A large graph-only quadrant supports the claim that category completion is structural rather than semantic similarity.
