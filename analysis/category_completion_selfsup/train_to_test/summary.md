# Category Completion Self-supervised Evaluation

Rules are learned from `bi_train.txt`; evaluation uses `bi_test_input.txt` as observed categories and `bi_test_gt.txt` as held-out categories.

| dataset | method | n | coverage | hit@1 | hit@3 | hit@5 | recall@3 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pog | backoff_confidence | 4000 | 1.000 | 0.460 | 0.756 | 0.869 | 0.475 | 0.633 |
| pog | backoff_lift | 4000 | 1.000 | 0.080 | 0.226 | 0.406 | 0.121 | 0.242 |
| pog | backoff_pmi | 4000 | 1.000 | 0.103 | 0.296 | 0.480 | 0.158 | 0.281 |
| pog | direct_confidence | 4000 | 0.996 | 0.455 | 0.765 | 0.879 | 0.487 | 0.634 |
| pog | direct_lift | 4000 | 0.996 | 0.080 | 0.255 | 0.444 | 0.137 | 0.256 |
| pog | direct_pmi | 4000 | 0.996 | 0.080 | 0.255 | 0.444 | 0.137 | 0.256 |
| pog_dense | backoff_confidence | 5938 | 1.000 | 0.443 | 0.749 | 0.850 | 0.615 | 0.619 |
| pog_dense | backoff_lift | 5938 | 1.000 | 0.059 | 0.169 | 0.246 | 0.118 | 0.185 |
| pog_dense | backoff_pmi | 5938 | 1.000 | 0.097 | 0.254 | 0.359 | 0.182 | 0.248 |
| pog_dense | direct_confidence | 5938 | 0.995 | 0.465 | 0.787 | 0.875 | 0.673 | 0.643 |
| pog_dense | direct_lift | 5938 | 0.995 | 0.074 | 0.233 | 0.353 | 0.167 | 0.230 |
| pog_dense | direct_pmi | 5938 | 0.995 | 0.074 | 0.233 | 0.353 | 0.167 | 0.230 |
