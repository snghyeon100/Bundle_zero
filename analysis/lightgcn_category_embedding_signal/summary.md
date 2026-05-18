# LightGCN Category Input-GT Cosine Signal

This evaluates cosine similarity between the mean BI-LightGCN category embedding of input categories and held-out GT category embeddings.

| dataset | grouping | n | coverage | mean cosine | median cosine | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pog | dataset | 4000 | 1.000 | 0.692 | 0.804 | 0.282 | 0.502 | 0.616 | 0.437 |
| pog_dense | dataset | 5938 | 1.000 | 0.557 | 0.690 | 0.068 | 0.193 | 0.308 | 0.197 |
