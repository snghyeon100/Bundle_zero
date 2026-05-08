# OpenAI Embedding Semantic Signal Summary

Model: `text-embedding-3-large`

| dataset | n | LLM hit | GT top1 | GT top3 | MRR | rank mean | GT sim | neg mean | margin vs best neg | pred=semantic top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pog | 250 | 0.328 | 0.132 | 0.344 | 0.328 | 5.09 | 0.434 | 0.421 | -0.118 | 0.160 |
| pog_dense | 250 | 0.336 | 0.072 | 0.284 | 0.269 | 5.65 | 0.456 | 0.460 | -0.120 | 0.104 |
| spotify | 250 | 0.820 | 0.720 | 0.892 | 0.814 | 1.78 | 0.542 | 0.405 | 0.051 | 0.728 |
| spotify_sparse | 250 | 0.820 | 0.728 | 0.896 | 0.822 | 1.70 | 0.555 | 0.398 | 0.066 | 0.729 |
