# All Feature Signal Comparison

| dataset | feature | GT top1 | GT top3 | MRR | rank mean | GT sim | neg mean | margin vs best neg | pred=feature top1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pog | openai_text | 0.132 | 0.344 | 0.328 | 5.09 | 0.434 | 0.421 | -0.118 | 0.160 |
| pog | item_cf | 0.076 | 0.332 | 0.292 | 5.25 | 0.009 | 0.001 | -0.193 | 0.088 |
| pog | bi_lgcn | 0.184 | 0.408 | 0.378 | 4.76 | 0.096 | -0.004 | -0.167 | 0.144 |
| pog_dense | openai_text | 0.072 | 0.284 | 0.269 | 5.65 | 0.456 | 0.460 | -0.120 | 0.104 |
| pog_dense | item_cf | 0.152 | 0.348 | 0.324 | 5.69 | -0.014 | 0.071 | -0.532 | 0.148 |
| pog_dense | bi_lgcn | 0.368 | 0.676 | 0.561 | 3.05 | 0.586 | -0.033 | -0.245 | 0.200 |
| spotify | openai_text | 0.720 | 0.892 | 0.814 | 1.78 | 0.542 | 0.405 | 0.051 | 0.728 |
| spotify | item_cf | 0.204 | 0.440 | 0.390 | 4.65 | 0.070 | 0.010 | -0.133 | 0.196 |
| spotify | bi_lgcn | 0.272 | 0.628 | 0.493 | 3.40 | 0.680 | 0.143 | -0.319 | 0.268 |
| spotify_sparse | openai_text | 0.728 | 0.896 | 0.822 | 1.70 | 0.555 | 0.398 | 0.066 | 0.729 |
| spotify_sparse | item_cf | 0.204 | 0.416 | 0.391 | 4.50 | 0.081 | 0.011 | -0.119 | 0.206 |
| spotify_sparse | bi_lgcn | 0.528 | 0.740 | 0.662 | 2.85 | 0.588 | 0.044 | -0.175 | 0.522 |

## Best by GT Top1

| dataset | best feature | GT top1 | rank mean | margin vs best neg |
|---|---|---:|---:|---:|
| pog | bi_lgcn | 0.184 | 4.76 | -0.167 |
| pog_dense | bi_lgcn | 0.368 | 3.05 | -0.245 |
| spotify | openai_text | 0.720 | 1.78 | 0.051 |
| spotify_sparse | openai_text | 0.728 | 1.70 | 0.066 |
