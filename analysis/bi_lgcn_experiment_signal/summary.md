# POG BI-LightGCN Experiment Signal

| dataset | method | hit | GT top1 | GT top3 | MRR | rank mean | margin vs best neg | pred=input top1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pog | base | 0.328 | 0.184 | 0.408 | 0.378 | 4.76 | -0.167 | 0.144 |
| pog | IBxBI_item_desc | 0.296 | 0.184 | 0.408 | 0.378 | 4.76 | -0.167 | 0.152 |
| pog | IUxUI_item_desc | 0.252 | 0.184 | 0.408 | 0.378 | 4.76 | -0.167 | 0.132 |
| pog | BIxIB_bundle_context | 0.324 | 0.184 | 0.408 | 0.378 | 4.76 | -0.167 | 0.128 |
| pog | co_occur | 0.340 | 0.184 | 0.408 | 0.378 | 4.76 | -0.167 | 0.160 |
| pog_dense | base | 0.336 | 0.368 | 0.676 | 0.561 | 3.05 | -0.245 | 0.200 |
| pog_dense | IBxBI_item_desc | 0.388 | 0.368 | 0.676 | 0.561 | 3.05 | -0.245 | 0.296 |
| pog_dense | IUxUI_item_desc | 0.228 | 0.368 | 0.676 | 0.561 | 3.05 | -0.245 | 0.152 |
| pog_dense | BIxIB_bundle_context | 0.452 | 0.368 | 0.676 | 0.561 | 3.05 | -0.245 | 0.272 |
| pog_dense | co_occur | 0.576 | 0.368 | 0.676 | 0.561 | 3.05 | -0.245 | 0.312 |
