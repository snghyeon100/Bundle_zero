# Candidate Category Embedding Signal

Scores are cosine similarities between the input-category mean embedding and each candidate item's category embedding.

| dataset | n | mean GT score | mean max distractor | mean margin | item hit@1 | item hit@3 | category hit@1 | category hit@3 | mean item rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pog | 8411 | 0.509 | 0.816 | -0.306 | 0.165 | 0.462 | 0.199 | 0.581 | 4.19 |
| pog_dense | 9185 | 0.451 | 0.873 | -0.422 | 0.118 | 0.338 | 0.162 | 0.471 | 4.94 |
