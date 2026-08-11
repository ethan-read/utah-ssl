# GRU Representation Accessibility

A five-fold, trial-grouped linear probe compared raw neural input windows with
trained Brain-to-Text 2025 GRU hidden states using identical trials and
fold-local scaling/PCA.

| Representation | Macro-F1 | Balanced accuracy |
|---|---:|---:|
| Raw input windows | `0.188` | `0.380` |
| GRU hidden states | `0.803` | `0.899` |

GRU hidden states also predicted the model's probability outputs much more
accurately (`R²=0.846–0.920`) than raw inputs (`R²=0.088–0.190`). The targets
are the GRU's own beliefs, not independent phoneme labels, so this demonstrates
representational accessibility rather than standalone decoding accuracy.
