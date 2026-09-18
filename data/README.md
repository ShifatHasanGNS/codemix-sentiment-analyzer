# Data Directory

## Source

The corpus is built entirely by code from **BanglishRev**, a large-scale,
real, publicly available dataset of Bangla / English / Banglish /
code-mixed **e-commerce product reviews** (~1.74M reviews across ~128k
products), released on Hugging Face under CC-BY-NC-SA-4.0:
<https://huggingface.co/datasets/BanglishRev/bangla-english-and-code-mixed-ecommerce-review-dataset>

Product reviews were chosen because they're a topic anyone can understand
without domain expertise (satisfies the "corpus must be familiar to
everyone" requirement), and because the whole collection pipeline can run
unattended via code — no manual writing, translation, or labeling required.

**Text only.** This project works on review text only. BanglishRev's raw records are
per-product and may carry product-image fields alongside the reviews; `flatten_reviews()`
must drop those and keep only the text/metadata columns in the schema below. No image
files are downloaded, stored, or used anywhere in this project.

> Non-commercial academic use only, per the dataset's CC-BY-NC-SA-4.0
> license. Cite the original paper when reporting results (see below).

## Pipeline (see `src/data/dataset_builder.py`)

1. Download the raw dataset (`download_banglishrev`).
2. Flatten the nested per-product JSON into one row per review (`flatten_reviews`).
3. Derive a 3-class sentiment label from the review's star rating
   (`map_rating_to_label`): 1-2 stars → `negative`, 3 stars → `neutral`,
   4-5 stars → `positive`.
4. Auto-tag each review's language condition (`detect_language_condition`)
   into one of `english`, `bangla`, `banglish`, `code_switched`, using a
   Unicode-script + dictionary-based heuristic (see `src/config.py` for the
   exact thresholds).
5. `filter_familiar_categories()` — a no-op by default (see "Known
   limitations" below for why).
6. Clean and deduplicate (`clean_and_dedupe`).
7. Subsample down to a large, label-balanced corpus across (label × language)
   (`subsample_balanced`) — target size and balancing strategy documented
   in `src/config.py`.
8. Write the assembled corpus to `raw/dataset.csv`, then split into a
   held-out test set + a k-fold CV pool (`processed/`).

Run the whole pipeline with:

```bash
python scripts/build_dataset.py
```

This downloads a multi-GB raw file on first run and can take a while — per `CLAUDE.md`'s
"long-running commands" ground rule, this is a command to run yourself in your own terminal
once the pipeline code is implemented, not something the assistant should execute inline.
At the current target size, this must also flatten/tag the full ~1.74M raw reviews (not a
small sample), so expect a few minutes even with the raw file cached locally.

## `raw/`
`dataset.csv` — the full assembled (labeled, language-tagged, filtered,
cleaned, subsampled) corpus before splitting.

## `processed/`
- `cv_pool.csv` — 90% of the corpus, used for k-fold cross-validation. Has
  every column in the schema below plus a `fold` int column (0..N_FOLDS-1,
  see `src.config.N_FOLDS`); `src.data.dataset_builder.get_fold()` splits
  it into a given fold's (train, val) pair.
- `test.csv` — the remaining 10%, held out from all training/CV and used
  only for the final, unbiased evaluation (`results/metrics_comparison.csv`).

Both are produced by `src.data.dataset_builder.create_cv_splits()`.

## Schema

| column             | type | description                                                                 |
|--------------------|------|-------------------------------------------------------------------------------|
| `id`               | str  | unique row id                                                                 |
| `product_category` | str  | product category from BanglishRev metadata                                   |
| `rating`           | int  | original 1-5 star rating from the review                                     |
| `label`            | str  | derived sentiment: one of `positive`, `negative`, `neutral`                  |
| `language`         | str  | auto-detected: one of `english`, `bangla`, `banglish`, `code_switched`       |
| `text`             | str  | the review content                                                            |

`cv_pool.csv` additionally has a `fold` int column (dropped by `get_fold()`
before returning a fold's train/val DataFrames).

Both `raw/` and `processed/` contents are git-ignored (only `.gitkeep`
placeholders are tracked) — regenerate them locally by running
`python scripts/build_dataset.py` once Phase 1 (see `CLAUDE.md`) is implemented.

## Known limitations

- **Category scope was broadened from an original 5-category "familiar
  topics" design.** Sampling the raw corpus showed the original narrow
  keyword-matched allowlist (Fashion/Electronics/Grocery/Home & Living/Beauty)
  excluded ~74% of it, including everyday items (Smartwatches, T-Shirts,
  Shampoo, Toothpaste, Diapers, Coffee) that are just as "universally
  familiar" as the original 5 buckets. `filter_familiar_categories()` is
  now a no-op by default (`src.config.ALLOWED_CATEGORIES = None`) — the
  whole corpus is ordinary e-commerce consumer goods by construction. The
  function still accepts an explicit `allowed_categories` list if anyone
  wants to re-narrow the scope.
- **Code-switched reviews are intrinsically rare in this corpus**: only
  ~1,600 total across the full ~1.74M raw reviews (~0.14%), regardless of
  category scope — this is a property of the underlying data as detected
  by this project's language-detection heuristic, not a filtering
  artifact. `subsample_balanced()` uses all of it rather than forcing an
  equal share alongside English/Bangla/Banglish (which would mean either
  duplicating data, violating the "100% code-driven, no fabricated data"
  rule, or capping the whole corpus at a few thousand rows). Treat
  code-switched metrics as the noisiest of the four language conditions
  when interpreting results — its test-set slice is smaller than the other
  three.

## Citation

If reporting results derived from this data, cite the source dataset:

> BanglishRev: A Large-Scale Bangla-English and Code-mixed Dataset of
> Product Reviews in E-Commerce. arXiv:2412.13161.
