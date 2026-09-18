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
5. Keep only reviews from a short list of universally familiar product
   categories (`filter_familiar_categories`, e.g. Fashion, Electronics,
   Grocery, Home & Living, Beauty — see `src.config.ALLOWED_CATEGORIES`).
6. Clean and deduplicate (`clean_and_dedupe`).
7. Subsample down to a small, balanced corpus across (label × language)
   (`subsample_balanced`) — the raw source is millions of reviews, but the
   project intentionally keeps the working dataset small per its scope.
8. Write the assembled corpus to `raw/dataset.csv`, then split into
   `processed/{train,val,test}.csv`, stratified by `(label, language)`.

Run the whole pipeline with:

```bash
python scripts/build_dataset.py
```

This downloads a multi-GB raw file on first run and can take a while — per `CLAUDE.md`'s
"long-running commands" ground rule, this is a command to run yourself in your own terminal
once the pipeline code is implemented, not something the assistant should execute inline.

## `raw/`
`dataset.csv` — the full assembled (labeled, language-tagged, filtered,
cleaned, subsampled) corpus before splitting.

## `processed/`
`train.csv`, `val.csv`, `test.csv` — stratified splits, produced by
`src.data.dataset_builder.split_dataset()`.

## Schema

| column             | type | description                                                                 |
|--------------------|------|-------------------------------------------------------------------------------|
| `id`               | str  | unique row id                                                                 |
| `product_category` | str  | product category from BanglishRev metadata (filtered to familiar ones)       |
| `rating`           | int  | original 1-5 star rating from the review                                     |
| `label`            | str  | derived sentiment: one of `positive`, `negative`, `neutral`                  |
| `language`         | str  | auto-detected: one of `english`, `bangla`, `banglish`, `code_switched`       |
| `text`             | str  | the review content                                                            |

Both `raw/` and `processed/` contents are git-ignored (only `.gitkeep`
placeholders are tracked) — regenerate them locally by running
`python scripts/build_dataset.py` once Phase 1 (see `CLAUDE.md`) is implemented.

## Citation

If reporting results derived from this data, cite the source dataset:

> BanglishRev: A Large-Scale Bangla-English and Code-mixed Dataset of
> Product Reviews in E-Commerce. arXiv:2412.13161.
