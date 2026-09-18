#!/usr/bin/env python
"""
CLI entry point: runs the full pipeline end to end -- dataset check ->
feature/embedding preparation -> training (classical, neural, BERT) ->
evaluation -- with flags to run a subset for faster local iteration.

Usage:
    python scripts/run_pipeline.py [--models ann,lstm,transformer] [--skip-bert]
                                    [--skip-classical] [--epochs N]

This is a long-running command once real training is wired up (see CLAUDE.md's
"long-running commands" ground rule): implement the orchestration below, then
have the human run it themselves and report back when it finishes (or fails),
rather than running a full/partial pipeline inline in an AI session. `--help`
is quick and fine to run directly to check flag parsing.

TODO:
- Parse CLI flags via argparse:
    --models (comma-separated subset of {ngram,bow,tfidf,ann,rnn,lstm,
              attention,transformer,bert}; default: all)
    --skip-bert (bool flag, skips the slower fine-tuning step)
    --epochs (override default epoch count for neural/BERT training)
- Orchestrate, in order:
    1. Verify data/processed/{train,val,test}.csv exist (else instruct the
       user to run scripts/build_dataset.py first).
    2. src.features.embeddings.train_word2vec (if any neural model is selected)
    3. src.training.train_classical.main() for selected classical baselines
    4. src.training.train_neural.main() for selected from-scratch models
    5. src.training.train_bert.main() unless --skip-bert
    6. src.evaluation.metrics + src.evaluation.error_analysis to produce
       results/metrics_comparison.csv, results/metrics_comparison.png,
       results/error_analysis.md
- Print a final summary of what was trained/evaluated and where outputs live.
"""


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
