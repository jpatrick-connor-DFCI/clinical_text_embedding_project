"""Explore and model patient-level pooled note embeddings.

Deliberately kept import-light at package level so `python -m
semantic_search.<stage>` does not pay for numpy/sklearn/lifelines before the
stage that needs them.  Import the stage modules directly.
"""
