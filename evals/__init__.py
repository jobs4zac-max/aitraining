"""DeepEval-based agent evaluation.

Kept separate from ``tests/`` because these runs cost money and take minutes,
whereas ``tests/`` is meant to stay fast and free.

Entry points:
    python -m udaplay eval          scorecard for every scenario
    pytest evals -m live            the same scenarios as pytest assertions
"""
