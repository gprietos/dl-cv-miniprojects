# Recommended additions
1. Per-class metrics

Very useful for imbalance:

average=None

2. Top-k accuracy

For ImageNet-style tasks:

MulticlassAccuracy(
    num_classes=num_classes,
    top_k=5
)

3. AUROC support

For binary or multilabel:

MulticlassAUROC
BinaryAUROC