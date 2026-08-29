# ADR-002: VL-JEPA Contrastive Collapse

## Status: Accepted (negative result)

## Context
Cross-modal alignment (image-HV <-> caption-HV) via contrastive
InfoNCE training on COCO val (2400 train / 600 held-out).

## Decision
Three variants all reached held-out acc@1 ≈ chance (0.0017):
1. Handcrafted hashes + InfoNCE: memorized (train 1.00), held-out 0.003
2. ResNet18-frozen + InfoNCE: collapsed to point (logit_std 0.83→0.007)
3. ResNet18 + InfoNCE + Var-barrier: collapse prevented, no structure found

## Rationale
Both sides of the bridge are semantically empty — frozen ResNet without
joint training and random VSA word basis share no overlapping statistics.
CLIP-scale alignment requires millions of pairs and large encoders.

## Consequences
- Path forward: pretrained CLIP/ViT-B encoder, or full COCO train2017 + GPU
- Vision facts in PersistentVSAMemory remain useful for intra-modal search
