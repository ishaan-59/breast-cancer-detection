# Model checkpoints

Checkpoint binaries are intentionally omitted from Git because they are large
generated artifacts. No alternate checkpoint should be substituted silently.

## Required for end-to-end inference

| Filename | Bytes | SHA-256 |
|---|---:|---|
| `resnet50_layer4_best.pth` | 213,929,049 | `0f7a7bf459d2706acff9539ef1acc5b231b1ea8ec81298c4e84ba7eea61ff7ca` |
| `resnet50_unet_layer4_best.pth` | 422,660,587 | `2db5afca774cd137fb350325f0e23d6fa1fb8b412f555a79569d8c44fcd34f1c` |

## Other experiment checkpoints (also omitted)

| Filename | Bytes | Purpose | SHA-256 |
|---|---:|---|---|
| `unet_single_image_overfit_best.pth` | 93,175,393 | Sanity-check overfit | `46f29721be4d54473a8c5d408857bc2f244711a2334799e7d00a90603ec6707b` |
| `unet_best.pth` | 93,228,683 | Vanilla U-Net baseline | `45f6a59b99bcbd3483ec03bbd21b5c2219d3806c8c8118e6a18dd83145fab5db` |
| `resnet50_unet_decoder_best.pth` | 303,114,343 | Frozen-encoder transfer stage | `99b5f2e2ba54721733ab8b8a0a1a8362b7e49bcfbee52da70d46360c0a6606a9` |

The hashes identify the supplied artifacts that were audited during repository
reconstruction. Checkpoints are expected to come from the project owner; this
repository does not currently publish a download URL or use Git LFS.
