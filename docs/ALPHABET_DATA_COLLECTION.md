# Alphabet Dataset Collection (OpenCV)

This guide adds a practical webcam data collection flow inspired by notebook-based
alphabet recognizer projects (ASL/ISL style).

Script:
- `scripts/collect_alphabet_dataset.py`

## Output Structure

```text
datasets/asl_alphabet/
  train/
    A/*.jpg
    B/*.jpg
    ...
  val/
    A/*.jpg
    B/*.jpg
    ...
  capture_log.csv
```

## Quick Start

Capture A-Z with random train/val split:

```bash
python scripts/collect_alphabet_dataset.py --camera 0 --flip --labels "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z" --split-mode random --val-ratio 0.1 --image-size 224
```

Capture custom ISL label set:

```bash
python scripts/collect_alphabet_dataset.py --camera 0 --labels "AA,AE,KA,KHA,GA,GHA" --split-mode random
```

## Controls

- `SPACE`: Save one sample from ROI
- `A`: Toggle auto-capture
- `N`/`]`: Next label
- `P`/`[`: Previous label
- `T`: Force train split
- `V`: Force val split
- `R`: Random split mode
- `U`: Undo last saved sample
- `Q`: Quit

## Recommended Capture Settings

- Keep hand centered in ROI box.
- Vary lighting, background, and hand orientation.
- Capture multiple distances and small rotations.
- Aim for balanced class counts across labels.
- Use `--max-per-label` for controlled balanced sessions.

## Training After Collection

Train InceptionV3 alphabet baseline:

```bash
python training/train_asl_cnn_keras.py --data-dir datasets/asl_alphabet --arch inceptionv3 --image-size 224 --epochs 15 --fine-tune-epochs 5
```

Run live finger-spelling:

```bash
python scripts/run_asl_cnn_live.py --camera 0 --spell-mode --min-confidence 0.5
```
