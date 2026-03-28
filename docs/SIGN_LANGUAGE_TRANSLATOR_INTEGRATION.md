# sign-language-translator Integration (Concatenative + Embeddings)

This integration brings key capabilities inspired by:
- `sign-language-translator/sign-language-translator` (Apache-2.0)

Added to SignVerse:
- Modular concatenative synthesis engine (`text/gloss -> clip plan -> sentence video`)
- Synthetic sentence-level sign video dataset generation
- MediaPipe-based video embedding extractor for training pipelines

## New Components

- Concatenative planner + renderer:
  - `nlp_translation/concatenative_synthesis.py`
- Dictionary manifest builder:
  - `training/data_pipeline/build_sign_video_dictionary.py`
- Synthetic sentence video generator:
  - `training/data_pipeline/generate_synthetic_sign_video_pairs.py`
- Video embedding extractor:
  - `vision_pipeline/video_embedding.py`
- Embedding dataset script:
  - `training/data_pipeline/extract_video_embeddings.py`
- API endpoint:
  - `POST /translate/text-to-sign-video-plan`

## 1) Build Dictionary Manifest

Prepare sign clips in either layout:

```text
datasets/sign_dictionary/clips/
  HELLO/*.mp4
  HOW/*.mp4
  YOU/*.mp4
```

Then build manifest:

```bash
python training/data_pipeline/build_sign_video_dictionary.py --clips-root datasets/sign_dictionary/clips --output-manifest datasets/sign_dictionary/manifest.csv
```

## 2) Generate Synthetic Sentence Videos

Requires:
- text/gloss pairs CSV (default: `datasets/text_sign_pairs/expanded_pairs.csv`)
- dictionary manifest from step 1

```bash
python training/data_pipeline/generate_synthetic_sign_video_pairs.py --pairs-csv datasets/text_sign_pairs/expanded_pairs.csv --dictionary-manifest datasets/sign_dictionary/manifest.csv --output-dir datasets/synthetic_sentence_videos --output-manifest training-data/synthetic_sign_video_pairs.csv --min-coverage 0.6 --max-samples 500
```

Outputs:
- `training-data/synthetic_sign_video_pairs.csv`
- `reports/synthetic_sign_video_report.json`

## 3) Extract Video Embeddings

From isolated sign videos:

```bash
python training/data_pipeline/extract_video_embeddings.py --input-dir datasets/isolated_videos --output-dir training-data/video_embeddings --manifest training-data/video_embeddings_manifest.csv --sample-every 2 --max-frames 120
```

Embedding format:
- mean(frame_features) + std(frame_features)
- default 450 dimensions

## 4) API: Text -> Sign Video Plan

Endpoint:
- `POST /translate/text-to-sign-video-plan`

Body:

```json
{
  "text": "hello how are you",
  "dictionary_manifest": "datasets/sign_dictionary/manifest.csv",
  "strict_manifest": false
}
```

Response includes:
- translated tokens
- clip plan per token
- coverage and missing tokens

## Notes

- This path is modular and production-friendly for fast iteration.
- It complements neural models by generating synthetic aligned data.
- Quality depends on dictionary clip coverage and clip consistency.
