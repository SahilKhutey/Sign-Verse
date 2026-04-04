# Data Management Documentation

## Overview

This document describes the data layer architecture, storage formats, versioning strategy, and data processing pipelines for the SignVerse system.

## Data Architecture

### Storage Hierarchy

```text
data/
├── raw/                 # Original, immutable data
│   ├── uploads/         # User-uploaded videos
│   └── youtube_videos/  # Downloaded YouTube content
├── processed/           # Derived data
│   ├── frames/          # Extracted video frames
│   ├── poses/           # 2D/3D pose estimations
│   └── annotations/     # Auto-generated annotations
├── labeled/             # Human-annotated data
│   ├── actions/         # Temporal action labels
│   └── joints/          # Spatial joint annotations
└── datasets/            # Curated training datasets
    ├── train/           # Training splits
    ├── val/             # Validation splits
    └── test/            # Testing splits
```

### Data Formats

#### Video Files
- **Formats**: MP4, AVI, MOV, MKV
- **Resolution**: 720p+ recommended
- **FPS**: 30 FPS standard
- **Metadata**: EXIF and custom JSON metadata

#### Pose Data (Unified Skeleton Format)
The SignVerse system uses a unified, high-fidelity JSON schema for pose data, ensuring consistency across perception, training, and storage.

```json
{
  "frame_id": 102,
  "timestamp": 3.42,
  "person_id": "P1",
  "source_video": "video_001.mp4",

  "body": {
    "nose": {"x": 0.512, "y": 0.234, "z": 0.001, "confidence": 0.923},
    "left_shoulder": {"x": 0.412, "y": 0.345, "z": -0.002, "confidence": 0.876},
    "right_shoulder": {"x": 0.608, "y": 0.342, "z": -0.001, "confidence": 0.881},
    "left_elbow": {"x": 0.321, "y": 0.456, "z": 0.003, "confidence": 0.812},
    "right_elbow": {"x": 0.689, "y": 0.451, "z": 0.002, "confidence": 0.804}
  },

  "left_hand": {
    "wrist": {"x": 0.289, "y": 0.512, "z": 0.015, "confidence": 0.765},
    "thumb_tip": {"x": 0.267, "y": 0.489, "z": 0.023, "confidence": 0.712}
  },

  "right_hand": {
    "wrist": {"x": 0.723, "y": 0.508, "z": 0.014, "confidence": 0.781},
    "thumb_tip": {"x": 0.745, "y": 0.491, "z": 0.021, "confidence": 0.723}
  },

  "face": {
    "landmarks": [
      {"x": 0.501, "y": 0.201, "z": 0.005, "confidence": 0.892},
      {"x": 0.489, "y": 0.198, "z": 0.006, "confidence": 0.876}
    ],
    "expression": {
      "smile": 0.82,
      "blink": 0.12,
      "surprise": 0.05
    }
  },

  "head_pose": {
    "yaw": 10.2,
    "pitch": -3.1,
    "roll": 1.5,
    "confidence": 0.87
  },

  "overall_confidence": 0.845,
  "tracking_quality": 0.912,
  "processing_time": 0.045,
  "resolution": [1920, 1080]
}
```

#### Dataset Manifest
```json
{
  "name": "signlanguage-v1",
  "version": "1.0.0",
  "created_at": "2023-10-15T10:30:00Z",
  "description": "Sign language action dataset",
  "source_data": ["video1.mp4", "video2.mp4"],
  "statistics": {
    "total_samples": 1000,
    "action_distribution": {"hello": 150, "thank_you": 200},
    "average_sequence_length": 45.2
  },
  "splits": {
    "train": 700,
    "val": 150,
    "test": 150
  }
}
```

## Data Versioning

### DVC (Data Version Control)
```bash
# Initialize DVC
dvc init

# Add data directory
dvc add data/raw/
dvc add data/processed/

# Track with Git
git add data/raw.dvc data/processed.dvc .dvc/

# Push to remote storage
dvc remote add -d myremote s3://signverse-data
dvc push
```

### Versioning Strategy
- **Raw Data**: Immutable, never modified
- **Processed Data**: Versioned with processing parameters
- **Datasets**: Snapshotted with manifest files
- **Models**: Versioned with training configurations

## Data Processing

### ETL Pipeline
```python
# Extract
video_path = "data/raw/uploads/video.mp4"
frames = extract_frames(video_path, fps=30)

# Transform
poses = estimate_poses(frames)
normalized = normalize_poses(poses)
augmented = augment_data(normalized)

# Load
save_path = "data/processed/poses/video_poses.json"
save_poses(normalized, save_path)
```

### Quality Control
```python
# Data validation
validate_video(video_path, {
    "min_duration": 1.0,
    "max_duration": 300.0,
    "min_resolution": [640, 480],
    "max_size": 100 * 1024 * 1024  # 100MB
})

# Pose data validation
validate_pose_data(pose_data, schema=pose_schema)
```

## Data Security

### Access Control

| Data Type | Access Level | Encryption | Retention |
| :--- | :--- | :--- | :--- |
| Raw Videos | Admin+ | At rest | 30 days |
| Processed Data | Users+ | At rest | 90 days |
| Annotations | Public | None | Permanent |
| Models | Admin+ | At rest | Permanent |

### Privacy Protection
```python
# Anonymization
def anonymize_video(video_path):
    # Blur faces
    # Remove metadata
    # Hash identifiers
    return anonymized_path
```

## Data Operations

### Backup Strategy
```bash
# Daily incremental backups
dvc push --remote myremote

# Weekly full backups
tar -czf backup_$(date +%Y%m%d).tar.gz data/
aws s3 cp backup_*.tar.gz s3://signverse-backups/
```

### Storage Optimization
```bash
# Data compression
compressed = compress_poses(poses, method="zstd")

# Data deduplication
deduplicated = remove_duplicate_frames(frames)
```

## Metadata Management

### Catalog System
```python
# Data catalog
catalog = DataCatalog("data/metadata/catalog.db")
catalog.add_video("video1.mp4", {
    "source": "upload",
    "duration": 45.2,
    "resolution": [1280, 720],
    "subjects": ["person1"],
    "actions": ["hello", "thank_you"]
})
```

### Search and Query
```python
# Find videos with specific actions
videos = catalog.search({
    "actions": ["hello"],
    "min_duration": 10,
    "max_duration": 60
})
```

## Data Governance

### Compliance
- **GDPR**: Right to be forgotten
- **CCPA**: Data access requests
- **HIPAA**: Medical data protection (if applicable)

### Audit Logging
```python
# Data access logging
logger.info("Data accessed", extra={
    "user": "user123",
    "file": "video1.mp4",
    "action": "download",
    "timestamp": "2023-10-15T10:30:00Z"
})
```

## Performance Considerations

### Storage Backends

| Backend | Use Case | Performance | Cost |
| :--- | :--- | :--- | :--- |
| Local SSD | Processing | Very High | High |
| S3-Compatible | Archive | Medium | Low |
| Glacier | Backup | Low | Very Low |

### Data Pipeline Optimization
```python
# Parallel processing
with ProcessPoolExecutor() as executor:
    results = executor.map(process_video, video_files)

# Memory mapping
poses = np.memmap("poses.dat", dtype=np.float32, mode="r")
```

## Troubleshooting

### Common Issues
- **Corrupted Videos**: Use `ffmpeg -v error -i input.mp4 -f null -`
- **Missing Frames**: Check frame extraction logs
- **Invalid Poses**: Validate against schema
- **Storage Full**: Implement cleanup policies

### Recovery Procedures
```bash
# Restore from backup
dvc pull --remote myremote

# Rebuild processed data
python scripts/run_pipeline.py --rebuild
```
