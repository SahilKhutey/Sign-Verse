# SignVerse API Documentation

## Base URL

`https://api.signverse.example.com/api/v1`

## Authentication

### API Key Authentication

Include your API key in the request header:

```http
X-API-Key: your_api_key_here
```

### Rate Limiting

- 100 requests per minute per API key
- 1000 requests per hour per IP address

## Endpoints

### Health Check

**GET** `/health`

Response:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 12345.67,
  "requests_processed": 1000,
  "services": {
    "redis": "connected",
    "database": "connected",
    "ml_models": "loaded"
  }
}
```

### Video Upload

**POST** `/upload/video`

**Content-Type**: `multipart/form-data`

Parameters:

- `file`: Video file (MP4, AVI, MOV, MKV)
- `source`: "upload" or "youtube"
- `metadata`: Optional JSON metadata

Response:

```json
{
  "id": "uuid-string",
  "filename": "video.mp4",
  "source": "upload",
  "upload_time": "2023-10-15T10:30:00Z",
  "status": "completed",
  "file_path": "/data/raw/uploads/video.mp4"
}
```

### Pose Estimation

**POST** `/inference/pose`

**Content-Type**: `application/json`

Request Body:

```json
{
  "video_id": "uuid-string",
  "model_name": "mediapipe",
  "confidence_threshold": 0.5,
  "include_3d": true,
  "smooth_output": true
}
```

Response:

```json
{
  "job_id": "uuid-string",
  "video_id": "uuid-string",
  "status": "queued",
  "total_frames": 0,
  "processed_frames": 0,
  "estimated_time": null
}
```

### Get Pose Job Status

**GET** `/inference/pose/{job_id}`

Response:

```json
{
  "job_id": "uuid-string",
  "video_id": "uuid-string",
  "status": "processing",
  "total_frames": 300,
  "processed_frames": 150,
  "estimated_time": 30.5,
  "result_path": "/data/processed/poses/job_uuid.json"
}
```

### Get Pose Results

**GET** `/inference/pose/{job_id}/result`

Response:

```json
{
  "video_id": "uuid-string",
  "frames": [
    {
      "frame_number": 1,
      "timestamp": 0.033,
      "keypoints": [
        {
          "id": 0,
          "name": "nose",
          "x": 0.512,
          "y": 0.234,
          "z": 0.0,
          "confidence": 0.923,
          "visible": true
        }
      ],
      "bounding_box": {
        "x": 0.1,
        "y": 0.2,
        "width": 0.3,
        "height": 0.4,
        "confidence": 0.95
      }
    }
  ],
  "model_name": "mediapipe",
  "processing_time": 45.2,
  "resolution": [1920, 1080]
}
```

### Simulation

**POST** `/simulation`

**Content-Type**: `application/json`

Request Body:

```json
{
  "pose_data_id": "uuid-string",
  "character_type": "humanoid",
  "output_format": "fbx",
  "include_constraints": true,
  "max_velocity": 180.0
}
```

Response:

```json
{
  "simulation_id": "uuid-string",
  "status": "processing",
  "output_formats": ["fbx", "bvh"],
  "file_paths": {
    "fbx": "/data/simulations/uuid/animation.fbx",
    "bvh": "/data/simulations/uuid/animation.bvh"
  },
  "processing_time": 120.5,
  "constraints_violations": 0
}
```

### Batch Processing

**POST** `/inference/pose/batch`

**Content-Type**: `application/json`

Request Body:

```json
{
  "items": [
    {
      "video_id": "uuid1",
      "model_name": "mediapipe"
    },
    {
      "video_id": "uuid2",
      "model_name": "openpose"
    }
  ]
}
```

Response:

```json
{
  "processed": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "video_id": "uuid1",
      "status": "completed",
      "result_path": "/data/processed/poses/uuid1.json"
    }
  ],
  "errors": []
}

## WebSocket Events

### Connection

```javascript
const ws = new WebSocket('wss://api.signverse.example.com/ws');
```

### Events

#### Job Progress

```json
{
  "event": "job_progress",
  "job_id": "uuid-string",
  "progress": 50,
  "processed_frames": 150,
  "total_frames": 300
}
```

#### Job Completed

```json
{
  "event": "job_completed",
  "job_id": "uuid-string",
  "result_path": "/data/processed/poses/uuid.json"
}
```

#### Error

```json
{
  "event": "error",
  "job_id": "uuid-string",
  "error": "Processing failed",
  "details": "Video format not supported"
}
```

## Error Responses

### HTTP Status Codes

- **200 OK**: Success
- **400 Bad Request**: Invalid input
- **401 Unauthorized**: Invalid API key
- **404 Not Found**: Resource not found
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error

### Error Format

```json
{
  "error": "invalid_input",
  "code": 400,
  "message": "Invalid video format",
  "details": {
    "field": "file",
    "expected": ["mp4", "avi", "mov", "mkv"],
    "received": "webm"
  }
}
```

## SDK Examples

### Python SDK

```python
from signverse_sdk import SignVerseClient

client = SignVerseClient(api_key="your_api_key")

# Upload video
upload_result = client.upload_video("video.mp4", source="upload")

# Estimate poses
job = client.estimate_poses(upload_result["id"])

# Get results
results = client.get_pose_results(job["job_id"])
```

### JavaScript SDK

```javascript
import { SignVerseClient } from 'signverse-sdk';

const client = new SignVerseClient({ apiKey: 'your_api_key' });

// Upload video
const upload = await client.uploadVideo(file, {
  source: 'upload',
  metadata: { description: 'Test video' }
});

// Monitor progress
const unsubscribe = client.subscribeToJob(upload.job_id, (event) => {
  console.log('Progress:', event.progress);
});
```

## Rate Limits

| Endpoint | Limit | Window |
| :--- | :--- | :--- |
| `/upload/video` | 10 | 1 minute |
| `/inference/pose` | 20 | 1 minute |
| `/simulation` | 5 | 1 minute |
| **All endpoints** | 100 | 1 minute |

## Versioning

API version is included in the path: `/api/v1/`

> [!NOTE]
> Backward compatibility is maintained within major versions.
