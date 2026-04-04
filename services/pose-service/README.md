# Sign-Verse Pose Extraction Service (Layer 2)

This service is a robust, asynchronous Pose Extraction component built with **FastAPI** and **MediaPipe**. It handles video-based pose detection by queuing jobs and tracking their status via **Redis**.

## Core Features
- **Asynchronous Execution**: Submits jobs and processes them in the background.
- **Persistent Status**: Tracks `pending`, `processing`, `completed`, and `failed` states.
- **Scalability**: Designed for horizontal scaling with multiple workers.

## Manual Local Run (Development)

### 1. Build the Docker Image
```bash
docker build -t pose-service .
```

### 2. Run the Service
Ensure you have a Redis instance running (e.g., via Docker Compose) or provide the `REDIS_HOST`.
```bash
docker run -p 8000:8000 -e REDIS_HOST=host.docker.internal pose-service
```

### 3. Submit a Job
Use `curl` or Postman to submit a video path for extraction.
```bash
curl -X POST "http://localhost:8000/extract-pose" \
     -H "Content-Type: application/json" \
     -d '{"video_url": "/path/to/your/video.mp4"}'
```

The service will return a `job_id`, which you can use to check the status:
```bash
curl "http://localhost:8000/job-status/{job_id}"
```

---
Part of the **Sign-Verse** Ecosystem.
