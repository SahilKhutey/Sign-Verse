-- Sign-Verse MetaData Schema Initialization

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    camera_id VARCHAR(50),
    frame_rate INTEGER,
    resolution_x INTEGER,
    resolution_y INTEGER,
    status VARCHAR(20) DEFAULT 'ACTIVE'
);

CREATE TABLE IF NOT EXISTS pose_data (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    frame_number BIGINT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    keypoints_3d JSONB NOT NULL, -- Normalized 3D coordinates
    confidence JSONB NOT NULL,     -- Conf scores for each point
    metadata JSONB               -- Device specific info
);

CREATE INDEX idx_pose_session ON pose_data(session_id);
CREATE INDEX idx_pose_timestamp ON pose_data(timestamp);

-- Hardware/Telemetric Status
CREATE TABLE IF NOT EXISTS telemetry_logs (
    id BIGSERIAL PRIMARY KEY,
    component VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    metrics JSONB, -- CPU/GPU load, latency
    logged_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
