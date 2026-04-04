# Technical Spec: Human-Object Interaction (HOI) Platform

The SignVerse HOI platform enables metric-absolute behavioral analytics by fusing monocular 3D perception with multi-class object tracking.

## 🛰️ Perception Stacks

### 🔍 Multi-Class Detection
The foundation layer utilizes YOLOv8/v9 for high-speed identification of persons and COCO-80+ object categories. We've introduced **Entity Groupings** to optimize detection for specific industry domains:
- **Fitness**: dumbbells, barbells, benches, etc.
- **Kitchen**: knives, spoons, bowls, bottles, etc.
- **Office**: laptops, mice, keyboards, books, etc.

### 🔄 Multi-Object Tracking (MOT)
Unlike traditional person trackers, our MOT engine manages independent ID namespaces for **Humans (P-series)** and **Objects (O-series)**. It uses Hungarian ROI association to maintain stable identities during occlusions.

### 📐 Monocular 3D & Depth
To solve the 2D spatial ambiguity problem, we integrate **MiDaS/DPT** for monocular depth estimation.
- **Intrinsic Mapping**: Backprojects 2D image coordinates into 3D camera space.
- **Model Fidelity**: Supports `fast` (Small) and `accurate` (Large) depth models as per workload requirements.

## 🧠 Behavioral Diagnostics

The [Interaction Engine](file:///c:/Users/User/Documents/Sign-Verse/core/processing/interaction_engine.py) provides the final inference layer for behavioral diagnostics:

| State | Primary Trigger | Contextual Cue |
| :--- | :--- | :--- |
| **Holding** | Hand-Object Contact | Motion Correlation > 0.6 |
| **Touching** | Hand-Object Proximity | - |
| **Using** | Intention Mapping | Body-Object Orientation |
| **Looking At** | Gaze Alignment | Head-Pose Yaw/Pitch |

## 📊 Temporal Consistency
All behavioral states are passed through a **15-frame temporal window**, filtering out transient noise and intermittent occlusions to ensure a stable behavioral audit trail.
