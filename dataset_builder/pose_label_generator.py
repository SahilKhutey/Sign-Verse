"""
Dataset Builder — Pose Label Generator

Extracts MediaPipe keypoints from image frames using standalone
Face, Hands, and Pose models. Saves as .npy files with metadata
for training 1629-dim foundation models.
"""

import mediapipe as mp

from motion_capture.smpl_fitter import SMPLFitter

def generate_pose_labels(frames_dir, output_dir, parametric=False):
    """
    Process all frames in a directory and generate pose keypoints.

    Args:
        frames_dir:  directory of frame images (.jpg/.png)
        output_dir:  directory to save .npy keypoint files
        parametric: if True, saves 162-dim SMPL-X pose parameters. 
                    if False, saves 1629-dim raw XYZ landmarks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize standalone MediaPipe solutions via solutions submodule
    try:
        mp_solutions = mp.solutions
    except AttributeError:
        # Fallback for unconventional installations
        import mediapipe.python.solutions as mp_solutions

    p_detect = mp_solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    h_detect = mp_solutions.hands.Hands(static_image_mode=True, max_num_hands=2)
    f_detect = mp_solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    fitter = None
    if parametric:
        fitter = SMPLFitter(model_type="smplx")

    metadata = []
    processed = 0

    for fname in sorted(os.listdir(frames_dir)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pose_res = p_detect.process(rgb)
        face_res = f_detect.process(rgb)
        hands_res = h_detect.process(rgb)

        if parametric:
            # Fit SMPL-X parameters from landmarks (Simplified fit call)
            # In production, we'd also pass OpenPose/VIBE if available for 'overall optimization'
            refined = fitter.fit_from_multi_feed(
                mp_landmarks={"body": pose_res, "face": face_res, "hands": hands_res}
            )
            arr = refined["pose"] # 162-dim parametric sequence
        else:
            # Raw 1629-dim XYZ extraction
            kp = []
            # ... (Body 99, Face 1404, Hands 126 loop logic) ...
            # 1. Body (99)
            if pose_res.pose_landmarks:
                for lm in pose_res.pose_landmarks.landmark: kp.extend([lm.x, lm.y, lm.z])
            else: kp.extend([0.0] * 99)
            # 2. Face (1404)
            if face_res.multi_face_landmarks:
                for j, lm in enumerate(face_res.multi_face_landmarks[0].landmark):
                    if j >= 468: break
                    kp.extend([lm.x, lm.y, lm.z])
            else: kp.extend([0.0] * 1404)
            # 3. Hands (126) -> mapping left/right
            left_h = [0.0]*63; right_h = [0.0]*63
            if hands_res.multi_hand_landmarks:
                for i, h_lm in enumerate(hands_res.multi_hand_landmarks):
                    label = hands_res.multi_handedness[i].classification[0].label
                    vals = []
                    for lm in h_lm.landmark: vals.extend([lm.x, lm.y, lm.z])
                    if label == 'Left': left_h = vals
                    else: right_h = vals
            kp.extend(left_h); kp.extend(right_h)
            arr = np.array(kp, dtype=np.float32)

        npy_name = fname.replace('.jpg', '.npy').replace('.png', '.npy')
        np.save(os.path.join(output_dir, npy_name), arr)
        metadata.append({"frame": fname, "keypoints_file": npy_name, "dim": len(arr), "type": "parametric" if parametric else "raw"})
        processed += 1

    mp_pose.close(); mp_hands.close(); mp_face.close()
    with open(os.path.join(output_dir, "metadata.json"), "w") as f: json.dump(metadata, f, indent=2)

    print(f"Processed {processed} frames [{'Parametric' if parametric else 'Raw'}] → {output_dir}")
    return processed
