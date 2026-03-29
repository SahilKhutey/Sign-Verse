# Motion Intelligence Configuration

# Thresholds for landmark detection
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Feature dimensions
# (210 LHand + 210 RHand + 3 Orientation + 1 Mouth) = 424
POS_DIM = 424
VEL_DIM = 424
TOTAL_DIM = POS_DIM + VEL_DIM

# MediaPipe IDs for features
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
UPPER_LIP = 13
LOWER_LIP = 14
