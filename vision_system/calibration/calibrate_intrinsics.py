import cv2
import numpy as np
import glob
import os
import json

def calibrate_intrinsics(image_dir, board_size=(9, 6), square_size=25.0):
    """
    Calibrate a single camera's intrinsic parameters (K, dist).
    board_size: (columns, rows) of inner corners.
    square_size: size of a square in real-world units (e.g. mm).
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    if not images:
        print(f"No images found in {image_dir}")
        return None

    gray = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            objpoints.append(objp)
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Optional: Draw and display
            cv2.drawChessboardCorners(img, board_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    if not imgpoints:
        return None

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Save results
    calib_data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeff": dist.tolist(),
        "reprojection_error": ret
    }
    
    save_path = os.path.join(image_dir, 'intrinsics.json')
    with open(save_path, 'w') as f:
        json.dump(calib_data, f, indent=4)
        
    print(f"Calibration successful. Reprojection error: {ret}")
    print(f"Intrinsics saved to {save_path}")
    return mtx, dist

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing calibration images")
    args = parser.parse_args()
    calibrate_intrinsics(args.dir)
