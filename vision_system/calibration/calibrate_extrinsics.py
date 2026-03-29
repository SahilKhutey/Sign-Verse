import cv2
import numpy as np
import os
import json
import glob

def calibrate_extrinsics(image_dir_cam1, image_dir_cam2, mtx1, dist1, mtx2, dist2, board_size=(9, 6), square_size=25.0):
    """
    Calibrate extrinsic parameters (R, t) between two cameras.
    Expects synchronized images (named similarly) in both directories.
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints1 = []
    imgpoints2 = []

    images1 = glob.glob(os.path.join(image_dir_cam1, '*.jpg'))
    images2 = glob.glob(os.path.join(image_dir_cam2, '*.jpg'))
    
    # Sort and match images
    images1.sort()
    images2.sort()
    
    if not images1 or not images2:
        print("No images found in directories.")
        return None

    gray1 = gray2 = None
    for f1, f2 in zip(images1, images2):
        img1 = cv2.imread(f1)
        img2 = cv2.imread(f2)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        ret1, corners1 = cv2.findChessboardCorners(gray1, board_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, board_size, None)
        
        if ret1 and ret2:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)
            
            # Show stereo match
            # combined = np.hstack((img1, img2))
            # cv2.imshow('stereo', cv2.resize(combined, (0,0), fx=0.5, fy=0.5))
            # cv2.waitKey(100)

    cv2.destroyAllWindows()

    if not imgpoints1:
        return None

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, gray1.shape[::-1],
        criteria=criteria_stereo, flags=flags)

    # Save results
    calib_data = {
        "rotation_matrix": R.tolist(),
        "translation_vector": T.tolist(),
        "reprojection_error": ret
    }
    
    save_path = os.path.join(image_dir_cam1, 'extrinsics.json')
    with open(save_path, 'w') as f:
        json.dump(calib_data, f, indent=4)
        
    print(f"Extrinsic Calibration successful. Reprojection error: {ret}")
    return R, T

if __name__ == "__main__":
    # In a real use-case, load intrinsics first then call this.
    pass
    # calibrate_extrinsics("cam1", "cam2", mtx1, dist1, mtx2, dist2)
