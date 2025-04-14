import numpy as np
import cv2

def calibrate_camera(objectPoints, imagePoints, image_size):
    # Initial camera matrix guess
    focal_length = image_size[0]  # Assume focal length in pixels (adjust if necessary)
    camera_matrix = np.array([
        [focal_length, 0, image_size[0] / 2],
        [0, focal_length, image_size[1] / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    # Calibration flags
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_PRINCIPAL_POINT | \
            cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | \
            cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6

    # Run calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints, imagePoints, image_size, camera_matrix, np.zeros((5, 1)), flags=flags
    )

    return camera_matrix, dist_coeffs, rvecs, tvecs

def calibrate_camera_from_bounds(vid_table_bounds, vid_base_bounds, video_frame_size):
    # Calibration points (3D points in the world coordinate system)
    table_bounds = np.array([
        [+450, +250, +250],
        [-450, +250, +250],
        [+450, +250, -250],
        [-450, +250, -250],
        [   0, +250, +250],
        [   0, +250, -250],
    ])
    base_bounds = np.array([
        [+450, 0, +250],
        [-450, 0, +250],
        [+450, 0, -250],
        [-450, 0, -250],
    ])
    # Number of frames
    num_frames = len(vid_table_bounds)

    # Stack table_bounds and base_bounds for each frame
    table_bounds = np.tile(table_bounds[np.newaxis, :, :], (num_frames, 1, 1))
    base_bounds = np.tile(base_bounds[np.newaxis, :, :], (num_frames, 1, 1))

    # Prepare 2D and 3D points
    points_2d = np.concatenate((vid_table_bounds, vid_base_bounds), axis=1)
    points_3d = np.concatenate((table_bounds, base_bounds), axis=1) # * 0.003048  # Convert to meters

    # Prepare objectPoints and imagePoints for calibrateCamera
    objectPoints = [points_3d[i].astype(np.float32) for i in range(len(points_3d))]
    imagePoints = [points_2d[i].astype(np.float32) for i in range(len(points_2d))]

    # Reshape points for calibrateCamera
    objectPoints = [op.reshape(-1,1,3) for op in objectPoints]
    imagePoints = [ip.reshape(-1,1,2) for ip in imagePoints]

    # Calibrate camera
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(objectPoints, imagePoints, video_frame_size)

    # Compute extrinsic matrices
    rmats = []
    for rvec in rvecs:
        R, _ = cv2.Rodrigues(rvec)
        rmats.append(R)

    return camera_matrix, rmats, tvecs
