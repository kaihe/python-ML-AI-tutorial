import cv2
import mediapipe as mp
import pandas as pd

# n11111s
# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize video capture.
cap = cv2.VideoCapture('data/clark剪辑/C4S104.mp4')

# Prepare to write the output video.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the video codec to mp4v
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))  # Change the file extension to .mp4

# Prepare to write the CSV file.
csv_file = open('keypoints.csv', 'w')
csv_file.write('frame,shoulder_x,shoulder_y,elbow_x,elbow_y,wrist_x,wrist_y,hip_x,hip_y,knee_x,knee_y,ankle_x,ankle_y,foot_x,foot_y,eye_x,eye_y,ear_x,ear_y\n')

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find the pose landmarks.
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # landmarks is a list of landmarks, where each landmark is a dictionary with x, y, and z fields.
        # z is the depth (distance from the camera) and is only available in 3D mode.
        
        # Extract the required keypoints output to a csv file.
        keypoints = {
            'shoulder': (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
            'elbow': (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y),
            'wrist': (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y),
            'hip': (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y),
            'knee': (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y),
            'ankle': (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y),
            'foot': (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y),
            'eye': (landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y),
            'ear': (landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y)
        }

        # Write keypoints to CSV.
        csv_file.write(f"{frame_idx},{keypoints['shoulder'][0]:.2f},{keypoints['shoulder'][1]:.2f},{keypoints['elbow'][0]:.2f},{keypoints['elbow'][1]:.2f},{keypoints['wrist'][0]:.2f},{keypoints['wrist'][1]:.2f},{keypoints['hip'][0]:.2f},{keypoints['hip'][1]:.2f},{keypoints['knee'][0]:.2f},{keypoints['knee'][1]:.2f},{keypoints['ankle'][0]:.2f},{keypoints['ankle'][1]:.2f},{keypoints['foot'][0]:.2f},{keypoints['foot'][1]:.2f},{keypoints['eye'][0]:.2f},{keypoints['eye'][1]:.2f},{keypoints['ear'][0]:.2f},{keypoints['ear'][1]:.2f}\n")

        # Draw keypoints on the frame.
        for key, point in keypoints.items():
            cv2.circle(frame, (int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])), 5, (0, 255, 0), -1)

    # Write the frame to the output video.
    out.write(frame)
    frame_idx += 1

# Release resources.
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
