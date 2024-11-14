import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize video capture.
cap = cv2.VideoCapture(r'data\Louis剪辑\L4S30.mp4')

# Prepare to write the output video.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

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

        # Extract the required keypoints.
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
        csv_file.write(f"{frame_idx},{keypoints['shoulder'][0]},{keypoints['shoulder'][1]},{keypoints['elbow'][0]},{keypoints['elbow'][1]},{keypoints['wrist'][0]},{keypoints['wrist'][1]},{keypoints['hip'][0]},{keypoints['hip'][1]},{keypoints['knee'][0]},{keypoints['knee'][1]},{keypoints['ankle'][0]},{keypoints['ankle'][1]},{keypoints['foot'][0]},{keypoints['foot'][1]},{keypoints['eye'][0]},{keypoints['eye'][1]},{keypoints['ear'][0]},{keypoints['ear'][1]}\n")

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
