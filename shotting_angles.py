import cv2
import numpy as np
import mediapipe as mp

def calculate_angle(p1, p2, p3):
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Calculate the vectors
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate the angle between the vectors
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)

    # Convert the angle from radians to degrees
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

def mark_keypoints_and_render(video_path, output_path):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find the pose
        results = pose.process(rgb_frame)

        # Draw the specific keypoints and connections on the frame
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Draw circles at the keypoints
            cv2.circle(frame, (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height)), 5, (0, 255, 0), -1)

            # Draw lines connecting the keypoints
            cv2.line(frame, (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height)),
                 (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height)), (0, 255, 0), 2)
            cv2.line(frame, (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height)),
                 (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height)), (0, 255, 0), 2)

            # Calculate the angle between the shoulder, elbow, and wrist
            shoulder = (right_shoulder.x * frame_width, right_shoulder.y * frame_height)
            elbow = (right_elbow.x * frame_width, right_elbow.y * frame_height)
            wrist = (right_wrist.x * frame_width, right_wrist.y * frame_height)
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Print the angle on the frame with red text
            cv2.putText(frame, f"Angle: {angle:.2f} degrees", 
                    (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    pose.close()

# Example usage
if __name__ == "__main__":
    mark_keypoints_and_render(r"data\clark剪辑\C10S93.mp4", r"out.mp4")

