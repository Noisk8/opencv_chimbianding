import cv2
import face_recognition

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find all the faces and their landmarks in the grayscale frame
    face_locations = face_recognition.face_locations(gray)
    face_landmarks_list = face_recognition.face_landmarks(gray, face_locations)

    # Iterate over each face found
    for (top, right, bottom, left), face_landmarks in zip(face_locations, face_landmarks_list):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw circles around the eyes and mouth
        eye_locations = face_landmarks['left_eye'] + face_landmarks['right_eye']
        for eye in eye_locations:
            cv2.circle(frame, eye, 2, (0, 255, 0), -1)
        mouth_locations = face_landmarks['top_lip'] + face_landmarks['bottom_lip']
        for mouth in mouth_locations:
            cv2.circle(frame, mouth, 2, (0, 255, 0), -1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
