import cv2
import os
import face_recognition
import pandas as pd
from datetime import datetime
import numpy as np

def capture_and_save_images(name, dataset_path="dataset", num_images=4):
    save_path = os.path.join(dataset_path, name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 1
    while count <= num_images:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{save_path}/{count}.jpg", frame)
        count += 1
        cv2.waitKey(1000)
    cap.release()

def mark_attendance(attendance_file="attendance.csv", dataset_path="dataset"):
    known_encodings = []
    known_names = []

    # Load known images and encodings
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)

    cap = cv2.VideoCapture(0)
    marked_names = []
    unknown_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        
        # Only run face encodings if faces are found
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        else:
            face_encodings = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            if name != "Unknown" and name not in marked_names:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if not os.path.exists(attendance_file):
                    with open(attendance_file, 'w') as f:
                        f.write('Name,DateTime\n')
                df = pd.read_csv(attendance_file)
                if not ((df['Name'] == name) & (df['DateTime'].str.startswith(datetime.now().strftime('%Y-%m-%d')))).any():
                    with open(attendance_file, 'a') as f:
                        f.write(f"{name},{now}\n")
                    marked_names.append(name)
            elif name == "Unknown":
                unknown_detected = True

        cv2.imshow('Marking Attendance (press q to quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return marked_names, unknown_detected

