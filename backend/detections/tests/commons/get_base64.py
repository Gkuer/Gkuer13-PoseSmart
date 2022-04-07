def get_base64_encoded_blob(face_type):
    if face_type == "face":
        with open("/Users/gkuer/Desktop/S06P22B201/backend/detections/tests/commons/face_base64.txt") as f:
            base64 = f.read()

    elif face_type == "noface":
        with open("/Users/gkuer/Desktop/S06P22B201/backend/detections/tests/commons/no_face_base64.txt") as f:
            base64 = f.read()

    return base64