def list_from_str(face_x_str, face_y_str, nose_to_center_str):
    if face_y_str and nose_to_center_str and face_x_str:
        face_x = face_x_str.split(",")
        face_y = face_y_str.split(",")
        nose_to_center = nose_to_center_str.split(",")

    else:
        # 기준 값
        face_x = []  # 얼굴 폭
        face_y = []  # 얼굴 y축 평균 좌표값
        nose_to_center = []  # 코와 얼굴 중간 간의 거리

    return face_x, face_y, nose_to_center