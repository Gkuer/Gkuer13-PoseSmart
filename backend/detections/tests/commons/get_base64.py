def get_blob_to_base64(typ):
    if typ == "face":
        with open("/Users/gkuer/Desktop/S06P22B201/backend/detections/tests/commons/face_base64.txt") as f:
            txt = f.read()

    elif typ == "noface":
        f = open("/Users/gkuer/Desktop/S06P22B201/backend/detections/tests/commons/noface_base64.txt")
        txt = f.read()
    return txt