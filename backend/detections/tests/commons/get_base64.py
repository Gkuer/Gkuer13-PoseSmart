def get_blob_to_base64(typ):
    if typ == "face":
        with open("face_base64.txt") as f:
            txt = f.read()

    elif typ == "noface":
        with open("noface_base64.txt") as f:
            txt = f.read()

    return txt