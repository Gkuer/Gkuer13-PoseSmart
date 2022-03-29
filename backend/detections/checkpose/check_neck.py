import numpy as np

from . import pose_estimation

POSE_ESTIMATION = pose_estimation.build_model()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def get_keypoints(frame):
    heatmaps = POSE_ESTIMATION.predict(np.expand_dims(frame, 0))
    scores = sigmoid(heatmaps[1])
    num_bodyparts = heatmaps[1].shape[3]
    heatmap_positions = np.zeros((num_bodyparts, 2))
    offset_vectors = np.zeros((num_bodyparts, 2))
    max_scores = np.zeros(num_bodyparts)
    for i in range(num_bodyparts):
        max_scores[i] = np.max(heatmaps[1][0, :, :, i])
        heatmap_positions[i, :] = np.unravel_index(np.argmax(scores[0, :, :, i]), (scores.shape[1], scores.shape[2]))
        offset_vectors[i, 0] = heatmaps[0][0, int(heatmap_positions[i, 0]), int(heatmap_positions[i, 1]), i]
        offset_vectors[i, 1] = heatmaps[0][
            0, int(heatmap_positions[i, 0]), int(heatmap_positions[i, 1]), i + num_bodyparts]
    key_points = heatmap_positions * 8 + offset_vectors
    return key_points

def get_mean_face_positions(key_points):
    face_mean_x = np.mean([key_points[0, 1], key_points[14, 1], key_points[15, 1]])
    face_mean_y = np.mean([key_points[0, 0], key_points[14, 0], key_points[15, 0]])
    return face_mean_x, face_mean_y

def get_dist_nose_to_face_center(key_points):
    nose_y = key_points[0, 0]
    face_mean_y = np.mean([key_points[0, 0], key_points[14, 0], key_points[15, 0]])
    return np.abs(nose_y - face_mean_y)