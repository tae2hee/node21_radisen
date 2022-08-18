import skimage.transform as skTrans
import numpy as np
import cv2


def possibility_map(outputs, patch_counts):
    scores = np.zeros((patch_counts * patch_counts))
    for idx, output in outputs:
        score = output
        scores[idx] = score.to('cpu')
    scores = np.reshape(scores, (patch_counts, patch_counts))

    return scores



def postprocessing(outputs, outputs2):
    image_size = 1024
    per_thres = 98
    erode = 7
    dilation = 7
    first_model_thres = 0.8
    count = 64

    _heatmap = possibility_map(outputs, count)
    _heatmap2 = possibility_map(outputs2, count)

    _heatmap = skTrans.resize(_heatmap, (image_size, image_size),preserve_range=True,
                              anti_aliasing=True)
    _heatmap2 = skTrans.resize(_heatmap2, (image_size, image_size), preserve_range=True,
                               anti_aliasing=True)

    _map = np.where(_heatmap > np.percentile(_heatmap, per_thres), 1, 0)
    _map = _map.astype('uint8')
    _map = cv2.erode(_map, np.ones((erode, erode), np.uint8), iterations=1)
    _map = cv2.dilate(_map, np.ones((dilation, dilation), np.uint8), iterations=1)

    components, output, stats, centroids = cv2.connectedComponentsWithStats(_map, connectivity=4)

    predict_outputs = {"scores": [], "boxes": []}

    ### thresholding with first model
    for k in range(1, components):
        sub_map = np.where(output == k, 1, 0)
        (x, y, w, h, area) = stats[k]
        score = np.max(sub_map * _heatmap)
        if score > first_model_thres:
            final_score = np.max(sub_map * _heatmap2)
            predict_outputs["scores"].append(final_score)
            predict_outputs["boxes"].append(np.array([x, y, (x+w), (y+h)]))

    return predict_outputs
