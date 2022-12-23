import cv2 
import numpy as np
import random
import torch

from coco_names import COCO_CLASSES as coco_names


# Give different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


# function to get output from model inference
def get_outputs(image, model, threshold):

    with torch.no_grad():
        outputs = model(image)

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # indexing for threshold
    threshold_indices = [scores.index(i) for i in scores if i > threshold]
    threshold_count = len(threshold_indices)
    
    # get masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks below threshold
    masks = masks[:threshold_count]

    # get bbox (x1, y1), (x2, y2)
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs[0]['boxes'].detach().cpu()]
    boxes = boxes[:threshold_count]

    # get the class labels
    labels = [coco_names[i] for i in outputs[0]['labels']]

    return masks, boxes, labels


# a function to draw segmentation map and bounding boxes
def draw_segmentation_map(image, masks, boxes, labels):

    alpha = 1
    beta = 0.6 # transparency for mask
    gamma = 0 # scalar added to each sum

    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)  
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)

        # apply a random color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        # convert PIL image to np format
        image = np.array(image)
        # convert RGB to BGR OpenCV format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

        # draw bounding box
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, thickness=2)
        # put the label text above the object
        cv2.putText(image, labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=2, lineType=cv2.LINE_AA)

    return image