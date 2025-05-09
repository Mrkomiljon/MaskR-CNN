import argparse
import cv2 
from PIL import Image

import torch 
import torchvision
import torchvision.transforms as transforms

from utils import draw_segmentation_map, get_outputs


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='path to the input data')
parser.add_argument('-t', '--threshold', default=0.965, type=float, help='score threshold for discarding detection')
args = vars(parser.parse_args())

# initialize the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# transform the image data to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])


# Read the image and apply instance segmentation
image_path = args['input']
image = Image.open(image_path).convert('RGB')

# keep copy of original image
orig_image = image.copy()

# transform the data
image = transform(image) 
# add a batch dimension
image = image.unsqueeze(0).to(device)

# get the prediction
masks, boxes, labels = get_outputs(image, model, args['threshold'])
# get the final result
result = draw_segmentation_map(orig_image, masks, boxes, labels)

# visualize the imageS
# cv2.imshow("Segmented image", result)
# cv2.waitKey(0)

# set the save path
Root ="your_input_path"
save_path = f'Root{args["input"].split("/")[-1].split(".")[0]}.jpg'
cv2.imwrite(save_path, result) 
print("Result saved")

######## TASK -> Inference on Video data and save the video (Trip to ZOO)
