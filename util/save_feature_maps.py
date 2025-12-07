from Generate_feature_maps import Aligned_Reid_class
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_maps(feature_maps, image, file_name):
    directory_path = '/market1501/Heatmaps/' # directory path for saving FMs
    file_name = directory_path + file_name
    num_rows = 16  # Number of rows in the grid
    num_cols = 16   # Number of columns in the grid
    scale = 224 / 7
    plt.figure(figsize=(16, 16))
    feature_maps = np.array(feature_maps)
    stacked_feature_maps = np.stack(feature_maps, axis=0)
    aggregated_feature_map = np.max(stacked_feature_maps, axis=0)

    normalized_feature_map = (aggregated_feature_map - aggregated_feature_map.min()) / (aggregated_feature_map.max() - aggregated_feature_map.min())
    heatmap = cv2.resize(normalized_feature_map, (image.shape[1], image.shape[0]))
    heatmap = plt.cm.jet(heatmap)
    superimposed_image = heatmap[:, :, :3] * 0.5 + image / 255.0 * 0.5
    plt.imshow(superimposed_image)
    plt.axis('off')
    plt.savefig(file_name)

directory = '/market1501/query/' # replace with query path..
file_list = os.listdir(directory)
for name in file_list:
    print('Currently processing....',name)
    path = directory + name
    original_image = cv2.imread(path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(original_image)

    model = Aligned_Reid_class()
    features, weight = model.inference(persons = img)
    feature_maps = features[0].detach().cpu()
    feature_maps = feature_maps.squeeze()
    file_name = os.path.basename(path)
    visualize_maps(feature_maps,original_image,name)



