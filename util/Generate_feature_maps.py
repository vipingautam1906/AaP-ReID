import numpy as np
import torch
from util.Extractor import FeatureExtractor
from torchvision import transforms
from IPython import embed
import models
from scipy.spatial.distance import cosine, euclidean
from  util.utils import *
from sklearn.preprocessing import normalize
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

def pool2d(tensor, type= 'max', batching=False):
    tensor = tensor.cuda() # size (1,2048, 8, 4)
    sz = tensor.size()
    if type == 'max':
        maxpool = torch.nn.MaxPool2d(kernel_size=(sz[2] // 8, sz[3]))
        maxpool = maxpool.cuda()
        x = maxpool(tensor) # size (1,2048,8) #first dim is batch...
    if type == 'mean':
        x = torch.nn.functional.mean_pool2d(tensor, kernel_size=(sz[2]//8, sz[3]) )

    res = [data.permute(2,1,0)[0] for data in x]
    return res

class Aligned_Reid_class:
    def __init__(self):
        self.res = []
        self.exact_list = ['7']
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        self.use_gpu = torch.cuda.is_available()
        if torch.cuda.is_available():
            self.map_location = lambda storage, loc: storage.cuda()
        else:
            self.map_location = 'cpu'

        self.model = models.init_model(name='resnet50', num_classes=751, loss={'softmax', 'metric'}, use_gpu=self.use_gpu,
                                       aligned=True)
        self.checkpoint = torch.load("/home/vipin2113106/Person_ReID/log/train_models-and-logs/market1501/run6/best_model1689933729.321595.pth.tar", map_location=self.map_location, encoding='latin1')
        self.model.load_state_dict(self.checkpoint['state_dict'], strict=False)  # loads module parameter
        self.myexactor = FeatureExtractor(self.model, self.exact_list)
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model.eval()
        if self.use_gpu:
            self.model = self.model.cuda()

    def torch_normalizer(self, tensor):
        normalized_tensor = tensor / tensor.norm(dim=1, keepdim=True)
        return normalized_tensor

    def inference(self, persons): 
        img = img_to_tensor(persons, self.img_transform).cuda()
        features,weight = self.myexactor(img)
        return features,weight
