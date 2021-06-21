import torchvision.models as models
import torch.nn as nn

def liveness_model():

    lv_model = models.squeezenet1_1(pretrained=True)
    lv_model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Conv2d(512, 2, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(13)
    )
    lv_model.forward = lambda x: lv_model.classifier(lv_model.features(x)).view(x.size(0), 2)
    return lv_model