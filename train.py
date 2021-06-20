# USAGE
# python train.py --data Dataset --out_weights output_model.pth --init_weights init_model.pth -n 30


import numpy as np
import argparse
import cv2
import os
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from dataset import CustomDataset
from matrix_diff import F_feature
from torchvision import transforms



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", type=str, required=True,
	help="path to dataset")
ap.add_argument("-i", "--init_weights", type=str, required=False,
	help="path to input weights")
ap.add_argument("-o", "--out_weights", type=str, required=True,
	help="path to output weights")
ap.add_argument("-n", "--epochs", type=str, required=True,
	help="number of epochs for training")
args = vars(ap.parse_args())



lv_model = models.squeezenet1_1(pretrained=True)
lv_model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(512, 2, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.AvgPool2d(13)
)
lv_model.forward = lambda x: lv_model.classifier(lv_model.features(x)).view(x.size(0), 2)


protoPath = os.path.sep.join(["FaceDetector", "deploy.prototxt"])
modelPath = os.path.sep.join(["FaceDetector",
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


if args['init_weights'] is not None:
    lv_model.load_state_dict(torch.load(args["init_weights"], map_location=torch.device('cpu'))['state_dict'])

if not os.path.exists("maps"):
	os.makedirs("maps")


for cls in ["Fake", "Real"]:
  for path in os.listdir(os.path.join(args["data"],cls)):
    print(cls, path)
    if os.path.exists(os.path.join("maps", cls + "_" +path+".npy")):
      continue
    diff = np.array([])
    shape = 0
    n = len(os.listdir(os.path.join(args["data"], cls, path)))
    for file in range(0, n):
        if os.path.exists(os.path.join(args["data"], cls, path, str(file) + '.png')):
          if file != 0:
            prev_img = curr_img
          curr_img = cv2.imread(os.path.join(args["data"], cls, path, str(file) + '.png'))
          blob = cv2.dnn.blobFromImage(cv2.resize(curr_img, (300, 300)), 1.0,
                                       (300, 300), (104.0, 177.0, 123.0))
          (h, w) = curr_img.shape[:2]

          net.setInput(blob)
          detections = net.forward()
          i = np.argmax(detections[0, 0, :, 2])
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")
          if file != 0:
            map = F_feature(prev_img, curr_img).astype('float32')
            diff = np.append(diff, cv2.resize(map[startY:endY, startX:endX], (256, 256)))
    shape = (-1, 256, 256, 3)
    diff = diff.reshape(shape)
    # new_data = np.array([])
    # for img in diff:
    #   new_img = cv2.resize(img, (256, 256))
    #   new_data = np.append(new_data, new_img)
    # new_data = new_data.reshape(-1, 256, 256, 3)
    np.save(os.path.join('maps', cls + "_" +path+".npy"), diff)


def create_samples(cls):
    data = np.array([])
    for path in os.listdir('maps'):
        if path.split("_")[0] == cls:
            data = np.append(data, np.load('maps/' + path))
    return data.reshape(-1, 256, 256, 3)


true_data = create_samples('Real')
false_data = create_samples('Fake')


X = np.append(true_data, false_data).reshape(-1, 256, 256, 3)
Y = [1] * len(true_data) + [0] * len(false_data)


IMG_SIZE=256

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

X = []
y = []


transform_for_train_and_val = transforms.Compose(
    [
        transforms.ToTensor(),
    ])


batch_size=1

trainset = CustomDataset('./data/train/train', 'dd', transform_for_train_and_val, X_train, y_train)
valset = CustomDataset('./data/train/train', 'dd', transform_for_train_and_val, X_test, y_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=8, pin_memory=True)


params_to_update = []
for name,param in lv_model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = optim.SGD(params_to_update, lr=0.0001, momentum=0.91)


def run_epoch(epoch, is_train):
    """
    Training and evaluaton loop over samples
    Args:
        train_mode (bool): True for train mode
    """
    if is_train:
        lv_model.train()
        loader = trainloader
        print("Training epoch: ", epoch + 1, "/", num_epochs)
    else:
        lv_model.eval()
        loader = valloader
        print('Validation')

    running_loss = 0.0
    correct = 0.0
    total = 0.0

    for i, data in tqdm.tqdm(enumerate(loader)):
        images, labels = data

        # images, labels = images.cuda(), labels.cuda()

        outputs = lv_model(images)
        loss = criterion(outputs, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        total += images.data.size(0)

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.data).sum()

    print('Loss: {:.3f}, accuracy: {:.3f}'.format(running_loss / (i + 1), correct / total * 100.0))


criterion = nn.CrossEntropyLoss()

num_epochs = int(args["epochs"])

for epoch in range(num_epochs):
  run_epoch(epoch, is_train=True)

  with torch.no_grad():
      run_epoch(epoch, is_train=False)

  state = {
          'epoch': epoch + 1,
          'state_dict': lv_model.state_dict(),
          'optimizer' : optimizer.state_dict()
          }

  torch.save(state, args['out_weights'])
