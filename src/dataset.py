from PIL import Image
import torch.utils.data as data


class CustomDataset(data.Dataset):

    def __init__(self, root, flist, transform, X, y):
        self.root = root
        self.imlist = flist
        self.transform = transform
        self.X = X
        self.y = y

    def __getitem__(self, index):
        img, target = Image.fromarray(self.X[index], 'RGB'), self.y[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.y)