import torchvision

class ImageFolderNoLabels(torchvision.datasets.ImageFolder):
    def __getitem__(self, *args, **kwargs):
        x, _ = super().__getitem__(*args, **kwargs)
        return x
