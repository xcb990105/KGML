import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import copy

class ImageRegressionDataset(Dataset):
    def __init__(self, data, root_dir, metal_features_list,transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.metal_features_list = metal_features_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.root_dir}/{self.data.iloc[idx]['compound']}.png"
        label = self.data.iloc[idx]['delta_G']

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, self.metal_features_list[idx], torch.tensor(label, dtype=torch.float32), self.data.iloc[idx]['smiles']

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),         
    transforms.ToTensor(),   
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

def prepare_dataloader(path, image_path, shuffle, bs):
    df = pd.read_csv(path) if isinstance(path, str) else copy.deepcopy(path)
    smi_list = df['smiles'].tolist()
    ion_radius = df['Ionic_Radius'].tolist()
    atom_num = df['Atomic_Number'].tolist()
    electronegativity = df['electronegativity'].tolist()
    mw = df['MW'].tolist()
    coord_num = df['coord_num'].tolist()
    FirstIE = df['FirstIE'].tolist()
    SecondIE = df['SecondIE'].tolist()
    ThirdIE = df['ThirdIE'].tolist()

    metal_features_list = []
    for i in range(len(smi_list)):
        metal_features_list.append(torch.tensor([ion_radius[i], atom_num[i], electronegativity[i], mw[i],
                                                 coord_num[i], FirstIE[i], SecondIE[i], ThirdIE[i]], dtype=torch.float))
    dataset = ImageRegressionDataset(df, image_path, metal_features_list, transform=preprocess)
    assert isinstance(bs, int)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=0)
    return dataloader