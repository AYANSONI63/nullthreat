import torch 
from torch.utils.data import Dataset 


class URLDataset(Dataset):

    def __init__(self, X, y):
        
        self.X = X
        self.y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):
        
        url = self.X[index]
        label = self.y[index]

        url = torch.tensor(url, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)


        return url, label 

    
    
