from code_review import IWAE,preprocess
import pdb
import os
import numpy as np
from sklearn.decomposition import PCA
import random
import torch
import random

def set_seed(seed=42):
    """Set seed for reproducibility across PyTorch, NumPy, and Python's random module"""
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # Set seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for Python's random module
    random.seed(seed)
    
    # Make PyTorch operations deterministic (optional, may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Usage
set_seed(42)

# Alternative: Set individual components
torch.manual_seed(42)
torch.cuda.manual_seed(42)  # if using CUDA
np.random.seed(42)
random.seed(42)
def ReduceDimension(train_dataset,usePCA=True,PCAdim=576):
    if usePCA:
        train_dataset = np.array(train_dataset)
        train_dataset = train_dataset.reshape(train_dataset.shape[0],train_dataset.shape[-1])
        pca = PCA(n_components=PCAdim)
        pca_data = pca.fit_transform(train_dataset)
        pca_data = np.split(pca_data, pca_data.shape[0])
    return pca_data

def get_score(dataset,score,train_label,test_label,metric):
    train_score = {}
    test_score = {}
    for label in train_label:
        for key, value in score[label][metric].items():
            if key != 'problem':
                if key not in train_score:
                    train_score[key] = value
                else:
                    train_score[key] = train_score[key] + value
    for label in test_label:
            for key, value in score[label][metric].items():
                if key != 'problem':
                    # pdb.set_trace()
                    if key not in test_score:
                        test_score[key] = value
                    else:
                        train_score[key] = train_score[key] + value  
    train_dataset = [item for item in dataset[train_label[0]]]
    test_dataset = [item for item in dataset[test_label[0]]]
    return train_dataset, train_score, test_dataset, test_score          
    
def process(VAE_path,dataset_label,i,input_dim,dataset,score,metric,submetric):
    test_label = dataset_label[i]
    train_label = [i for i in dataset_label if i != test_label]
    VAE_store_path = f'{VAE_path}/{test_label}/trainVAE_dim{input_dim}.pkl'
    VAE_test_path = f'{VAE_path}/{test_label}/testVAE_dim{input_dim}.pkl'
    if not os.path.exists(f'{VAE_path}/{test_label}'):
        os.makedirs(f'{VAE_path}/{test_label}')
    train_dataset, train_score, test_dataset, test_score = get_score(dataset,score,train_label,[test_label],metric)
        # submetric preprocess
    if metric == 'Halstead_complexity' or metric == 'codebleu' :
        indices_set = set(range(len(train_dataset)))
        for key,value in train_score.items():
            invalid_indices = []
            for index, i in enumerate(value):
                if submetric not in i:
                    invalid_indices.append(index)
            indices_set = indices_set - set(invalid_indices) 
            indices_list = list(indices_set)
            train_score[key] = [train_score[key][i][submetric] for i in indices_list]
        train_dataset = [train_dataset[i] for i in indices_list]
        indices_set = set(range(len(test_dataset)))
        for key,value in test_score.items():
            invalid_indices = []
            for index, i in enumerate(value):
                if submetric not in i:
                    invalid_indices.append(index)
            indices_set = indices_set - set(invalid_indices) 
            indices_list = list(indices_set)
            test_score[key] = [test_score[key][i][submetric] for i in indices_list]
        test_dataset = [test_dataset[i] for i in indices_list]
        indices_set = set(range(len(train_dataset)))
        for key,value in train_score.items():
            positive_indices = [i for i, val in enumerate(value) if (isinstance(val,float) or isinstance(val,int)) and val > 0 ]
            indices_set = indices_set & set(positive_indices)
        indices_list = list(indices_set)
        train_dataset = [train_dataset[i] for i in indices_list]
        for key,value in train_score.items():
            train_score[key] = [train_score[key][i] for i in indices_list]
    
    return test_label,VAE_store_path,VAE_test_path,train_dataset,test_dataset,test_score,train_score
        
    
if __name__ == '__main__':
    dataset_label = ['bigcodebench','Evoeval']
    folder_path = '/root/autodl-tmp/processed_data'
    VAE_path = '/root/autodl-tmp/IWAE_model'
    Pca = False
    train_Mask = False
    test_Mask = False
    input_dim = 768
    test = 10
    random.seed(42)
    hidden_dim1 = input_dim // 2
    hidden_dim2 = hidden_dim1 // 2
    latent_dim = hidden_dim2 // 2
    metric = 'maintainability' #'Halstead_complexity', 'codebleu', 'maintainability', 'problem', 'security', 'circle_complexity', 'correctness']
    submetric = 'time'
    dataset,score = preprocess.preprocess(folder_path,metric=metric)
    for i in range(len(dataset_label)):
        test_label,VAE_store_path,VAE_test_path,train_dataset,test_dataset,test_score,train_score = process(VAE_path,dataset_label,i,input_dim,dataset,score,metric,submetric)
        
        if Pca:
            train_dataset = ReduceDimension(train_dataset,PCAdim=input_dim)
            test_dataset = ReduceDimension(test_dataset,PCAdim=input_dim)
        if train_Mask :
            indices = list(range(len(train_dataset)))
            random.shuffle(indices)
            selected_indices = indices[:train_Mask]
            train_dataset = [train_dataset[i] for i in selected_indices]
            train_score = {k: [v[i] for i in selected_indices] for k, v in train_score.items()}
        if test_Mask :
            indices = list(range(len(test_dataset)))
            random.shuffle(indices)
            selected_indices = indices[:test_Mask]
            test_dataset = [test_dataset[i] for i in selected_indices]
            test_score = {k: [v[i] for i in selected_indices] for k, v in test_score.items()}                      

        IWAE.train_VAE(dataloader=train_dataset,input_dim=input_dim,hidden_dim1=hidden_dim1,
                      hidden_dim2=hidden_dim2,latent_dim=latent_dim,VAE_path=VAE_store_path)
        IWAE.train_VAE(dataloader=test_dataset,input_dim=input_dim,hidden_dim1=hidden_dim1,
                      hidden_dim2=hidden_dim2,latent_dim=latent_dim,VAE_path=VAE_test_path)
        
        IWAE.test_VAE(f'{VAE_path}/{test_label}', input_dim,input_dim, train_dataset,train_score,test_score,max_min_normalization=True)
    pdb.set_trace()
    