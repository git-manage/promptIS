import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.distributions as dist
import tqdm
import pdb
from sklearn.preprocessing import StandardScaler
import os
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        
    def forward(self, x):
        h = F.relu(self.fc(x))
        mu, logvar = self.mu(h), self.logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128)
        self.out = nn.Linear(128, input_dim)
    
    def forward(self, z):
        h = F.relu(self.fc(z))
        return self.out(h)

class IWAE(nn.Module):
    def __init__(self, input_dim, latent_dim, K=5):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.latent_dim = latent_dim
        self.K = K
        self.batch_size = 16
        self.epochs = 15
        self.lr = 1e-3
        self.warmup_epoch = 5
        self.device = 'cuda:0'
        
    def forward(self, x):
        B = x.size(0)
        mu, logvar = self.encoder(x)

        # 扩展成 [B, K, latent_dim]
        mu = mu.unsqueeze(1).expand(B, self.K, self.latent_dim)
        logvar = logvar.unsqueeze(1).expand(B, self.K, self.latent_dim)
        
        # 采样 z_k
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # [B, K, latent_dim]
        
        # 解码
        x_recon = self.decoder(z.view(B * self.K, self.latent_dim))
        x_recon = x_recon.view(B, self.K, -1)

        # log p(x|z)
        recon_log_prob = -F.mse_loss(x_recon, x.unsqueeze(1).expand_as(x_recon), reduction='none').sum(-1)

        # log p(z)
        log_pz = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * torch.pi))).sum(-1)

        # log q(z|x)
        log_qz = -0.5 * ((z - mu) ** 2 / torch.exp(logvar) + logvar + torch.log(torch.tensor(2 * torch.pi))).sum(-1)

        # importance weights
        log_w = log_pz + recon_log_prob - log_qz
        w = F.softmax(log_w, dim=1)  # 可用于重参数或样本加权

        # IWAE Loss: -log(1/K ∑ w_k)
        log_iwae = torch.logsumexp(log_w, dim=1) - torch.log(torch.tensor(self.K, dtype=torch.float))
        loss = -log_iwae.mean()

        return loss


def batch(arrays_list,group_size):
    # 计算可以形成的矩阵数量
    num_matrices = len(arrays_list) // group_size

    # 创建新的列表来保存结果矩阵
    result_matrices = []

    for i in range(num_matrices):
    # 获取当前组的起始和结束索引
        start_index = i * group_size
        end_index = start_index + group_size
    
    # 从原列表中提取当前组的所有数组，并堆叠成一个矩阵
        matrix = np.stack(arrays_list[start_index:end_index])
    
    # 将生成的矩阵添加到结果列表中
        result_matrices.append(matrix)
    return result_matrices
# --------------------------
# 训练过程
# --------------------------
def train_VAE(dataloader, input_dim, hidden_dim1, hidden_dim2, latent_dim, VAE_path, useIWAE=True, iw_samples=20):
    if os.path.exists(VAE_path):
        print(f"Model file {VAE_path} exists, skipping training.")
        return

    model = IWAE(input_dim, latent_dim, K=iw_samples)
    scaler = StandardScaler()
    model.to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lowest_loss = float('inf')

    for epoch in tqdm.tqdm(range(model.epochs)):
        model.train()
        total_loss = 0

        for batch_x in dataloader:
            batch_x = scaler.fit_transform(batch_x)
            x = torch.from_numpy(batch_x).to(model.device).float()
            optimizer.zero_grad()
            # forward 返回值现在是 (recon_mu, recon_logvar, mu_z, logvar_z, loss)
            loss = model(x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{model.epochs}, Loss: {total_loss:.4f}")
        if total_loss < lowest_loss:
            lowest_loss = total_loss
            torch.save(model, VAE_path)
            print(f"Saved model with loss {lowest_loss:.4f}")

    return

def test_VAE(VAE_path, input_dim, test_dim, train_dataset, train_score, test_score, max_min_normalization=False, full_score=False):
    train_dataset = torch.tensor(train_dataset).reshape(-1, len(train_dataset[0][0])).to('cuda:0')
    IWAE_test = torch.load(f'{VAE_path}/testVAE_dim{input_dim}.pkl')
    IWAE_train = torch.load(f'{VAE_path}/trainVAE_dim{test_dim}.pkl')
    test = 100
    result_dict = {}
    Weight = estimate_importance_weight(test_dim, IWAE_test, IWAE_train, train_dataset).to('cpu')
    for i in range(test):
        for key, value in train_score.items():
            Trainscore = torch.tensor(train_score[key])
            Testscore = torch.tensor(test_score[key])
            if max_min_normalization:
                combined = torch.cat((Trainscore, Testscore), dim=0)
                min_val = combined.min()
                max_val = combined.max()
                normalized_combined = (combined - min_val) / (max_val - min_val + 1e-8)
                Trainscore = normalized_combined[:len(Trainscore)]
                Testscore = normalized_combined[len(Trainscore):]
            if full_score:
                Trainscore = Trainscore / full_score
                Testscore = Testscore / full_score
            pdb.set_trace()
            result = torch.dot(Weight, Trainscore) / sum(Weight)
            goal = sum(Testscore) / len(Testscore)
            if key not in result_dict:
                result_dict[key] = []
            else:
                result_dict[key].append(result - goal)
    
    for key,value in result_dict.items():
        print(f"{key}:{sum(value)/len(value)}")
        
def estimate_log_likelihood(model, x, iw_samples=10):
    B = x.size(0)
    mu, logvar = model.encoder(x)
    mu = mu.unsqueeze(1).expand(B, iw_samples, model.latent_dim)
    logvar = logvar.unsqueeze(1).expand(B, iw_samples, model.latent_dim)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std

    x_recon = model.decoder(z.view(B * iw_samples, model.latent_dim))
    x_recon = x_recon.view(B, iw_samples, -1)

    recon_log_prob = -F.mse_loss(x_recon, x.unsqueeze(1).expand_as(x_recon), reduction='none').sum(-1)
    log_pz = -0.5 * (z.pow(2) + torch.log(torch.tensor(2 * torch.pi))).sum(-1)
    log_qz = -0.5 * ((z - mu) ** 2 / torch.exp(logvar) + logvar + torch.log(torch.tensor(2 * torch.pi))).sum(-1)

    log_weights = recon_log_prob + log_pz - log_qz
    log_likelihood = torch.logsumexp(log_weights, dim=1) - torch.log(torch.tensor(iw_samples, dtype=torch.float32))
    return log_likelihood

def estimate_importance_weight(test_dim, model_p, model_q, sample):
    log_px = estimate_log_likelihood(model_p, sample)
    sample = sample.to('cuda:0')
    log_qx = estimate_log_likelihood(model_q, sample)
    log_weights = log_px - log_qx
    weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
    clip = torch.quantile(weights, 0.98)
    weights = torch.minimum(weights, clip)
    return weights