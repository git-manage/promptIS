import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.distributions as dist
import tqdm
import pdb
from sklearn.preprocessing import StandardScaler
import os

class IWAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim, iw_samples=5):
        super(IWAE, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = 16
        self.epochs = 15
        self.lr = 1e-3
        self.warmup_epoch = 5
        self.device = 'cuda:0'
        self.iw_samples = iw_samples  # IWAE的K值

        self.linear = nn.Sequential(nn.Linear(768, input_dim))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.Tanh()
        )
        self.fc_recon_mu = nn.Linear(hidden_dim1, input_dim)
        self.fc_recon_logvar = nn.Linear(hidden_dim1, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder(z)
        return self.fc_recon_mu(h), self.fc_recon_logvar(h)

    def forward(self, x):
        B = x.size(0)
        mu, logvar = self.encode(x)
        mu = mu.unsqueeze(1).expand(B, self.iw_samples, self.latent_dim)
        logvar = logvar.unsqueeze(1).expand(B, self.iw_samples, self.latent_dim)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # [B, K, latent_dim]

        # 解码
        z_flat = z.view(B * self.iw_samples, self.latent_dim)
        h = self.decoder(z_flat)
        recon_mu = self.fc_recon_mu(h).view(B, self.iw_samples, -1)
        recon_logvar = self.fc_recon_logvar(h).view(B, self.iw_samples, -1)

        # 重建对数似然 log p(x|z)
        recon_log_prob = -0.5 * (torch.log(2 * torch.pi) + recon_logvar + (x.unsqueeze(1) - recon_mu).pow(2) / torch.exp(recon_logvar))
        recon_log_prob = recon_log_prob.sum(-1)

        # 先验和后验的log概率
        log_pz = -0.5 * (z.pow(2) + torch.log(2 * torch.pi)).sum(-1)
        log_qz = -0.5 * (((z - mu) ** 2) / torch.exp(logvar) + logvar + torch.log(2 * torch.pi)).sum(-1)

        log_weights = recon_log_prob + log_pz - log_qz
        log_weights = torch.logsumexp(log_weights, dim=1) - np.log(self.iw_samples)

        loss = -log_weights.mean()

        return recon_mu[:, 0, :], recon_logvar[:, 0, :], mu[:, 0, :], logvar[:, 0, :], loss




# --------------------------
# VAE 模型定义
# --------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2,latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # self.linear_layer = nn.Sequential(nn.Linear(Embed_dim, input_dim),nn.ReLU())
        self.batch_size = 16
        self.epochs = 15
        self.lr = 1e-3
        self.warmup_epoch = 5
        self.device = 'cuda:0'
        self.linear = nn.Sequential(nn.Linear(768, input_dim))
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)
        self.mu = nn.Linear(hidden_dim1, latent_dim)
        self.logvar = nn.Linear(hidden_dim1, latent_dim)
        self.fc_recon_mu = nn.Linear(hidden_dim1, input_dim)
        self.fc_recon_logvar = nn.Linear(hidden_dim1, input_dim) #两个重构线性层
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z):
        h = self.decoder(z)
        return self.fc_recon_mu(h), self.fc_recon_logvar(h)
    
    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.decode(z)
        return mu_x, logvar_x, mu_z, logvar_z

    
# --------------------------
# 损失函数：重构损失 + KL散度
# --------------------------
def vae_loss(recon_x, x, mu, logvar, kl_weight):
    
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_div, recon_loss, kl_div


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
def train_VAE(dataloader,input_dim, hidden_dim1, hidden_dim2,latent_dim,VAE_path,useVAE=True):
    if os.path.exists(VAE_path):
        return 
    if useVAE == True: 
        model = VAE(input_dim, hidden_dim1, hidden_dim2, latent_dim)
    else: 
        model = IWAE(input_dim, hidden_dim1, hidden_dim2, latent_dim)
    dataloader = batch(dataloader,model.batch_size)
    scaler = StandardScaler()
    model.to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lowest_loss = 10000
    for epoch in tqdm.tqdm(range(model.epochs)):
        model.train()
        total_loss = 0
        for x in dataloader:
            x = x.reshape((model.batch_size, -1))
            x = scaler.fit_transform(x)
            x = torch.from_numpy(x[0]).to(model.device)
            optimizer.zero_grad()
            mu_x, logvar_x, mu_z, logvar_z = model(x)
            kl_weight = min(1.0, epoch / model.warmup_epoch)
            loss, recon_loss , kl_loss = vae_loss(mu_x, logvar_x, mu_z, logvar_z, kl_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if total_loss < lowest_loss:
            lowest_loss = total_loss
            torch.save(model,f'{VAE_path}')
        print(f"Recon: {recon_loss.item():.3f}, KL: {kl_loss.item():.3f}")
        print(f"Epoch {epoch+1}/{model.epochs}, Loss: {total_loss:.4f}")
    return

def test_VAE(VAE_path, input_dim,test_dim, train_dataset,train_score,test_score,max_min_normalization=False,full_score=False):
    # pdb.set_trace()
    train_dataset = torch.tensor(train_dataset).reshape(-1,len(train_dataset[0][0])).to('cuda:0')
    VAE_test = torch.load(f'{VAE_path}/testVAE_dim{input_dim}.pkl')
    VAE_train = torch.load(f'{VAE_path}/trainVAE_dim{test_dim}.pkl')
    Weight =  extimate_importance_weight(test_dim,VAE_test, VAE_train,train_dataset).to('cpu')
    for key,value in train_score.items():
        Trainscore  = torch.tensor(train_score[key])
        Testscore  = torch.tensor(test_score[key])
        if max_min_normalization:
            combined = torch.cat((Trainscore, Testscore), dim=0)  # shape: [n + m]
            # 进行 Min-Max 归一化
            min_val = combined.min()
            max_val = combined.max()
            normalized_combined = (combined - min_val) / (max_val - min_val + 1e-8)  # 防止除以零
            # 再次拆分
            Trainscore = normalized_combined[:len(Trainscore)]
            Testscore = normalized_combined[len(Trainscore):]
        if full_score:
            Trainscore = Trainscore / full_score
            Testscore = Testscore / full_score
        result = torch.dot(Weight, Trainscore) / sum(Weight)
        goal = sum(Testscore) / len(Testscore)

        print(f'error of {key}:{result-goal}')


def estimate_log_likelihood(vae, x):
    """
    使用重构误差估计 log p(x)。注意这只是一个简化的例子，
    实际应用中可能需要更复杂的估计方法如 Importance Weighted Bound。
    """

    mu_z, logvar_z = vae.encode(x)
    z = vae.reparameterize(mu_z, logvar_z)
    mu_x, logvar_x = vae.decode(z)

    # 这里简单地使用重构误差作为 log p(x|z) 的近似
    # 在实际应用中，你可能需要更精确的方法来估算这个值
    reconstruction_error = -0.5 * ((x - mu_x) ** 2 / logvar_x.exp() + logvar_x).sum(dim=1)
    prior_log_likelihood = -0.5 * (mu_z ** 2 + logvar_z.exp()).sum(dim=1) - 0.5 * mu_z.size(1) * torch.log(torch.tensor(2 * torch.pi))
    
    return reconstruction_error + prior_log_likelihood
    
def extimate_importance_weight(test_dim,VAE_p,VAE_q,sample):
    log_px = estimate_log_likelihood(VAE_p, sample)
    sample.to('cuda:0')
    log_qx = estimate_log_likelihood(VAE_q, sample)
    log_weights = log_px - log_qx
    weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))  # 对权重进行归一化处理
    clip = torch.quantile(weights, 0.9)
    weights = torch.minimum(weights, clip)
    return weights


