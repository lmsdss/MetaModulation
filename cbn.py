import torch
import torch.nn as nn


class CBN(nn.Module):

    def __init__(self, task2_size, hidden_size, out_size, batch_size, channels, height, width, use_betas=True,
                 use_gammas=True, eps=1.0e-5):
        super(CBN, self).__init__()

        self.task2_size = task2_size  # size of the lstm emb which is input to MLP  lstm_size
        self.hidden_size = hidden_size  # size of hidden layer of MLP       emb_size
        self.out_size = out_size  # output of the MLP - for each channel
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width

        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(self.batch_size, self.channels).cuda())  # 5 32
        self.gammas = nn.Parameter(torch.ones(self.batch_size, self.channels).cuda())  # 5 32
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma_mu = nn.Sequential(
            nn.Linear(self.task2_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.out_size),
        ).cuda()

        self.fc_gamma_var = nn.Sequential(
            nn.Linear(self.task2_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.out_size),
        ).cuda()

        self.fc_beta_mu = nn.Sequential(
            nn.Linear(self.task2_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.out_size),
        ).cuda()

        self.fc_beta_var = nn.Sequential(
            nn.Linear(self.task2_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.out_size),
        ).cuda()

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def create_cbn_input(self, task_2_feature):
        delta_betas_mu = self.fc_beta_mu(task_2_feature)
        delta_betas_var = self.fc_beta_var(task_2_feature)

        delta_gammas_mu = self.fc_gamma_mu(task_2_feature)
        delta_gammas_var = self.fc_gamma_var(task_2_feature)

        return delta_betas_mu, delta_betas_var, delta_gammas_mu, delta_gammas_var

    def forward(self, task_1_feature, task_2_feature):
        self.batch_size, self.channels, self.height, self.width = task_1_feature.shape

        # get delta values
        delta_betas_mu, delta_betas_var, delta_gammas_mu, delta_gammas_var = self.create_cbn_input(task_2_feature)

        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()

        cbn_kl_loss = kl_div(delta_betas_mu, delta_betas_var, torch.zeros(delta_betas_mu.shape).to("cuda"),
                             torch.ones(delta_betas_var.shape).to("cuda"))

        # update the values of beta and gamma
        bstas_sample = sample_normal(delta_betas_mu, delta_betas_var, 100)  # the num of samples
        gammas_sample = sample_normal(delta_gammas_mu, delta_gammas_var, 100)  # the num of samples

        betas_cloned += bstas_sample.mean(0)
        gammas_cloned += gammas_sample.mean(0)

        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(task_1_feature)
        batch_var = torch.var(task_1_feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        betas_expanded = torch.stack([betas_cloned] * self.height, dim=2)
        betas_expanded = torch.stack([betas_expanded] * self.width, dim=3)

        gammas_expanded = torch.stack([gammas_cloned] * self.height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded] * self.width, dim=3)

        # normalize the feature map
        feature_normalized = (task_1_feature - batch_mean) / torch.sqrt(batch_var + self.eps)

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out, cbn_kl_loss


def sample_normal(mu, log_variance, num_samples):
    shape = mu.unsqueeze(0).expand(num_samples, mu.shape[0])

    eps = torch.randn(shape.shape).to("cuda")

    return mu + eps * torch.sqrt(torch.exp(log_variance))


def kl_div(m, log_v, m0, log_v0):
    v = log_v.exp()
    v0 = log_v0.exp()

    dout = m.shape[0]
    const_term = -0.5 * float(dout)

    log_std_diff = 0.5 * torch.sum(torch.log(v0) - torch.log(v))
    mu_diff_term = 0.5 * torch.sum((v + (m0 - m) ** 2) / v0)
    kl = const_term + log_std_diff + mu_diff_term
    return kl
