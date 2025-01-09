from torch import nn
import torch
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_, normal_
from torch.nn import init
# from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F

# according to VAE for CF paper, 0 or 1 layer is best. So the architecture will be [I-128-I], tanh
class MultiVAE(nn.Module):
    def __init__(self, n_users, n_items, device, mode='mse', total_anneal_steps=1000):
        super(MultiVAE, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.lat_dim = 128
        self.drop_out = 0.0 # config["dropout_prob"]
        self.anneal_cap = 0.8 # config["anneal_cap"] # the beta in the paper, 0.8
        print("---------------------- annel_cap: %.2f  ------------------------" % self.anneal_cap)
        self.total_anneal_steps = total_anneal_steps # config["total_anneal_steps"] # 1000 for coat, 5000 for yahoo
        print("---------------------- total_anneal_steps: %d  ------------------------" % self.total_anneal_steps)

        self.encode_layer_dims = [self.n_items, self.lat_dim]
        self.decode_layer_dims = [int(self.lat_dim / 2), self.n_items]
        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = self.mlp_layers(self.decode_layer_dims)
        
        self.update = 0

        assert mode == 'mse'
        self.mode = mode
        self.loss_fct = nn.MSELoss(reduction='none')
        self.apply(self._init_weights)

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu

    def build_history_matrix(self, trainset, train_propensities_inversed=None):
        # get a dict: {user_id: [0,..,3,4,...]}, the current way may not suitable for large datasets
        history_matrix = np.zeros((self.n_users, self.n_items), dtype=np.int32)
        observation_matrix = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        for idx, v in trainset.reset_index().iterrows():
            u, i, r = int(v['user_id']), int(v['item_id']), v['rating']
            history_matrix[u, i] = r
            if train_propensities_inversed is not None:
                observation_matrix[u, i] = train_propensities_inversed[idx]
            else:
                observation_matrix[u, i] = 1
        self.history_matrix = torch.Tensor(history_matrix).to(self.device)
        self.observation_matrix = torch.Tensor(observation_matrix).to(self.device)
        # @TODO: propensity_matrix 

    def build_matrix_valid(self, validset, valid_propensities_inversed=None): # @TODO: for evaluation
        # history matrix will use the trainset, the validset only looked at the generated z and evaluate on valid data.
        valid_rating_matrix = np.zeros((self.n_users, self.n_items), dtype=np.int32)
        observation_matrix = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        if valid_propensities_inversed is not None:
            valid_propensities_inversed = valid_propensities_inversed / valid_propensities_inversed.mean()
        for idx, v in validset.reset_index().iterrows():
            u, i, r = int(v['user_id']), int(v['item_id']), v['rating']
            valid_rating_matrix[u, i] = r
            if valid_propensities_inversed is not None:
                observation_matrix[u, i] = valid_propensities_inversed[idx]
            else:
                observation_matrix[u, i] = 1
        self.valid_rating_matrix = torch.Tensor(valid_rating_matrix).to(self.device)
        self.valid_observation_matrix = torch.Tensor(observation_matrix).to(self.device)

    def get_rating_matrix(self, user):
        return self.history_matrix[user], self.observation_matrix[user] # (B, n_items)
    def get_valid_matrix(self, user):
        return self.valid_rating_matrix[user], self.valid_observation_matrix[user]

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, rating_matrix):
        h = F.normalize(rating_matrix)

        h = F.dropout(h, self.drop_out, training=self.training)

        h = self.encoder(h)

        mu = h[:, : int(self.lat_dim / 2)]
        logvar = h[:, int(self.lat_dim / 2) :]

        z = self.reparameterize(mu, logvar)
        z = self.decoder(z)
        return z, mu, logvar

    def calculate_loss(self, interactions, get_preds=False):
        users, items, ratings = interactions
        rating_matrix, observation_matrix = self.get_rating_matrix(users.long())
        
        if self.training:
            self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap
        z, mu, logvar = self.forward(rating_matrix)

        # KL loss
        kl_loss = (
            -0.5
            * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            * anneal
        )

        # # CE loss
        # ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()
        # # MSE loss
        if not self.training:
            rating_matrix, observation_matrix = self.get_valid_matrix(users.long())
        mse_loss = 0.5 * (self.loss_fct(z, rating_matrix) * observation_matrix).sum(1) # [B, n_items] -> [B, ]
        if get_preds:
            return mse_loss, kl_loss, z[[torch.arange(len(items)).to(self.device), items.long()]]
        return mse_loss, kl_loss
    
    @torch.no_grad()
    def full_sort_predict(self):
        user_embeddings = self.user_factors.weight.view(self.n_users, self.n_factors) # [M, D]
        item_embeddings = self.item_factors.weight.view(self.n_items, self.n_factors) # [N D]
        scores = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))  # [M D], [D N] -> [M N]
        if self.offset:
            scores += self.b + self.b_u.weight.view(self.n_users, 1) + self.b_i.weight.view(1, self.n_items)
        if self.mode in ['ce', 'bce']:
            return self.m(scores)
        elif self.mode == 'mse':
            return scores
        # return np.clip(scores.numpy(), 0, 5)