from torch import nn
import torch
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_, normal_
from torch.nn import init
# from torch.nn.parameter import Parameter
import numpy as np


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, device, n_factors=20, offset=True, mode='ce'):
        super(MatrixFactorization, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_factors = nn.Embedding(n_users, n_factors, dtype=torch.float32)
        # self.user_factors = nn.Embedding.from_pretrained(torch.ones(size=(n_users, n_factors), dtype=torch.float32)).to(device)
        # offset = False
        # print("************** User factors are ones and won't be trained. Offset False !!!!!!!!!!")
        self.item_factors = nn.Embedding(n_items, n_factors, dtype=torch.float32)
        self.offset = offset
        if self.offset:
            self.b = nn.Parameter(torch.randn(1))
            self.b_u = nn.Embedding(self.n_users, 1)
            self.b_i = nn.Embedding(self.n_items, 1)
        self.device = device
        self.mode = mode
        if mode == 'ce':
            self.layer = nn.Linear(n_factors, 5)
            self.m = nn.Softmax(dim=-1)
            self.loss_fct = nn.MSELoss(reduction='none')
        elif mode == 'bce':
            self.m = nn.Sigmoid()
            self.loss_fct = nn.BCELoss(reduction='none')
        elif mode == 'mse':
            self.loss_fct = nn.MSELoss(reduction='none')
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            if self.mode == 'mse':
                fan_in, fan_out = init._calculate_fan_in_and_fan_out(module.weight)
                std = np.sqrt(2.0 / (fan_in + fan_out))
                normal_(module.weight, mean=np.sqrt(1.0 /self.n_factors), std=std)
                print("--------- Initialize the weights with the mean equaling to sqrt(1.0/n_factors) !!!!")
            else:
                xavier_normal_(module.weight)
        elif isinstance(module, nn.Parameter):
            constant_(module.weight, 0.0)

    def forward(self, user, item):
        if self.mode == 'ce':
            scores = self.layer((self.user_factors(user) * self.item_factors(item)))
            return scores
        scores = (self.user_factors(user) * self.item_factors(item)).sum(1)
        if self.offset:
            return scores + self.b + self.b_u(user).squeeze() + self.b_i(item).squeeze()
        return scores

    def calculate_loss(self, interactions, get_preds=False):
        users, items, ratings = interactions
        # if self.mode == 'ce':
        #     ratings = ratings.to(torch.long).to(self.device)
        #     ratings = ratings - 1 # because range 1 - 5 means categories 0 - 4
        if self.mode in ['ce','bce']:
            preds = self.m(self.forward(users, items))
        elif self.mode == 'mse':
            preds = self.forward(users, items)
        if self.mode == 'ce':
            loss = self.loss_fct((preds * torch.arange(1, 6, dtype=torch.float32).to(self.device)).sum(1), ratings)
        else:
            loss = self.loss_fct(preds, ratings)
        if get_preds:
            return loss, preds
        return loss
    
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