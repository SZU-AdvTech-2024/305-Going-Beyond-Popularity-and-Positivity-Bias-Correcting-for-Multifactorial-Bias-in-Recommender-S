# %% [markdown]
# **Debiased offline RL-based top-K ranking method**
# 
# Here we choose Yahoo! R3 dataset.

# %%
import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
from scipy.stats import powerlaw
import random
from sklearn.metrics import ndcg_score, log_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os 
import wandb


def set_hparams():
    import argparse
    parser = argparse.ArgumentParser()
    # config.update({'debiasing':'none', 'lr':1e-5, 'reg':1e-4, 'dim':128})
    parser.add_argument('--dataset_name', type=str, default='yahoo')
    parser.add_argument('--CF_model', type=str, default='MF')
    parser.add_argument('--debiasing', type=str, default='none')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--reg', type=float, default=0)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--alpha1', type=int, default=5)
    parser.add_argument('--alpha2', type=int, default=5)
    args = parser.parse_args()
    return args
args = set_hparams() # for multiple jobs
print(os.getcwd())

# %%
# # load original data
# # path = "/Users/huangjin/researchWork/data/Yahoo!R3/"
# path = "../data/yahoo/"
# trainset = pd.read_csv(path + 'ydata-ymusic-rating-study-v1_0-train.txt', header=None, sep='\t')
# trainset.columns = ['user_id', 'item_id', 'rating']
# testset = pd.read_csv(path + 'ydata-ymusic-rating-study-v1_0-test.txt', header=None, sep='\t')
# testset.columns = ['user_id', 'item_id', 'rating']
# trainset['reward'] = (trainset['rating'] >= 4).astype(np.float32)
# testset['reward'] = (testset['rating'] >= 4).astype(np.float32)
# testset_ = testset.copy()
# trainset_ = trainset.copy()

# %%
# # # Get 5% of MCAR testset to generate the propensities. And the remaining 95% as the testset.
# # # more like a positivity bias
# testset = testset_.copy()
# selection = np.random.uniform(size=len(testset)) <= 0.05
# MCAR = testset[selection]
# p_Y_O = [np.mean(trainset['rating'] == x) for x in range(1, 6)]
# p_Y = [np.mean(MCAR['rating'] == x) for x in range(1, 6)]
# testset = testset[~selection]
# p_O_Y = np.array(p_Y_O, dtype=np.float32) / np.array(p_Y, dtype=np.float32)
def get_positivity_bias(MCAR, trainset):
    p_Y_O = [np.mean(trainset['rating'] == x) for x in range(1, 6)] #记录每个分数在D中的比例
    p_Y = [np.mean(MCAR['rating'] == x) for x in range(1, 6)] #记录每个分数在MCAR中的比例
    p_O = len(trainset) / (trainset['user_id'].nunique() * trainset['item_id'].nunique()) #计算观察到的训练集的评分比例
    p_O_Y = np.array(p_Y_O, dtype=np.float32) / np.array(p_Y, dtype=np.float32) * p_O # 这个就是偏差值
    return p_O_Y, p_Y, p_Y_O, p_O

# # # estimation of proposities in term of popularity bias
def get_popularity_bias(trainset):
    item_freq = trainset.groupby(by="item_id")['user_id'].count() #groupby 方法按 item_id 分组，并统计每组中 user_id 的数量，得到每个项目的出现次数
    item_prob = np.zeros(num_items, dtype=np.float32)
    item_prob[item_freq.index.astype(int)] = item_freq.values / len(trainset)
    return item_prob

# # Estimation of propensities in term of positivity-popularity correlated bias
# # P(O|I,Y) = (P(I|Y, O) / P(I|Y)) * (P(Y|O) * P(O) / P(Y))
# # P(I|Y) is proportional to P(I) in the unbiased setting @TODO: It may not be true
def get_pos_pop_bias(MCAR, trainset): ## 原始的多因素倾向
    p_O_IY = np.zeros((num_items, 5), dtype=np.float32)
    p_O_Y, _, _, _ = get_positivity_bias(MCAR, trainset)
    p_I_YO = np.zeros((num_items, 5), dtype=np.float32)
    # item_freq_MCAR = MCAR.groupby(by="item_id").size() / len(MCAR) # @TODO: MCAR did not be done new item_id mapping
    # p_I = np.zeros(num_items, dtype=np.float32)
    # p_I[item_freq_MCAR.index.astype(int)] = item_freq_MCAR.values
    p_I = MCAR['item_id'].nunique() / len(MCAR) * 1.0  # @TODO: here we assume all the items get same exposure prob ignoring the ratings
    for i in range(1, 6):
        item_freq_i = trainset[trainset['rating'] == i].groupby(by="item_id")['user_id'].count()
        p_I_YO[item_freq_i.index.astype(int), i-1] = item_freq_i.values / item_freq_i.sum()
        p_O_IY[:, i-1] = p_I_YO[:, i-1] / p_I * p_O_Y[i-1]
    return p_O_IY, p_I, p_O_Y, p_I_YO

# # Estimation of propensities in term of positivity-popularity correlated bias
# # P(O|I,Y) = P(I,Y|O) / P(I,Y); P(I,Y) = P(I|Y) * P(Y)
def get_pos_pop_bias_updated(MCAR, trainset, clip=(1e-3, 0.2), alpha=(5, 5)): ##拉普拉斯平滑后的倾向
    p_O_Y, p_Y, _, _ = get_positivity_bias(MCAR, trainset)
    # P(I,Y|O) and P(I,Y)
    p_IY_O, p_IY = np.zeros((num_items, 5), dtype=np.float32), np.zeros((num_items, 5), dtype=np.float32)
    # # smoothing
    n_MCAR, n_train = len(MCAR), len(trainset)
    # alpha1, alpha2 = 5.0, 5.0 # a smaller value is typically chosen in practice.
    alpha1, alpha2 = alpha
    # selection = MCAR.copy().groupby(by='item_id').size()
    # MCAR_ = MCAR.copy()[MCAR['item_id'].isin(selection[selection>=5].index)]
    # print("----------- p(IY) estimation filtering")
    for r, group in MCAR.groupby(by='rating'):
        item_count = group.groupby(by='item_id').size() * 1.0
        v = np.zeros(num_items, dtype=np.float32)
        v[item_count.index.astype(int)] = item_count.values
        # # Laplace smoothing overall
        # p_IY[:, int(r)-1] = v * 1.0 / n_MCAR # (v + alpha1) / (n_MCAR + alpha1 * P_IY.size)
        # p_IY = p_I_Y * p_Y, p_I_Y estimation with Laplace smoothing
        p_I_Y = (v * 1.0 + alpha1) / (len(group) + alpha1 * num_items)
        p_IY[:, int(r-1)] = p_I_Y * p_Y[int(r-1)]
    for r, group in trainset.groupby(by='rating'):
        item_count = group.groupby(by='item_id').size() * 1.0
        v = np.zeros(num_items, dtype=np.float32)
        v[item_count.index.astype(int)] = item_count.values
        p_IY_O[:, int(r)-1] = (v + alpha2) / (n_train + alpha2 * p_IY_O.size)
    p_O_IY = p_IY_O / p_IY * (n_train * 1.0 / (num_items * num_users))
    print("!!!!!!! Laplace smoothing, alpha1&2 are", alpha1, alpha2)
    
    p_O_IY = np.clip(p_O_IY, clip[0], min(clip[1], p_O_IY.max()))
    print("Proposity clipping: [%.6f, %.6f]" % (p_O_IY.min(), p_O_IY.max()))
    
    # print("!!!!!!! Normalize p_O_IY per item")
    # p_O_IY = p_O_IY / p_O_IY.sum(1).reshape(-1, 1)
    print("!!!!!!! Normalize inversed p_O_IY per item")
    p_O_IY_inversed = np.reciprocal(p_O_IY)
    p_O_IY_inversed = p_O_IY_inversed / p_O_IY_inversed.sum(1).reshape(-1, 1)
    p_O_IY = np.reciprocal(p_O_IY_inversed)
    return p_O_IY

# %%
# # Generate the episodes/trajectories by order the interactions per user
def generate_transitions(trainset, behavior_policy='random', user_histories=None):
    transitions = []
    gamma_list = [0.0, 0.1, 0.4, 0.9, 1.0]
    gamma_Gt = {}
    trainset_ = trainset.copy()
    for u in trainset['user_id'].unique():
        interactions = trainset[trainset['user_id'] == u]
        if behavior_policy == 'random':
            interactions = interactions.sample(frac=1)
            trainset_[trainset_['user_id'] == u] = interactions.values
        
        # # # get the Gt used for REINFORCE, given the list of gamma, get the corresponding Gt
        # Gt_list = [[] for _ in range(len(gamma_list))]
        # for r in interactions['reward'].values[::-1]:
        #     if not Gt_list[0]:
        #         for i in range(len(gamma_list)):
        #             Gt_list[i].append(r)
        #     else:
        #         for i, gamma in enumerate(gamma_list):
        #             Gt_list[i].append(r + gamma * Gt_list[i][-1])
        # for i, gamma in enumerate(gamma_list):
        #     gamma_Gt.setdefault(gamma, [])
        #     Gt_list[i].reverse()
        #     gamma_Gt[gamma] += Gt_list[i]
        
        state = [[], []]
        if user_histories is not None:
            history = user_histories[user_histories['user_id'] == u]
            state = [list(history['item_id'].values), list(history['reward'])]
        for idx, (vid, r) in enumerate(interactions[['item_id', 'reward']].values):
            next_state = [state[0]+[vid], state[1]+[r]]
            assert len(state[0]) + 1 == len(next_state[0])
            transitions.append((u, state, int(vid), r, next_state))
            state = next_state
    transitions = np.array(transitions, dtype='object')
    return transitions, gamma_Gt, trainset_

# %%
# # user and item id to idx
# trainset = trainset_.copy()
# items = list(set(trainset['item_id'].unique()) | set(testset['item_id'].unique()))
# items2id = dict(zip(items, list(range(len(items)))))
# pd.options.mode.chained_assignment = None  # default='warn'
# trainset['item_id'] = trainset['item_id'].map(items2id)
# testset['item_id'] = testset['item_id'].map(items2id)
# items = list(range(len(items)))

# users = list(set(trainset['user_id'].unique()) | set(testset['user_id'].unique()))
# users2id = dict(zip(users, list(range(len(users)))))
# trainset['user_id'] = trainset['user_id'].map(users2id)
# testset['user_id'] = testset['user_id'].map(users2id)
# users = list(range(len(users)))
# num_items, num_users = len(items), len(users)

# %%
# trainsets, validsets = [], []  # 取训练集验证集
# train_transitions_lists, valid_transitions_list = [], []
# for trainset in [trainset.copy()]:
#     # # transition -based train/valid spltting.
#     selection = np.random.uniform(size=len(trainset)) <= 0.8
#     trainset, validset = trainset[selection], trainset[~selection]
#     # # # user-based train/valid splitting
#     # all_train_users = trainset['user_id'].unique()
#     # train_users = np.random.choice(all_train_users, size=int(0.8 * len(all_train_users)), replace=False)
#     # trainset, validset = trainset[trainset['user_id'].isin(train_users)], trainset[~trainset['user_id'].isin(train_users)]

#     # generate transitions for RL
#     train_transitions, _, trainset = generate_transitions(trainset)
#     valid_transitions, _, validset = generate_transitions(validset, user_histories=trainset)
#     print("The numbers of training and validation set are %d and %d" % (len(train_transitions), len(valid_transitions)))
#     trainsets.append(trainset)
#     validsets.append(validset)
#     train_transitions_lists.append(train_transitions)
#     valid_transitions_list.append(valid_transitions)

# %%
dataset_name = args.dataset_name
print("@@@@@@@@@@@ The dataset we used is", dataset_name)
if dataset_name == 'yahoo':
    out_path = "./data/yahoo_preprocessed/"
elif dataset_name == 'coat':
    out_path = "./data/coat/"
elif dataset_name == 'kuairec':
    out_path = "./data/kuairec_preprocessed/"
is_to_save, is_to_load = False, True  ## 这里你要手动改 ,开始的时候保存数据集save，后面就是加载load
if is_to_save:
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    trainsets[-1].to_csv(out_path + "trainset.csv", header=True, index=False)
    validsets[-1].to_csv(out_path + "validset.csv", header=True, index=False)
    testset.to_csv(out_path + "testset.csv", header=True, index=False)
    MCAR.to_csv(out_path + "MCAR.csv", header=True, index=False)

if is_to_load:
    trainsets = [pd.read_csv(out_path + "trainset.csv", header=0)]
    validsets = [pd.read_csv(out_path + "validset.csv", header=0)]
    testset = pd.read_csv(out_path + "testset.csv", header=0)
    task, mode = 'ratingPrediction', 'mse'
    if task == 'ratingPrediction':
        print("!!!! The current task is rating prediction task !!!!")
        trainsets[0]['reward'] = trainsets[0]['rating']
        validsets[0]['reward'] = validsets[0]['rating']
        testset['reward'] = testset['rating']

    MCAR = pd.read_csv(out_path + "MCAR_mapped.csv", header=0)
    train_transitions, _, _ = generate_transitions(trainsets[0], behavior_policy='none')
    valid_transitions, _, _ = generate_transitions(validsets[0], behavior_policy='none', user_histories=trainsets[0])
    users = (set(trainsets[0]['user_id'].values.astype(np.int32)) | set(validsets[0]['user_id'].values.astype(np.int32)))
    items = (set(trainsets[0]['item_id'].values.astype(np.int32)) | set(validsets[0]['item_id'].values.astype(np.int32)))
    num_users, num_items = len(users), len(items)
    print("The numbers of users and items are %d and %d" % (num_users, num_items))
    print("The numbers of training and validation set are %d and %d" % (len(train_transitions), len(valid_transitions)))

# # %%
# p_O_Y, p_Y, p_Y_O, p_O = get_positivity_bias(MCAR, pd.concat([trainsets[0], validsets[0]], ignore_index=True))

# # %%
# item_prob = get_popularity_bias(pd.concat([trainsets[0], validsets[0]], ignore_index=True))

# # # p_O_IY, _, _, _ = get_pos_pop_bias(MCAR, pd.concat([trainsets[0], validsets[0]], ignore_index=True))
# p_O_IY = get_pos_pop_bias_updated(MCAR, pd.concat([trainsets[0], validsets[0]], ignore_index=True))

def get_GT_P_O_IY(trainset, GT):
    num_users, num_items = trainset['user_id'].nunique(), trainset['item_id'].nunique()
    p_IY_O, p_IY = np.zeros((num_items, 5), dtype=np.float32), np.zeros((num_items, 5), dtype=np.float32)
    n_train = len(trainset)
    for r, group in trainset.groupby(by='rating'):
        item_count = group.groupby(by='item_id').size() * 1.0
        v = np.zeros(num_items, dtype=np.float32)
        v[item_count.index.astype(int)] = item_count.values
        p_IY_O[:, int(r)-1] = v / n_train
    alpha = 1
    for r, group in GT.groupby(by='rating'):
        item_count = group.groupby(by='item_id').size() * 1.0
        v = np.zeros(num_items, dtype=np.float32)
        v[item_count.index.astype(int)] = item_count.values
        p_IY[:, int(r-1)] = (v + alpha) * 1.0 / (len(GT) + alpha * num_items * 5)
    p_O_IY = p_IY_O / p_IY * (n_train * 1.0 / (num_users * num_items))
    print("!!!!!!! Normalize inversed p_O_IY per item")
    p_O_IY_inversed = np.reciprocal(np.clip(p_O_IY, 1e-4, p_O_IY.max()))
    p_O_IY_inversed = p_O_IY_inversed / p_O_IY_inversed.sum(1).reshape(-1, 1)
    p_O_IY = np.reciprocal(p_O_IY_inversed)
    return p_O_IY, p_IY_O, p_IY
# GT = pd.concat([MCAR, testset], axis=0, ignore_index=True)
# GT_p_O_IY, GT_p_IY_O, GT_p_IY = get_GT_P_O_IY(trainsets[0], GT)
# p_O_IY = GT_p_O_IY
# print("!!!!!!!!!!!!!!! Attention: Using the GT (MCAR+testset) to learn p_O_IY. Just for checking.")


# %% [markdown]
# The evaluation function for DQN and REINFORCE method. 
# Validation function only for DQN method.

# %%
# define the functions for validate and evaluate DQN
import torch
import importlib
import torch.optim as optim
from tqdm.notebook import trange
from functions import EarlyStopper
import seaborn as sns
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed=517):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

# %% [markdown]
# MF baseline

# %%
# MF implementation
import importlib
import mf
importlib.reload(mf)
from mf import MatrixFactorization
from vae import MultiVAE
from torch.optim.lr_scheduler import MultiStepLR

def eval_RMSE_per_group(preds, testset):
    testset_ = testset.copy()
    testset_['pred'] = preds
    results = {}
    for k in ['user_id','item_id']:
        rmse = []
        for kid, group in testset_.groupby(by=k):
            pred, true = group['pred'].values, group['rating'].values
            rmse.append(mean_squared_error(pred, true, squared=False))
        if 'user' in k:
            results['RMSE_U'] = np.mean(rmse)
        else:
            results['RMSE_I'] = np.mean(rmse)
    return results

def evaluate_mse(pred, true):
    results = {}
    results['eval_mse'] = mean_squared_error(true, pred)
    results['eval_mae'] = mean_absolute_error(true, pred)
    return results

def post_processing(model, testset):
    with torch.no_grad():
        test_ratings, test_users, test_items = input_to_cuda(testset, device)
        test_pred = model.calculate_loss([test_users, test_items, test_ratings], get_preds=True)[-1].cpu().numpy() # (B, 5) softmax done
        # no debiasing
        test_pred_naive = (test_pred * np.arange(1, 6)).sum(1)
        # IPS w/ positivity
        inversed_p_O_Y = np.reciprocal(np.array(p_O_Y)) # (5, )
        test_pred_ips_pos = test_pred * inversed_p_O_Y
        test_pred_ips_pos /= test_pred_ips_pos.sum(1).reshape(-1, 1)
        test_pred_ips_pos = (test_pred_ips_pos * np.arange(1, 6)).sum(1)
        # IPS w/ pos_pop
        inversed_p_O_IY = np.reciprocal(np.clip(p_O_IY[testset['item_id'].values.astype(np.int32)], 1e-4, 0.2)) # (B, 5)
        test_pred_ips_pos_pop = test_pred * inversed_p_O_IY
        test_pred_ips_pos_pop /= test_pred_ips_pos_pop.sum(1).reshape(-1, 1)
        test_pred_ips_pos_pop = (test_pred_ips_pos_pop * np.arange(1, 6)).sum(1)
        for test_pred in [test_pred_naive, test_pred_ips_pos, test_pred_ips_pos_pop]:
            result = {}
            testset_ = testset.copy()
            testset_['pred'] = test_pred
            assert task == 'ratingPrediction'
            result.update(eval_RMSE_per_group(testset_['pred'].values.clip(1, 5), testset))
            result['rmse'] = mean_squared_error(testset['reward'].values, testset_['pred'].values.clip(1, 5), squared=False)
            result['mse'] = mean_squared_error(testset['reward'].values, testset_['pred'].values.clip(1, 5))
            result['mae'] = mean_absolute_error(testset['reward'].values, testset_['pred'].values.clip(1, 5))
            # adding ranking evaluation metrics
            ndcgs = []
            for u, group in testset_.groupby(by='user_id'):
                ndcgs.append(ndcg_score([group['reward'].values], [group['pred'].values], k=10))
            result['ndcg'] = np.mean(ndcgs)
            print(result)

@torch.no_grad()
def MF_validate(model, valid_ratings, valid_users, valid_items, valid_propensities=None, batch_size=512):
    model.eval()
    start, total_loss = 0, 0.0
    if valid_propensities is not None:
        inversed_prop = torch.reciprocal(valid_propensities)
    while start < valid_ratings.shape[0]:
        end = start + batch_size
        batch_users, batch_items, batch_ratings = valid_users[start:end], valid_items[start:end], valid_ratings[start:end]
        loss = model.calculate_loss([batch_users, batch_items, batch_ratings])
        if valid_propensities is not None:
            loss = loss * inversed_prop[start:end] / inversed_prop.sum()
            total_loss += loss.sum().item()
        else:
            total_loss += loss.sum().item() / len(valid_users)
        start = end
    model.train()
    return total_loss

@torch.no_grad()
def VAE_validate(model, valid_users, batch_size=512):
    model.eval()
    valid_users = torch.unique(valid_users)
    start, total_loss = 0, 0.0
    while start < valid_users.shape[0]:
        end = start + batch_size
        batch_users = valid_users[start:end]
        loss, loss_KL = model.calculate_loss([batch_users, None, None])
        # loss_KL will not used in validation, as that is not our goal but just optimization loss
        total_loss += loss.mean().item()
        start = end
    model.train()
    return total_loss


# # # validation: mse-ips over users
# valid_loss_over_users = False
# print("************ Validatio
# n: Using average loss over %s ************ " % ('users' if valid_loss_over_users else "items")) # @here
# def avgloss_over_groups(loss, group_ids):
#     avg_losses = []
#     for gid in np.unique(group_ids):
#         avg_losses.append(np.mean(loss[group_ids == gid]))
#     return np.mean(avg_losses)
# @torch.no_grad()    
# def MF_validate(model, valid_ratings, valid_users, valid_items, valid_propensities=None, batch_size=512):
#     start, total_loss = 0, []
#     if valid_propensities is not None:
#         inversed_prop = torch.reciprocal(valid_propensities)
#     while start < valid_ratings.shape[0]:
#         end = start + batch_size
#         batch_users, batch_items, batch_ratings = valid_users[start:end], valid_items[start:end], valid_ratings[start:end]
#         _, preds = model.calculate_loss([batch_users, batch_items, batch_ratings], get_preds=True)
#         loss = torch.square(batch_ratings - preds)
#         if valid_propensities is not None:
#             loss = loss * inversed_prop[start:end] / inversed_prop.mean()
#         total_loss.append(loss.cpu().numpy())
#         start = end
#     total_loss = np.concatenate(total_loss, axis=0)
#     if valid_loss_over_users:
#         users = valid_users.cpu().numpy()
#         avgloss = avgloss_over_groups(total_loss, users)
#     else:
#         items = valid_items.cpu().numpy()
#         avgloss = avgloss_over_groups(total_loss, items)
#     return avgloss

def input_to_cuda(dataset, device):
    ratings = torch.tensor(dataset['reward'].values, dtype=torch.float32).to(device)
    users = torch.tensor(dataset['user_id'].values, dtype=torch.int32).to(device)
    items = torch.tensor(dataset['item_id'].values, dtype=torch.int32).to(device)
    return ratings, users, items
def MF_optmize(trainset, validset, testset, config=None, debiasing='none', use_wandb=False, CF_model='MF'):
    def evaluate_on_testset(get_pred=False):
        result = {}
        with torch.no_grad():
            # test_pred = model.full_sort_predict().cpu().numpy()[testset['user_id'].values.astype(np.int32), testset['item_id'].values.astype(np.int32)]
            test_ratings, test_users, test_items = input_to_cuda(testset, device)
            test_pred = model.calculate_loss([test_users, test_items, test_ratings], get_preds=True)[-1] # before we add vae, it is [1]
            testset_ = testset.copy()
            if config['mode'] == 'ce':
                testset_['pred'] = (test_pred.cpu().numpy() * np.arange(1, 6)).sum(1)
            else:
                testset_['pred'] = test_pred.cpu().numpy()
            if task == 'ratingPrediction':
                result.update(eval_RMSE_per_group(testset_['pred'].values.clip(1, 5), testset))
                result['rmse'] = mean_squared_error(testset['reward'].values, testset_['pred'].values.clip(1, 5), squared=False)
                result['mse'] = mean_squared_error(testset['reward'].values, testset_['pred'].values.clip(1, 5))
                result['mae'] = mean_absolute_error(testset['reward'].values, testset_['pred'].values.clip(1, 5))
                ndcgs = []
                for u, group in testset_.groupby(by='user_id'):
                    ndcgs.append(ndcg_score([group['reward'].values], [group['pred'].values], k=10))
                result['ndcg'] = np.mean(ndcgs)
            else:
                ndcgs = []
                for u, group in testset_.groupby(by='user_id'):
                    ndcgs.append(ndcg_score([group['reward'].values], [group['pred'].values], k=10))
                result['ndcg'] = np.mean(ndcgs)
                result['nll'] = log_loss(testset['reward'].values.astype(int), testset_['pred'].values)
                result['ppl'] = np.exp(result['nll'])
        if get_pred:
            wandb.log({"avg_pred": testset_['pred'].mean()})
        return result

    snips = config.get('SNIPS', True)
    if CF_model == 'MF':
        model = MatrixFactorization(num_users, num_items, device, n_factors=config.get("dim", 20), offset=True, mode=config.get("mode", 'ce')).to(device)
    elif CF_model == 'VAE':
        total_anneal_steps = 1000 if dataset_name == 'coat' else 0
        model = MultiVAE(num_users, num_items, device, mode=config.get("mode", 'mse'), total_anneal_steps=total_anneal_steps).to(device)

    if config.get("optim", 0) == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.get('lr', 1e-2), weight_decay=config.get('reg', 0.0))
    elif config.get("optim", 0) == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=config.get('reg', 0.0))
    elif config.get("optim", 0) == 'amsgrad':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=config.get('reg', 0.0), amsgrad=True)
    # scheduler = MultiStepLR(optimizer, milestones=[200, 500], gamma=0.2)
    earlystopping = EarlyStopper(patience=config.get("patience", 5))
    batch_size = 512
    epoch_info = trange(config.get('epochs', 1000))
    train_ratings, train_users, train_items = input_to_cuda(trainset, device)
    valid_ratings, valid_users, valid_items = input_to_cuda(validset, device)
    train_propensities, valid_propensities = None, None
    if debiasing == 'positivity':
        train_propensities = torch.tensor(p_O_Y[(trainset['rating'].values - 1).astype(np.int32)],dtype=torch.float32).to(device)
        valid_propensities =  torch.tensor(p_O_Y[(validset['rating'].values - 1).astype(np.int32)],dtype=torch.float32).to(device)
    elif debiasing == 'popularity':
        item_prob_ = np.clip(item_prob.copy(), 1e-3, item_prob.max())
        train_propensities = torch.tensor(item_prob_[trainset['item_id'].values.astype(np.int32)], dtype=torch.float32).to(device)
        valid_propensities = torch.tensor(item_prob_[validset['item_id'].values.astype(np.int32)], dtype=torch.float32).to(device)
    elif debiasing == 'multifactorial':
        # min_propensity = 1e-3
        # p_O_IY_ = np.clip(p_O_IY.copy(), min_propensity, min(0.2, p_O_IY.max()))
        # print("Proposity clipping: [%.6f, %.6f]" % (p_O_IY_.min(), p_O_IY_.max()))
        train_propensities = torch.tensor(p_O_IY[trainset['item_id'].values.astype(np.int32), trainset['rating'].values.astype(np.int32)-1], dtype=torch.float32).to(device)
        valid_propensities = torch.tensor(p_O_IY[validset['item_id'].values.astype(np.int32), validset['rating'].values.astype(np.int32)-1], dtype=torch.float32).to(device)
        # p_U_O = np.zeros(num_users, dtype=np.float32)
        # user_group = trainset.groupby(by='user_id').size()
        # p_U_O[user_group.index.values.astype(np.int32)] = user_group.values * 1.0 / num_items
        # p_U_O /= 10.0 / 1000.0 # TODO: (p_U)
        # print("######### with considering p_U_O, but U is independent to I and Y.")
        # train_propensities = torch.tensor(p_O_IY_[trainset['item_id'].values.astype(np.int32), trainset['rating'].values.astype(np.int32)-1] * p_U_O[trainset['user_id'].values.astype(np.int32)], dtype=torch.float32).to(device)
        # valid_propensities = torch.tensor(p_O_IY[validset['item_id'].values.astype(np.int32), validset['rating'].values.astype(np.int32)-1] * p_U_O[validset['user_id'].values.astype(np.int32)], dtype=torch.float32).to(device)
    elif debiasing == 'mf':
        # load propensities from files and then assign them to train and valid
        propensities = pd.read_csv('./propensities_gen_by_mf/%s.csv' % dataset_name)
        propensities = propensities.set_index(['user_id', 'item_id'])
        if dataset_name == 'yahoo':
            propensities['propensities'].clip(0.001, 1.0, inplace=True) # for yahoo
        elif dataset_name == 'coat':
            propensities['propensities'].clip(0.005, 1.0, inplace=True) # for coat
        print("Proposity clipping: [%.6f, %.6f]" % (propensities['propensities'].min(), propensities['propensities'].max()))
        train_propensities = torch.tensor(propensities.loc[zip(trainset['user_id'].values, trainset['item_id'].values)]['propensities'].values, dtype=torch.float32).to(device)
        valid_propensities = torch.tensor(propensities.loc[zip(validset['user_id'].values, validset['item_id'].values)]['propensities'].values, dtype=torch.float32).to(device)
    if debiasing != 'none':
        train_propensities_inversed = torch.reciprocal(train_propensities)
        train_propensities_inversed /= train_propensities_inversed.mean()
    # valid_propensities =  torch.tensor(p_O_Y[(validset['rating'].values - 1).astype(np.int32)],dtype=torch.float32).to(device)
    # print("************* Using P(O|Y) in validation ***************")
    # print("************* Using MCAR for validation ***************")
    # valid_ratings, valid_users, valid_items = input_to_cuda(MCAR, device)
    # valid_propensities = None

    if CF_model == 'VAE':
        model.build_history_matrix(trainset, train_propensities_inversed if debiasing != 'none' else None)
        model.build_matrix_valid(validset, torch.reciprocal(valid_propensities) if valid_propensities is not None else None)

    if config.get("plot", False):
        inversed_valid_prop = torch.reciprocal(valid_propensities)
        inversed_valid_prop /= inversed_valid_prop.mean()
        results_for_plot_per_repeat = []
    do_shuffle = True
    print("&&&&&&&&&& Do shuffle per epoch")
    for i in epoch_info:
        start, total_loss = 0, 0
        if do_shuffle:
            shuffled_index = torch.randperm(len(train_ratings))
            train_ratings, train_users, train_items = train_ratings[shuffled_index], train_users[shuffled_index], train_items[shuffled_index]
            if train_propensities is not None:
                train_propensities = train_propensities[shuffled_index]
                train_propensities_inversed = train_propensities_inversed[shuffled_index]

        while start < train_ratings.shape[0]:
            end = start + batch_size
            batch_users, batch_items, batch_ratings = train_users[start:end], train_items[start:end], train_ratings[start:end]
            optimizer.zero_grad()
            # Change the calculate loss not to be mean, and add P_OY inside
            loss = model.calculate_loss([batch_users, batch_items, batch_ratings])
            if CF_model == 'VAE':
                loss, loss_KL = loss[0], loss[1]
            elif CF_model == 'MF':
                if train_propensities is not None:
                    inversed_prop = train_propensities_inversed[start:end]
                    if snips:
                        loss = loss * inversed_prop / inversed_prop.mean()
                    else:
                        loss = loss * inversed_prop
            else:
                exit(1)
            loss = loss.mean()
            if CF_model == 'VAE':
                loss += loss_KL
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            start = end
        if CF_model == 'MF':
            valid_loss = MF_validate(model, valid_ratings, valid_users, valid_items, valid_propensities=valid_propensities, batch_size=batch_size)
        elif CF_model == 'VAE':
            valid_loss = VAE_validate(model, valid_users, batch_size=512)

        # valid_pred = model.full_sort_predict().cpu().numpy()[valid_users, valid_items]
        # results = evaluate_mse(valid_pred, valid_ratings)
        epoch_info.set_description("Epoch: %d , trainLoss: %.4f, EvalLoss: %.4f" % (i, total_loss, valid_loss))
        # print("Epoch: %d , trainLoss: %.4f, EvalLoss: %.4f" % (i, total_loss, valid_loss))
        # print("Testset:", evaluate_on_testset())
        if use_wandb:
            wandb.log({"train_loss":total_loss, "valid_loss":valid_loss})
            if (1 + i) % 1 == 0:
                wandb.log({"results":evaluate_on_testset(get_pred=True)})
        if config.get('plot', False) and (i % 10 == 0):
            valid_pred = model.full_sort_predict()[valid_users.long(), valid_items.long()]
            valid_mse = torch.square(valid_ratings - valid_pred)
            if valid_propensities is not None:
                valid_mse_ips = torch.mean(valid_mse * inversed_valid_prop).item()
            else:
                valid_mse = torch.mean(valid_mse).item()
            test_mse = evaluate_on_testset()['mse']
            results_for_plot_per_repeat.append((i, valid_mse_ips if valid_propensities is not None else valid_mse, test_mse))
            # print(results_for_plot_per_repeat[-1], datetime.now(), flush=True)
        if ((config.get('plot', False) and (i >= 1000)) or (not config.get('plot', False))) and earlystopping.early_stop(valid_loss):
            print("Early stopped at epoch %d" % i)
            break
    result = evaluate_on_testset()
    result['valid_loss'] = valid_loss
    if config.get("plot", False):
        results_for_plot.append(results_for_plot_per_repeat)
    if use_wandb:
        wandb.log({"results":result})
    return result, model

CF_model = args.CF_model #'VAE' # or 'MF'
print("@@@@@@@@@@@ The CF model we used is", CF_model)
set_random_seed()
repeat = 10
results = []
use_wandb = False
config = {'optim':'adam'}
config.update({'debiasing':args.debiasing, 'lr':args.lr, 'reg':args.reg, 'dim':args.dim})

config['plot'] = False # default False
if config.get('plot', False):
    results_for_plot = [] # per repeat, a list of (epoch, valid_mse, test_mse)
config['SNIPS'] = False # default True
config['epochs'] = 2000
config['mode'] = 'mse' if task == 'ratingPrediction' else 'bce'
# config['mode'] = 'ce' # To generate P(Y|I,O)
config['patience'] = 5
config['train_on_testusers_only'] = True
config['fine_tune_smoothing_in_MCAR'] = False # use the results on MCAR to find the best smoothing parameters.
if config.get('fine_tune_smoothing_in_MCAR', False):
    print("!!!!!!!! Fine tuning the smoothing parameters based on the results on MCAR !!!!!!!!!!!! ")
print(config)
# # estimate the propensities
trainset, validset = trainsets[0], validsets[0]
if config.get("train_on_testusers_only", False):
    test_users = testset['user_id'].unique()
    trainset, validset = trainsets[0][trainsets[0]['user_id'].isin(test_users)], validsets[0][validsets[0]['user_id'].isin(test_users)]
p_O_Y, p_Y, p_Y_O, p_O = get_positivity_bias(MCAR, pd.concat([trainset, validset], ignore_index=True))
item_prob = get_popularity_bias(pd.concat([trainset, validset], ignore_index=True))
if dataset_name == 'yahoo':
    alpha = (2, 10) # for yahoo
    clip = (1e-3, 0.2)
elif dataset_name == 'coat':   
    alpha = (5, 5) 
    clip = (1e-3, 0.2)
elif dataset_name == 'kuairec':
    alpha = (5, 5)
    clip = (1e-3, 0.1)
p_O_IY = get_pos_pop_bias_updated(MCAR, pd.concat([trainset, validset], ignore_index=True), clip=clip, alpha=alpha)

if use_wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MF-IPS-ratingPrediction",
        # track hyperparameters and run metadata
        config=config
    )
    
for _ in range(repeat):
    result, model = MF_optmize(trainset, validset, testset, config=config, debiasing=config.get('debiasing', 'none'), use_wandb=use_wandb, CF_model=CF_model)
    results.append(result)
    print(results[-1], flush=True)
    
s = ""
for k in results[0].keys():
    v = [x[k] for x in results]
    s += " %.4f (%.4f)" % (np.mean(v), np.std(v))
print("The avg results on testset for", results[0].keys(), " are", s)



if use_wandb:
    wandb.finish()
