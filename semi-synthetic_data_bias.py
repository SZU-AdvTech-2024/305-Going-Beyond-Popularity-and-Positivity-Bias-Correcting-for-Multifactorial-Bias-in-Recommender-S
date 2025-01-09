# %% [markdown]
# Due to the fact that the generated ratings above are based on the dot-product of user embeddings and item embeddings, MF matches the way to generate data and well predict ratings. MF performs better and is more robust than NMF.
# 
# Considering this, we need more complex and realistic user ratings. But it is hard to evaluate the realistic of the simulated user ratings. Hence, we would use the real user ratings, user/item profiles, and re-grouping/splitting the data.
# 
# Here, we choose [KuaiRec](https://arxiv.org/pdf/2202.10842.pdf) which contains a fully-observed user-item interaction matrix and item categories. 
# 

# %%
import numpy as np
from scipy.special import softmax
from matplotlib import pyplot as plt 
import pandas as pd
from scipy.stats import powerlaw
import wandb

def set_hparams():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ALS', type=str, default='False')
    parser.add_argument('--debiasing', type=str, default=None)
    parser.add_argument('--mul_alpha', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--reg', type=float, default=0)
    parser.add_argument('--dim', type=int, default=0)
    args = parser.parse_args()
    return args
args = set_hparams() # for multiple jobs


path = "./data/KuaiRec/data"
ratingMatrix = pd.read_csv(path + '/small_matrix.csv')
ratingMatrix.rename(columns={'video_id': 'item_id'}, inplace=True)
# item_categories = pd.read_csv(path + '/item_categories.csv')
# item_categories.rename(columns={'video_id': 'item_id'}, inplace=True)
items = ratingMatrix['item_id'].unique()
# item_categories = item_categories[item_categories['item_id'].isin(items)]

# user and item id to idx
ratingMatrix_ = ratingMatrix.copy()
items2id = dict(zip(items, list(range(len(items)))))
pd.options.mode.chained_assignment = None  # default='warn'
ratingMatrix['item_id'] = ratingMatrix['item_id'].map(items2id)
items = list(range(len(items)))

users = ratingMatrix['user_id'].unique()
users2id = dict(zip(users, list(range(len(users)))))
ratingMatrix['user_id'] = ratingMatrix['user_id'].map(users2id)
users = list(range(len(users)))
num_items, num_users = len(items), len(users)
ratingMatrix_ = ratingMatrix.copy()

# %%
import torch
import random
def set_random_seed(seed=517):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    
# # # estimation of proposities in term of popularity bias
def get_popularity_bias(trainset):
    item_freq = trainset.groupby(by="item_id")['user_id'].count()
    item_prob = np.zeros(num_items, dtype=np.float32)
    item_prob[item_freq.index.astype(int)] = item_freq.values / len(trainset)
    return item_prob

def get_positivity_bias(MCAR, trainset):
    p_Y_O = [np.mean(trainset['rating'] == x) for x in range(1, 6)]
    p_Y = [np.mean(MCAR['rating'] == x) for x in range(1, 6)]
    p_O = len(trainset) / (trainset['user_id'].nunique() * trainset['item_id'].nunique())
    p_O_Y = np.array(p_Y_O, dtype=np.float32) / np.array(p_Y, dtype=np.float32) * p_O
    return p_O_Y, p_Y, p_Y_O, p_O

# %%
p_Y = [0.5148, 0.2525, 0.1496, 0.0554, 0.0277]
splits = np.round(np.cumsum(p_Y) * len(ratingMatrix)).astype(np.int32)
splits = [0] + list(np.sort(ratingMatrix['watch_ratio'].values)[splits-1])
ratingMatrix['rating'] = 1
for i in range(1,len(splits)):
    ratingMatrix['rating'][ratingMatrix['watch_ratio'].between(splits[i-1], splits[i],inclusive='right')] = i
ratingMatrix_ = ratingMatrix.copy()

# %%
# # sampling only according to positivity, try to observe the popularity distribution.
ratingMatrix = ratingMatrix_.copy()
set_random_seed()
ratingMatrix['reward'] = ratingMatrix['rating']
# testset: per user 40 items. Group the DataFrame by user and randomly select items for each user
testset = ratingMatrix.groupby('user_id').apply(lambda x: x.sample(n=40, replace=True)).reset_index(drop=True)
ratingMatrix = pd.concat([ratingMatrix, testset]).drop_duplicates(keep=False).reset_index(drop=True)
# get 20% of testset as MCAR and the remaining 80% as the testset for evaluation.
selection = np.random.uniform(size=len(testset)) <= 0.2
MCAR = testset[selection]
testset = testset[~selection]

def get_power_law(num_items):
    alpha = 1.4
    print("Power-Law distribution generation on sharpness:", alpha)
    xmin = 120
    x = np.linspace(xmin, num_items, num_items)
    pdf = (alpha - 1) * (x / xmin)**(-alpha)
    probabilities = pdf / np.sum(pdf)
    return probabilities

# trainset
sampling_bias = "multifactorial" # "positivity" or "popularity" or "multifactorial"
print("!!!!! The training set is sampled from ratingMatrix following", sampling_bias + " bias.")
if sampling_bias == 'none':
    trainset = ratingMatrix[np.random.uniform(size=len(ratingMatrix)) <= 0.02]
elif sampling_bias == "positivity":
    p_O_Y = np.array([0.0123, 0.0102, 0.0213, 0.0568 , 0.1795])
    trainset = ratingMatrix[p_O_Y[ratingMatrix['rating'].values.astype(np.int32)-1] >= np.random.uniform(size=len(ratingMatrix))]
elif sampling_bias == "popularity":
    # the sampling is only based on popularity but popularity of items is correlated with avg
    pList = get_power_law(num_items)
    items_sorted = ratingMatrix.groupby(by=['item_id'])['rating'].mean().sort_values(ascending=False).index
    items_pList = dict(zip(items_sorted, pList))
    pList = np.array([x[1] for x in sorted(items_pList.items())])
    p_O_I = pList
    observed_prob = pList[ratingMatrix['item_id'].values.astype(np.int32)]
    num_interactions = int(0.02 * len(ratingMatrix))
    observed_prob = observed_prob * num_interactions * 1.0 / observed_prob.sum()
    selection = np.random.uniform(size=len(ratingMatrix)) <= observed_prob
    trainset = ratingMatrix[selection]
elif sampling_bias == 'multifactorial':
    # define p_O_Y and p_O_I respectively
    p_O_Y = np.array([0.0123, 0.0102, 0.0213, 0.0568 , 0.1795])
    pList = get_power_law(num_items)
    items_sorted = ratingMatrix.groupby(by=['item_id'])['rating'].mean().sort_values(ascending=False).index
    items_pList = dict(zip(items_sorted, pList))
    p_O_I = np.array([x[1] for x in sorted(items_pList.items())])
    # # combine both with alpha weights as final observed_prob
    num_interactions = int(0.02 * len(ratingMatrix))
    # mul_alpha = 0.2
    mul_alpha = args.mul_alpha
    print("@@@@@@@@@@@@@@ The sampling bias is multivaraite bias, and the alpha to combine p(O|Y) and p(O|I) is", mul_alpha)
    p_O_I = p_O_I * 0.02 / p_O_I.mean()
    p_O_IY = mul_alpha * p_O_Y.reshape(1, -1).repeat(num_items, axis=0) + \
                (1 - mul_alpha) * p_O_I.reshape(-1, 1).repeat(5, axis=1)    
    observed_prob = p_O_IY[ratingMatrix['item_id'].values.astype(np.int32), ratingMatrix['rating'].values.astype(np.int32)-1]
    print("Density is %.4f" % np.mean(observed_prob))
    selection = np.random.uniform(size=len(ratingMatrix)) <= observed_prob
    trainset = ratingMatrix[selection]

# visualization
plt.hist(trainset['rating'].values, bins=list(range(1,7)))
plt.xlabel('Rating')
plt.ylabel('Proportion')
plt.show()

item_counts = trainset.groupby('item_id').size().values
sorted_item_counts = sorted(item_counts, reverse=True)
plt.plot(np.arange(1, len(sorted_item_counts)+1),sorted_item_counts)
plt.xlabel('Item')
plt.ylabel('Item popularity')
plt.show()


# # Estimation of propensities in term of positivity-popularity correlated bias
# # P(O|I,Y) = P(I,Y|O) / P(I,Y); P(I,Y) = P(I|Y) * P(Y)
def get_pos_pop_bias_updated(MCAR, trainset, clip=(1e-3, 0.2), alpha=(5.0, 5.0)): #clip(1e-3, 0.2) for pop-pos on all users. And (1e-3, 0.08) for only testusers.
    p_O_Y, p_Y, _, _ = get_positivity_bias(MCAR, trainset)
    # P(I,Y|O) and P(I,Y)
    p_IY_O, p_IY = np.zeros((num_items, 5), dtype=np.float32), np.zeros((num_items, 5), dtype=np.float32)
    # # smoothing
    n_MCAR, n_train = len(MCAR), len(trainset)
    alpha1, alpha2 = alpha # a smaller value is typically chosen in practice.
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
    
    print("Proposity: [%.6f, %.6f]" % (p_O_IY.min(), p_O_IY.max()))
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
# training and validation splitting 
selection = np.random.uniform(size=len(trainset)) <= 0.8
trainset, validset = trainset[selection], trainset[~selection]

# Delete the interactions on items that donot appear in the training set.
testset = testset[testset['item_id'].isin(trainset['item_id'].values)]
print("After deleting the items which are not in trainset, the number of interactions and items in testset are %d and %d" % (len(testset), testset['item_id'].nunique()))
testset = testset.reset_index(drop=True) # avoid the old index being added as a column


# %% [markdown]
# After creating the dataset, we can train the simple matrix factorization (MF) and average (Avg) on the training set and evaluate it on the test sets.

# %%
import importlib
import mf
importlib.reload(mf)
from mf import MatrixFactorization
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import ndcg_score, log_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error
# import wandb
from tqdm.notebook import trange
from functions import EarlyStopper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

use_mse_loss_in_validation = False
@torch.no_grad()
def MF_validate(model, valid_ratings, valid_users, valid_items, valid_propensities=None, batch_size=512):
    start, total_loss = 0, 0.0
    if valid_propensities is not None:
        inversed_prop = torch.reciprocal(valid_propensities)
    while start < valid_ratings.shape[0]:
        end = start + batch_size
        batch_users, batch_items, batch_ratings = valid_users[start:end], valid_items[start:end], valid_ratings[start:end]
        loss = model.calculate_loss([batch_users, batch_items, batch_ratings], get_preds=True)
        mse_loss, preds = model.calculate_loss([batch_users, batch_items, batch_ratings], get_preds=True)
        mae_loss = torch.abs(batch_ratings - preds)
        loss = mse_loss if use_mse_loss_in_validation else mae_loss
        if valid_propensities is not None:
            loss = loss * inversed_prop[start:end] / inversed_prop.sum()
            total_loss += loss.sum().item()
        else:
            total_loss += loss.sum().item() / len(valid_users)
        start = end
    return total_loss

def input_to_cuda(dataset, device):
    ratings = torch.tensor(dataset['reward'].values, dtype=torch.float32).to(device)
    users = torch.tensor(dataset['user_id'].values, dtype=torch.int32).to(device)
    items = torch.tensor(dataset['item_id'].values, dtype=torch.int32).to(device)
    return ratings, users, items
def MF_optmize(trainset, validset, testset, config=None, debiasing='none', use_wandb=False):
    def evaluate_on_testset(get_pred=False):
        result = {}
        with torch.no_grad():
            # test_pred = model.full_sort_predict().cpu().numpy()[testset['user_id'].values.astype(np.int32), testset['item_id'].values.astype(np.int32)]
            test_ratings, test_users, test_items = input_to_cuda(testset, device)
            test_pred = model.calculate_loss([test_users, test_items, test_ratings], get_preds=True)[1]
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
    model = MatrixFactorization(num_users, num_items, device, n_factors=20, offset=True, mode=config.get("mode", 'ce')).to(device)
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
        item_prob = p_O_I.copy()
        item_prob_ = np.clip(item_prob.copy(), 1e-3, item_prob.max())
        train_propensities = torch.tensor(item_prob_[trainset['item_id'].values.astype(np.int32)], dtype=torch.float32).to(device)
        valid_propensities = torch.tensor(item_prob_[validset['item_id'].values.astype(np.int32)], dtype=torch.float32).to(device)
    elif debiasing == 'multifactorial':
        train_propensities = torch.tensor(p_O_IY[trainset['item_id'].values.astype(np.int32), trainset['rating'].values.astype(np.int32)-1], dtype=torch.float32).to(device)
        valid_propensities = torch.tensor(p_O_IY[validset['item_id'].values.astype(np.int32), validset['rating'].values.astype(np.int32)-1], dtype=torch.float32).to(device)
    if debiasing != 'none':
        train_propensities_inversed = torch.reciprocal(train_propensities)
        train_propensities_inversed /= train_propensities_inversed.mean()
    def optimize_per_epoch():
        start, total_loss = 0, 0
        while start < train_ratings.shape[0]:
            end = start + batch_size
            batch_users, batch_items, batch_ratings = train_users[start:end], train_items[start:end], train_ratings[start:end]
            optimizer.zero_grad()
            # Change the calculate loss not to be mean, and add P_OY inside
            loss = model.calculate_loss([batch_users, batch_items, batch_ratings])
            if train_propensities is not None:
                inversed_prop = train_propensities_inversed[start:end]
                if snips:
                    loss = loss * inversed_prop / inversed_prop.mean()
                else:
                    loss = loss * inversed_prop
            loss = loss.mean()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            start = end
        return total_loss
    do_shuffle = True
    print("&&&&&&&&&& Do shuffle per epoch")
    for i in epoch_info:
        if do_shuffle:
            shuffled_index = torch.randperm(len(train_ratings))
            train_ratings, train_users, train_items = train_ratings[shuffled_index], train_users[shuffled_index], train_items[shuffled_index]
            if train_propensities is not None:
                train_propensities = train_propensities[shuffled_index]
                train_propensities_inversed = train_propensities_inversed[shuffled_index]
        if config.get("sep-learning", False):
            # update u related params w/ P(O|IY)
            for name, param in model.named_parameters():
                param.requires_grad = True
                if name in ['item_factors.weight', 'b_i.weight']:
                    param.requires_grad = False
            optimize_per_epoch()
            # I-learning: update i related params 
            for name, param in model.named_parameters():
                param.requires_grad = True
                if name not in ['item_factors.weight', 'b_i.weight']:
                    param.requires_grad = False
            total_loss = optimize_per_epoch()
        else:
            total_loss = optimize_per_epoch()
        valid_loss = MF_validate(model, valid_ratings, valid_users, valid_items, valid_propensities=valid_propensities, batch_size=batch_size)
        # valid_pred = model.full_sort_predict().cpu().numpy()[valid_users, valid_items]
        # results = evaluate_mse(valid_pred, valid_ratings)
        epoch_info.set_description("Epoch: %d , trainLoss: %.4f, EvalLoss: %.4f" % (i, total_loss, valid_loss))
        # print("Epoch: %d , trainLoss: %.4f, EvalLoss: %.4f" % (i, total_loss, valid_loss))
        # print("Testset:", evaluate_on_testset())
        if use_wandb:
            wandb.log({"train_loss":total_loss, "valid_loss":valid_loss})
            if (1 + i) % 1 == 0:
                wandb.log({"results":evaluate_on_testset(get_pred=True)})
        if earlystopping.early_stop(valid_loss):
            print("Early stopped at epoch %d" % i)
            break
    result = evaluate_on_testset()
    result['valid_loss'] = valid_loss
    if use_wandb:
        wandb.log({"results":result})
    return result, model

task, mode = 'ratingPrediction', 'mse'
# %%
set_random_seed()
repeat = 10
results = []
# config = {'optim':'sgd', 'lr':1e-3}
config = {'optim':'adam', 'lr':1e-5}
# config = {'optim':'amsgrad', 'lr':1e-4}
config['lr'] = 1e-4 if args.lr is None else args.lr
config['reg'] = 1e-4 if args.reg is None else args.reg
config['dim'] = 32 if args.dim is None else args.dim
config['debiasing'] = args.debiasing if args.debiasing is not None else 'none'
config['SNIPS'] = False # default True
config['epochs'] = 2000
config['mode'] = 'mse' if task == 'ratingPrediction' else 'bce'
# config['mode'] = 'ce' # To generate P(Y|I,O)
config['patience'] = 5
# config['sep-learning'] = True # separate the learning of item-related and other params @here
config['sep-learning'] = True if args.ALS.lower() == 'true' else False
print(config)
use_wandb = False
if use_wandb:
    repeat = 1
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MF-IPS-simulation",
        # track hyperparameters and run metadata
        config=config
    )

# add estimation of p_O_IY
if config['debiasing'] not in ['GT', 'none']:
    p_O_I = get_popularity_bias(trainset)
    p_O_Y, p_Y, p_Y_O, p_O = get_positivity_bias(MCAR, trainset)
    # p_O_IY = get_pos_pop_bias_updated(MCAR, trainset, clip=(1e-3, 0.08), alpha=(4, 4)) # best for sampling bias: positivity 
    if sampling_bias == 'positivity':
        p_O_IY = get_pos_pop_bias_updated(MCAR, trainset, clip=(1e-4, 0.08), alpha=(4, 4)) # best for sampling bias: positivity 
    elif sampling_bias == 'popularity':
        p_O_IY = get_pos_pop_bias_updated(MCAR, trainset, clip=(1e-4, 0.01), alpha=(0.5, 0.5)) # best for sampling bias: popularity
    elif sampling_bias == 'multifactorial':
        a = 1 if (sampling_bias == 'multifactorial') and (mul_alpha <= 0.8) else 2
        max_ = 0.05
        if (sampling_bias == 'multifactorial'):
            if mul_alpha <= 0.2:
                max_ = 0.02
            elif mul_alpha >= 0.8:
                max_ = 0.08
        p_O_IY = get_pos_pop_bias_updated(MCAR, trainset, clip=(1e-4, max_), alpha=(a, a))
else:
    if config['debiasing'] == 'GT':
        print("!!!!!!!!!!!! Using GT propensities !!!!!!!!!!!!! ")
        config['debiasing'] = sampling_bias

set_random_seed()
for _ in range(repeat):
    result, model = MF_optmize(trainset, validset, testset, config=config, debiasing=config.get('debiasing', 'none'), use_wandb=use_wandb)
    results.append(result)
    print(results[-1], flush=True)

s = ""
for k in results[0].keys():
    v = [x[k] for x in results]
    s += " %.4f (%.4f)" % (np.mean(v), np.std(v))
print("The avg results on testset for", results[0].keys(), " are", s)

if use_wandb:
    wandb.finish()
