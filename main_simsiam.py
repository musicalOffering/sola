import matplotlib.pyplot as plt
import argparse
import yaml
import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from yaml.loader import FullLoader
from dataset import VideoPretrain
from network import FeatureTransformer, Projector, MLPHead
from utils import process_feature, make_evaluation_db, linear_probing

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def pairwise_cosine_similarity(x, y):
    '''
    calculate self-pairwise cosine similarity
    input:
    x: torch.FloatTensor [B,C,L,E']
    y: torch.FloatTensor [B,C,L,E']
    output:
    xcos_dist: torch.FloatTensor [B,C,L,L]
    '''
    x = x.detach()
    y = y.permute(0,1,3,2)
    dot = torch.matmul(x, y)
    x_dist = torch.norm(x, p=2, dim=3, keepdim=True)
    y_dist = torch.norm(y, p=2, dim=2, keepdim=True)
    dist = x_dist * y_dist
    cos = dot / (dist + 1e-8)
    return cos

def get_oracle_prob(size, coef):
    tmp = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                tmp[i, j] = 1.
            else:
                logit = ((1/np.abs(i-j))**2)*coef
                tmp[i, j] = sigmoid(logit)
                
    tmp = torch.from_numpy(tmp).float().unsqueeze(0).cuda()
    return tmp
    

def visualize(network, config, fig, name='initial'):
    network.eval()
    name = f'{str(name)}.png'
    vis_list = [torch.from_numpy(
                np.load(osp.join(config["visualizing_dir"], file_name)).astype(np.float32)
            ).cuda() for file_name in os.listdir(config['visualizing_dir'])]

    with torch.no_grad():
        cnt = 0
        for vis_feature in vis_list:
            o = network(vis_feature.unsqueeze(0)).unsqueeze(1)
            sim = pairwise_cosine_similarity(o, o).squeeze().cpu().numpy()
            ax = fig.add_subplot(3, 2, cnt+1)
            ax.imshow(sim, vmin=-1, vmax=1, interpolation='nearest')
            cnt += 1
    plt.savefig(osp.join(config["visualizing_save_path"], config["exp_name"], name))
    plt.clf()
    network.train()

def simsiam_visualize(sim_maps, oracle_tsm, fig, name='simsiam_vis'):
    #sim_maps: B,L,L
    name = f'{str(name)}.png'
    mean_tsm = torch.mean(sim_maps, dim=0).detach().cpu().numpy()
    sim_maps = sim_maps[:4]
    cnt = 0
    for sim_map in sim_maps:
        sim_map = sim_map.detach().cpu().numpy()
        ax = fig.add_subplot(3, 2, cnt+1)
        ax.imshow(sim_map, vmin=-1, vmax=1, interpolation='nearest')
        cnt += 1
    ax = fig.add_subplot(3,2,5)
    ax.imshow(mean_tsm, vmin=-1, vmax=1, interpolation='nearest')
    oracle_tsm = oracle_tsm.squeeze().detach().cpu().numpy()
    ax = fig.add_subplot(3, 2, 6)
    ax.imshow(oracle_tsm, vmin=0, vmax=1, interpolation='nearest')
    plt.savefig(f'{name}')
    plt.clf()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    torch.set_printoptions(threshold=np.inf)
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', required=True)
    args = parser.parse_args()
    yaml_path = args.yaml_path
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=FullLoader)
    print(f'EXPERIENCE_NAME: {config["exp_name"]}')
    model_save_path = osp.join(config["model_save_path"], config["exp_name"])
    if not osp.isdir(model_save_path):
        os.mkdir(model_save_path)
    visualizing_save_path = osp.join(config["visualizing_save_path"], config["exp_name"])
    if not osp.isdir(visualizing_save_path):
        os.mkdir(visualizing_save_path)
    dataset = VideoPretrain(subset='training', config=config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    feature_transformer = FeatureTransformer(config)
    projector = MLPHead(config)
    dissimhead = Projector(config, head='dissim')
    oracle_tsm = get_oracle_prob(config['tsm_size'], config['oracle_coef'])
    bceloss = nn.BCELoss()
    index = [config['step']*i + config['step_start'] for i in range(config['tsm_size'])]
    print(f'index: {index}')
    print(f'sampling_duration: {config["sampling_duration"]}')
    fig = plt.gcf()
    visualize(feature_transformer, config, fig, name=f'{config["exp_name"]}_epoch0')
    cnt = 0
    process_feature(config, 0, process_subset=True)
    make_evaluation_db(config, load_model='process_subset')
    eval_result = linear_probing(config, load_model='process_subset')
    print(f'eval_result in epoch{0}: {eval_result}')
    for epoch in range(config['epoch']):
        iter_cnt = 0
        epoch += 1
        sim_match_loss_cum = 0
        vector_norm_ratio_cum = 0
        #print(f'epoch {epoch}...')
        feature_transformer.train()
        for videos in dataloader:
            cnt += 1
            iter_cnt += 1
            videos = videos.cuda()
            processed = feature_transformer(videos)
            processed = projector(processed[:,index,:])
            dissim_in = processed
            if config['use_dissim_head']:
                dissim_out = dissimhead(dissim_in).unsqueeze(1)
            else:
                dissim_out = dissim_in.unsqueeze(1)
            dissim_similarities = pairwise_cosine_similarity(dissim_in.unsqueeze(1), dissim_out)
            if cnt % 25 == 1:
                simsiam_visualize(dissim_similarities.squeeze(), oracle_tsm, fig, config['exp_name'])
            probs = (1+dissim_similarities.squeeze())/2
            probs[torch.ge(probs, 1)] = 1 - 1e-5
            probs[torch.le(probs, 0)] = 1e-5
            if torch.any(torch.ge(probs, 1)):
                raise Exception('prob greater or equal to 1')
            elif torch.any(torch.lt(probs, 0)):
                raise Exception('probs lesser or equal to 0')
            _oracle_prob = oracle_tsm.repeat([probs.size(0), 1, 1])
            sim_match_loss = bceloss(probs, _oracle_prob)
            sim_match_loss_cum += sim_match_loss.detach().cpu().numpy()
            vector_norm_ratio_cum += feature_transformer.vector_norm_ratio
            loss = sim_match_loss
            feature_transformer.optimizer.zero_grad()
            projector.optimizer.zero_grad()
            dissimhead.optimizer.zero_grad()
            loss.backward()
            feature_transformer.optimizer.step()
            projector.optimizer.step()
            dissimhead.optimizer.step()
            if iter_cnt % config['visualizing_step'] == 0:
                visualize(feature_transformer, config, fig, name=f'{config["exp_name"]}_epoch{epoch}')
        #print(f'tanh(alpha): {torch.tanh(feature_transformer.alpha)}')
        #print(f'SimMatch: {sim_match_loss_cum/iter_cnt}')
        #print(f'vector_norm_ratio: {vector_norm_ratio_cum/iter_cnt}')
        
        if epoch % config['saving_epoch'] == 0:
            torch.save(feature_transformer.state_dict(), osp.join(model_save_path, f'{config["exp_name"]}_{epoch}.pt'))
            process_feature(config, epoch, process_subset=True)
            make_evaluation_db(config, load_model='process_subset')
            eval_result = linear_probing(config, load_model='process_subset')
            print(f'eval_result in epoch{epoch}: {eval_result}')
        