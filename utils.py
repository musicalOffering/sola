import torch
import os
import os.path as osp
import numpy as np
import pickle
import random

from network import FeatureTransformer, LinearClassifier
from dataset import VideoEvaluation
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

def process_feature(config, load_model, process_subset=True):
    model_path = osp.join(config["model_save_path"], config["exp_name"], f'{config["exp_name"]}_{load_model}.pt')
    if process_subset:
        feature_save_path = osp.join(config["feature_save_path"], f'{config["exp_name"]}_val')
    else:
        feature_save_path = osp.join(config["feature_save_path"], f'{config["exp_name"]}_{load_model}')
    if not os.path.isdir(feature_save_path):
        os.mkdir(feature_save_path)
    if load_model == 0:
        print('model is set to IDENTITY!')
        feature_transformer = torch.nn.Identity()
    else:
        feature_transformer = FeatureTransformer(config)
        feature_transformer.load_state_dict(torch.load(model_path))

    #print('processing training...')    
    if not os.path.isdir(osp.join(feature_save_path, 'training')):
        os.mkdir(osp.join(feature_save_path, 'training'))
    data_path = osp.join(config["data_root"], "training")
    if process_subset:
        with open(osp.join(config['data_root'], 'training_subset_list.pkl'), 'rb') as f:
            vid_list = pickle.load(f)
        for vid_name in vid_list:
            full_path = os.path.join(data_path, f'{vid_name}.npy')
            feature = torch.from_numpy(np.load(full_path).astype(np.float32)).unsqueeze(0).cuda()
            with torch.no_grad():
                o = feature_transformer(feature).squeeze()
                o = o.cpu().numpy()
            np.save(osp.join(feature_save_path, 'training', f'{vid_name}.npy'), o)
    else:
        for file_name in tqdm(os.listdir(data_path)):
            full_path = os.path.join(data_path, file_name)
            feature = torch.from_numpy(np.load(full_path).astype(np.float32)).unsqueeze(0).cuda()
            with torch.no_grad():
                o = feature_transformer(feature).squeeze()
                o = o.cpu().numpy()
            np.save(osp.join(feature_save_path, 'training', file_name), o)

    #print('processing validation...')    
    if not os.path.isdir(osp.join(feature_save_path, 'validation')):
        os.mkdir(osp.join(feature_save_path, 'validation'))
    data_path = osp.join(config["data_root"], "validation")
    if process_subset:
        with open(osp.join(config['data_root'], 'validation_subset_list.pkl'), 'rb') as f:
            vid_list = pickle.load(f)
        for vid_name in vid_list:
            full_path = os.path.join(data_path, f'{vid_name}.npy')
            feature = torch.from_numpy(np.load(full_path).astype(np.float32)).unsqueeze(0).cuda()
            with torch.no_grad():
                o = feature_transformer(feature).squeeze()
                o = o.cpu().numpy()
            np.save(osp.join(feature_save_path, 'validation', f'{vid_name}.npy'), o)
    else:
        for file_name in tqdm(os.listdir(data_path)):
            full_path = os.path.join(data_path, file_name)
            feature = torch.from_numpy(np.load(full_path).astype(np.float32)).unsqueeze(0).cuda()
            with torch.no_grad():
                o = feature_transformer(feature).squeeze()
                o = o.cpu().numpy()
            np.save(osp.join(feature_save_path, 'validation', file_name), o)
    #print('processing end')

def make_evaluation_db(config, load_model):
    UNIT = 100
    print(f'EXPERIENCE_NAME: {config["exp_name"]}, making expert DB...')
    if load_model == 'process_subset':
        process_subset = True
    else:
        process_subset = False
    print(f'process_subset: {process_subset}')
    label_base = config["label_base"]
    if process_subset:
        feature_base = osp.join(config['feature_save_path'], f'{config["exp_name"]}_val')
    else:
        feature_base = osp.join(config['feature_save_path'], f'{config["exp_name"]}_{load_model}')
    if not osp.exists(osp.join(feature_base, 'training_0')):
        os.mkdir(osp.join(feature_base, 'training_0'))
    if not osp.exists(osp.join(feature_base, 'validation_0')):
        os.mkdir(osp.join(feature_base, 'validation_0'))
    if not osp.exists(osp.join(feature_base, 'training_1')):
        os.mkdir(osp.join(feature_base, 'training_1'))
    if not osp.exists(osp.join(feature_base, 'validation_1')):
        os.mkdir(osp.join(feature_base, 'validation_1'))

    #print('process training...')
    if process_subset:
        with open(osp.join(config['data_root'], 'training_subset_list.pkl'), 'rb') as f:
            vid_list = pickle.load(f)
        file_list = [f'{vid_name}.npy' for vid_name in vid_list]
    else:
        file_list = os.listdir(osp.join(label_base, 'training_label'))
    random.shuffle(file_list)
    positives = None
    pos_cnt = 0
    negatives = None
    neg_cnt = 0
    for file_name in file_list:
        feature_path = os.path.join(feature_base, 'training', file_name)
        label_path = os.path.join(label_base, 'training_label', file_name)
        feature = np.load(feature_path).astype(np.float32)
        label = np.load(label_path)
        assert len(feature) == len(label), f'{len(feature)}, {len(label)}'
        label_idx = np.where(label == 1, True, False)
        action_features = feature[label_idx]
        non_action_features = feature[~label_idx]
        if positives is None:
            positives = action_features
        else:
            positives = np.concatenate([positives, action_features], axis=0)
        if negatives is None:
            negatives = non_action_features
        else:
            negatives = np.concatenate([negatives, non_action_features], axis=0)
        while len(positives) > UNIT:
            saving_positives = positives[:UNIT]
            positives = positives[UNIT:]
            np.save(os.path.join(feature_base, 'training_1', f'{pos_cnt}.npy'), saving_positives)
            pos_cnt += 1
        while len(negatives) > UNIT:
            saving_negatives = negatives[:UNIT]
            negatives = negatives[UNIT:]
            np.save(os.path.join(feature_base, 'training_0', f'{neg_cnt}.npy'), saving_negatives)
            neg_cnt += 1

    #print('process validation...')
    if process_subset:
        with open(osp.join(config['data_root'], 'validation_subset_list.pkl'), 'rb') as f:
            vid_list = pickle.load(f)
        file_list = [f'{vid_name}.npy' for vid_name in vid_list]
    else:
        file_list = os.listdir(osp.join(label_base, 'validation_label'))
    random.shuffle(file_list)
    positives = None
    pos_cnt = 0
    negatives = None
    neg_cnt = 0
    for file_name in file_list:
        feature_path = os.path.join(feature_base, 'validation', file_name)
        label_path = os.path.join(label_base, 'validation_label', file_name)
        feature = np.load(feature_path).astype(np.float32)
        label = np.load(label_path)
        assert len(feature) == len(label), f'{len(feature)}, {len(label)}'
        label_idx = np.where(label == 1, True, False)
        action_features = feature[label_idx]
        non_action_features = feature[~label_idx]
        if positives is None:
            positives = action_features
        else:
            positives = np.concatenate([positives, action_features], axis=0)
        if negatives is None:
            negatives = non_action_features
        else:
            negatives = np.concatenate([negatives, non_action_features], axis=0)
        while len(positives) > UNIT:
            saving_positives = positives[:UNIT]
            positives = positives[UNIT:]
            np.save(os.path.join(feature_base, 'validation_1', f'{pos_cnt}.npy'), saving_positives)
            pos_cnt += 1
        while len(negatives) > UNIT:
            saving_negatives = negatives[:UNIT]
            negatives = negatives[UNIT:]
            np.save(os.path.join(feature_base, 'validation_0', f'{neg_cnt}.npy'), saving_negatives)
            neg_cnt += 1


def linear_probing(config, load_model):
    '''
    if load_model = 'process_subset', it is in validation mode
    '''
    trainset = VideoEvaluation(subset='training', config=config, load_model=load_model)
    train_loader = DataLoader(trainset, batch_size=config['eval_batch_size'], shuffle=True, num_workers=0)
    valset = VideoEvaluation(subset='validation', config=config, load_model=load_model)
    val_loader = DataLoader(valset, batch_size=config['eval_batch_size'], shuffle=True, num_workers=0)
    linear_classifier = LinearClassifier(config)
    criterion = torch.nn.CrossEntropyLoss()
    eval_results = []
    
    for epoch in range(config['eval_epoch']):
        for data, label in train_loader:
            data = data.cuda()
            label = label.cuda()
            pred = linear_classifier(data)
            loss = criterion(pred, label)
            linear_classifier.optimizer.zero_grad()
            loss.backward()
            linear_classifier.optimizer.step()
        answer_cnt = 0
        prob_cnt = 0
        for data, label in val_loader:
            data = data.cuda()
            label = label.cuda()
            with torch.no_grad():
                pred = linear_classifier(data)
                pred = torch.argmax(pred, dim=1)
                answer_cnt += torch.sum(torch.where(pred == label, 1, 0)).detach().cpu().numpy().item()
                prob_cnt += len(data)
        eval_results.append(answer_cnt/prob_cnt)
    return sum(eval_results[5:15])/10