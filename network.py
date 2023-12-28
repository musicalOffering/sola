import torch.nn as nn
import torch

class FeatureTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        if config['projection'] == 'identity':
            self.projection = nn.Sequential(
                nn.Identity(),
            )
        else:
            raise NotImplementedError()
        self.conv_network = nn.Sequential(
            nn.Conv1d(
                config['out_feature_dim'], config['network_hidden_unit'], kernel_size=config['kernel_size1'], 
                padding=int((1/2)*config['dilation1']*(config['kernel_size1']-1)), dilation=config['dilation1']
            ),
            nn.ReLU(),
            nn.Conv1d(
                config['network_hidden_unit'], config['network_hidden_unit'], kernel_size=config['kernel_size2'], 
                padding=int((1/2)*config['dilation2']*(config['kernel_size2']-1)), dilation=config['dilation2']
            ),
            nn.ReLU(),
            nn.Conv1d(config['network_hidden_unit'], config['out_feature_dim'], kernel_size=1),
        )
        self.alpha = nn.parameter.Parameter(torch.FloatTensor([0]))
        self.vector_norm_ratio = 1
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config['learning_rate'])
        self.cuda()

    def update_ratio(self, original, modulation):
        modulation = torch.mean(torch.sqrt(torch.sum(modulation**2, dim=1))).detach().cpu().numpy()
        original = torch.mean(torch.sqrt(torch.sum(original**2, dim=1))).detach().cpu().numpy()
        self.vector_norm_ratio = modulation/(original + 1e-8)
        

    def forward(self, x):
        #IN: [N, L, C]
        x = self.projection(x)
        x = x.permute(0,2,1)
        non_linear = self.conv_network(x)
        alpha = torch.tanh(self.alpha)
        self.update_ratio(x, alpha*non_linear)
        x = alpha*non_linear + x
        x = x.permute(0,2,1)
        return x

class MLPHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config['mlp_layer_num'] == 0:
            self.network = nn.Identity()
            #dummy layer to avoid error
            self.tmp = nn.Linear(1,1)
        elif config['mlp_layer_num'] == 1:
            self.network = nn.Linear(config['out_feature_dim'], config['out_feature_dim'])
        elif config['mlp_layer_num'] == 2:
                self.network = nn.Sequential(
                nn.Linear(config['out_feature_dim'], config['mlphead_hidden_unit']),
                nn.ReLU(),
                nn.Linear(config['mlphead_hidden_unit'], config['out_feature_dim'])
            )
        elif config['mlp_layer_num'] == 3:
            self.network = nn.Sequential(
                nn.Linear(config['out_feature_dim'], config['mlphead_hidden_unit']),
                nn.ReLU(),
                nn.Linear(config['mlphead_hidden_unit'], config['mlphead_hidden_unit']),
                nn.ReLU(),
                nn.Linear(config['mlphead_hidden_unit'], config['out_feature_dim'])
            )
        else:
            raise NotImplementedError()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config['learning_rate'])
        self.config = config
        self.cuda()

    def forward(self, x):
        #IN: [N, L, C]
        x =  self.network(x)
        x = nn.functional.normalize(x, dim=-1)
        return x

class Projector(nn.Module):
    def __init__(self, config, head='sim') -> None:
        super().__init__()
        if head == 'sim':
            self.network = nn.Sequential(
                nn.Linear(config['out_feature_dim'], config['sim_proj_hidden_unit']),
                nn.ReLU(),
                nn.Linear(config['sim_proj_hidden_unit'], config['sim_proj_out'])
            )
        elif head == 'dissim':
            self.network = nn.Sequential(
                nn.Linear(config['out_feature_dim'], config['dissim_proj_hidden_unit']),
                nn.ReLU(),
                nn.Linear(config['dissim_proj_hidden_unit'], config['dissim_proj_out'])
            )
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=config['learning_rate'])
        self.cuda()

    def forward(self, x):
        #IN: [N, L, C]
        x =  self.network(x)
        return x
    

class LinearClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = nn.Linear(config['out_feature_dim'], 2)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=config['eval_lr'], momentum=config['eval_momentum'])
        self.cuda()

    def forward(self, x):
        #IN: [N,C]
        x = nn.functional.normalize(x, dim=-1)
        x = self.network(x)
        return x