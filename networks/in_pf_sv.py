import torch
from utils.nn.model.IN import INTagger


def get_model(data_config, **kwargs):

    pf_dims = data_config.input_shapes['pf_features'][-1]
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_dims = data_config.input_shapes['sv_features'][-1]
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = len(data_config.label_value)
    print(pf_dims, sv_dims, num_classes, pf_features_dims, sv_features_dims)
    model = INTagger(pf_dims, sv_dims, num_classes, pf_features_dims, sv_features_dims,
                     hidden=60,De=20,Do=24)
    
    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss(reduction='mean')
