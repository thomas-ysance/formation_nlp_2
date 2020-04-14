from torch.nn.modules.module import _addindent
import torch


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


def print_model(model, model_type, detailed_encoder=False, detailed_classifier=True):
    print('################## Printing model ##################')
    if detailed_encoder:
        if model_type == 'camembert':
            print(model.roberta)
        elif model_type == 'roberta':
            print(model.roberta)
        elif model_type == 'bert':
            print(model.bert)
        else:
            raise Exception('Not implemented for %s' % model_type)

    if detailed_classifier:
        print(model.classifier)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('############ Total number of params // Number of trainable params')
    print(pytorch_total_params, '//', pytorch_trainable_params)

    if model_type == 'camembert':
        pytorch_total_params_enc = sum(p.numel() for p in model.roberta.parameters())
        pytorch_trainable_params_enc = sum(p.numel() for p in model.roberta.parameters() if p.requires_grad)
    elif model_type == 'roberta':
        pytorch_total_params_enc = sum(p.numel() for p in model.roberta.parameters())
        pytorch_trainable_params_enc = sum(p.numel() for p in model.roberta.parameters() if p.requires_grad)
    else:
        pytorch_total_params_enc = sum(p.numel() for p in model.bert.parameters())
        pytorch_trainable_params_enc = sum(p.numel() for p in model.bert.parameters() if p.requires_grad)

    print('### Camembert : Total number of params // Number of trainable params')
    print(pytorch_total_params_enc, '//', pytorch_trainable_params_enc)

    pytorch_total_params_clas = sum(p.numel() for p in model.classifier.parameters())
    pytorch_trainable_params_clas = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    print('### Classifier : Total number of params // Number of trainable params')
    print(pytorch_total_params_clas, '//', pytorch_trainable_params_clas)


