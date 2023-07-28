import torch.nn as nn 

def find_layers_for_8bit(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    
    if module.__class__.__base__ in layers:
        ## 8bitの場合、moduleがLinear8bitLtなので、継承元のnn.Linearかどうかで判定する
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers_for_8bit(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity_for_8bit(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        print(layer)
        subset = find_layers_for_8bit(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
        if sub_params == 0:
            print('params is zero...', sub_count, sub_params)            
        else:
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    if total_params == 0:
        print('total params zero....', count, total_params)
        return 0    
    return float(count)/total_params 