
import torch
import torch.nn.utils.prune as prune

class KeepSparsityMethod(prune.BasePruningMethod):
   
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask[t == 0] = 0
     
        return mask

def KeepSparsity(model, name):
    
    for module in model.modules():
        if (isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)):
            KeepSparsityMethod.apply(module, 'weight')