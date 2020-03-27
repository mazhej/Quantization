
from torch.quantization.fuse_modules import fuse_conv_bn, fuse_conv_bn_relu
from torch import nn
import torch.nn.intrinsic.modules.fused as torch_fused

def fuse_conv_bn_identity(conv, bn, identity):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    relu = nn.ReLU(inplace=False)   # added by Mehrdad
    assert(conv.training == bn.training == relu.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    if conv.training:
        assert not relu.inplace, 'We only support fusion of non-inplace ReLU.'
        return torch_fused.ConvBnReLU2d(conv, bn, relu)
    else:
        return torch_fused.ConvReLU2d(
            nn.utils.fusion.fuse_conv_bn_eval(conv, bn), relu)



class FuseHook():
    def __init__(self, model):
        self.modules_to_fuse = []
        self.prev_module_name = ''
        self.is_prev_conv = False
        self.is_prev_linear = False
        self.is_prev_conv_bn = False
        for name, m in model.named_modules():
            m.register_forward_hook(self.hook_fn)
            m.name = name
    
    def CheckFuse(self, module):
        if (type(module) == nn.Conv2d):
            self.is_prev_conv = True
            self.is_prev_linear = False
        elif (type(module) == nn.Linear):
            self.is_prev_linear = True
            self.is_prev_conv = False
        else:
            self.Stage2_Conv2dBN(module)
            self.Stage2_LinearConv2dReLU(module)
            self.is_prev_conv = False
            self.is_prev_linear = False

    def Stage2_Conv2dBN(self, module):
        if ( self.is_prev_conv & (type(module) == nn.BatchNorm2d) ):
            self.modules_to_fuse.append([self.prev_module_name, module.name])
            self.is_prev_conv_bn = True
        else:
            self.Stage3_Conv2dBNReLU(module)
            self.is_prev_conv_bn = False 

    def Stage2_LinearConv2dReLU(self, module):
        if ( (self.is_prev_linear | self.is_prev_conv)  & (type(module) == nn.ReLU) ):
            self.modules_to_fuse.append([self.prev_module_name, module.name])

    def Stage3_Conv2dBNReLU(self, module):
        if ( self.is_prev_conv_bn & (type(module) == nn.ReLU) ):
            self.modules_to_fuse[-1].append(module.name)
            module.inplace = False

    def hook_fn(self, module, input, output):
        self.CheckFuse(module)
        self.prev_module_name = module.name


def modified_fuse_known_modules(mod_list):
    OP_LIST_TO_FUSER_METHOD = {
        (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
        (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
        (nn.Conv2d, nn.BatchNorm2d, nn.Identity): fuse_conv_bn_identity,
        (nn.Conv2d, nn.ReLU): nn.intrinsic.ConvReLU2d,
        (nn.Linear, nn.ReLU): nn.intrinsic.LinearReLU
    }

    types = tuple(type(m) for m in mod_list)
    fuser_method = OP_LIST_TO_FUSER_METHOD.get(types, None)
    if fuser_method is None:
        raise NotImplementedError("Cannot fuse modules: {}".format(types))
    new_mod = [None] * len(mod_list)
    new_mod[0] = fuser_method(*mod_list)

    for i in range(1, len(mod_list)):
        new_mod[i] = nn.Identity()
        new_mod[i].training = mod_list[0].training

    return new_mod