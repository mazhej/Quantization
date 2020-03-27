
from torch.quantization.fuse_modules import fuse_conv_bn, fuse_conv_bn_relu
from torch import nn
import torch.nn.intrinsic.modules.fused as torch_fused
from torch.nn import Conv2d, Conv3d, ReLU, Linear, BatchNorm2d, LeakyReLU
import torch
import torch.utils.data
from torch import nn

class ConvBnLeakyRelu(torch.nn.Sequential): #added by Maz
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and LeakyReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, bn, leakyrelu):
        assert type(conv) == Conv2d and type(bn) == BatchNorm2d and \
            type(leakyrelu) == LeakyReLU, 'Incorrect types for input modules{}{}{}' \
            .format(type(conv), type(bn), type(leakyrelu))
        super(ConvBnLeakyRelu, self).__init__(conv, bn, leakyrelu)

class ConvLeakyRelu_eval(torch.nn.Sequential): #added by Maz
    r"""This is a sequential container which calls the Conv 2d and LeakyReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, conv, leakyrelu):
        assert type(conv) == Conv2d and type(leakyrelu) == LeakyReLU, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(leakyrelu))
        super(ConvLeakyRelu_eval, self).__init__(conv, leakyrelu)


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
            torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn), relu)

def fuse_conv_leaky(conv,bn,leakyrelu):# added by Maz
    r"""Given the conv and leakyrelu modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d
        leakyrelu:instance of type LeakyRelu needs to be fused with the conv
    """
    
    assert(conv.training == bn.training == leakyrelu.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    if conv.training:
        assert not leakyrelu.inplace, 'We only support fusion of non-inplace LeakyReLU.'
        return ConvBnLeakyRelu(conv,bn,leakyrelu)

    else:
        return ConvLeakyRelu_eval(
            torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn), leakyrelu)

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
        if ( (self.is_prev_linear | self.is_prev_conv)  and ((type(module) == nn.ReLU) | (type(module) == nn.LeakyReLU) )):
            self.modules_to_fuse.append([self.prev_module_name, module.name])

    def Stage3_Conv2dBNReLU(self, module):
        if ( self.is_prev_conv_bn & ((type(module) == nn.ReLU) | (type(module) == nn.LeakyReLU)) ):
            self.modules_to_fuse[-1].append(module.name)
            module.inplace = False

    def hook_fn(self, module, input, output):
        self.CheckFuse(module)
        self.prev_module_name = module.name

def modified_fuse_known_modules(mod_list):
    OP_LIST_TO_FUSER_METHOD = {
        (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn,
        (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU): fuse_conv_bn_relu,
        (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Identity): fuse_conv_bn_identity,
        (torch.nn.Conv2d, torch.nn.ReLU): torch.nn.intrinsic.ConvReLU2d,
        (torch.nn.Linear, torch.nn.ReLU): torch.nn.intrinsic.LinearReLU,
        (torch.nn.Conv2d, torch.nn.BatchNorm2d ,torch.nn.LeakyReLU): fuse_conv_leaky
    }

    types = tuple(type(m) for m in mod_list)
    fuser_method = OP_LIST_TO_FUSER_METHOD.get(types, None)
    if fuser_method is None:
        raise NotImplementedError("Cannot fuse modules: {}".format(types))
    new_mod = [None] * len(mod_list)
    new_mod[0] = fuser_method(*mod_list)

    for i in range(1, len(mod_list)):
        new_mod[i] = torch.nn.Identity()
        new_mod[i].training = mod_list[0].training

    return new_mod