import torch
import numpy as np
from torch import nn

#model = Model()
#model = TupleParallel(model)
#model(data)

class TupleParallel(nn.DataParallel):
    def __init__(self,*args,transfer_data=True,**kwargs):
        """
        transfer_data: Use this if your data is not on the correct devices already.
            Does a nonblocking (if pinned memory) transfer of data to each gpu (small speedup).
        """
        super().__init__(*args,**kwargs)
        self.transfer_data = transfer_data
    def forward(self,input_tuple,cuda_kwargs={},**kwargs_shared):
        """
        IMPORTANT:
            -`input_tuple` must be a tuple of length equal to the number of
                GPUS TupleParallel was initialized with. Each element of
                the tuple should be a minibatch that would normally be sent
                as input to whatever module TupleParallel is wrapping.
            -whatever kwargs you pass into this function will be SHARED by each device
            -Note on minibatch sizes: use a batch of size batch_size/num_gpus if
                you want results equivalent to running a batch on a single gpu of size
                batch_size).
            -Returns results in tuple form just like input. The results will be on
                their respective devices.

        Conveniences:
            -Attribute references will be forwarded to the wrapped module so you can do
                tuplemodule.x and it will forward it to whatever module tuplemodule is
                wrapped around.
            -timers are built in. print(tuple_parallel.

        Implementation:
            -Transfer data to proper devices
            -If single GPU, just run the wrapped module as normal
            -If multiple GPUs, use DataParallel.replicate() to copy the model to each device
            -Use DataParallel.parallel_apply() (or a custom version of it) to
                spawn one CPU thread for each GPU to run the underlying module.
            -return results as a tuple

        Timing notes: 1% spent on each transfer, 9% spent on replicate, 89% spent
            on parallel_apply. Depends on model size and computation complexity of course.
        """
        assert isinstance(input_tuple,(list,tuple))
        input_tuple = list(input_tuple)

        # TODO should i make sure the model is on self.device_ids[0]? Should i move it there automatically?

        if len(input_tuple) != len(self.device_ids):
            raise ValueError(f"Expected input of length {len(self.device_ids)} and got {len(input_tuple)}")

        # transfer to gpu if requested (default yes)
        if self.transfer_data:
            for i in range(len(input_tuple)):
                with torch.cuda.device(self.device_ids[i]):
                    input_tuple[i] = recursive_cuda(input_tuple[i],self.device_ids[i],**cuda_kwargs)

        # single GPU case
        if len(self.device_ids) == 1:
            input = input_tuple[0]
            if not isinstance(input, (list, tuple)):
                    input = (input,)
            return [self.module(*input, **kwargs_shared)]

        replicas = self.replicate(self.module, self.device_ids)
        outputs = self.parallel_apply(replicas, input_tuple, [kwargs_shared]*len(input_tuple))

        return outputs

    def __getattr__(self, key):
        """
        Forward attribute requests to inner module
        """
        try:
            return super().__getattr__(key)
        except AttributeError:
            pass
        return getattr(self.module, key)

#def to_gpu(val,dev,non_blocking=True):
#    """
#    Calls val.cuda(non_blocking=True) or val.to(dev,non_blocking=True) and
#        recurses on lists/tuples
#    """
#    if type(val) == list:
#        return [to_gpu(item,dev,non_blocking=non_blocking) for item in val]
#    if type(val) == tuple:
#        return tuple([to_gpu(item,dev,non_blocking=non_blocking) for item in val])
#    if hasattr(val,'cuda'):
#        try:
#            return val.cuda(non_blocking=non_blocking)
#        except TypeError:
#            raise TypeError(f"{val.__class__} .cuda() implementation must allow the call .cuda(non_blocking=True)")
#    if hasattr(val,'to'):
#        try:
#            return val.to(dev,non_blocking=non_blocking)
#        except TypeError:
#            raise TypeError(f"{val.__class__} .to() implementation must allow the call .to(device,non_blocking=True)")
#    raise TypeError(f"Type {val.__class__} must implement a .to(device, non_blocking=False) or .cuda() function for your type")

def recursive_cuda(val,device=None,**kwargs):
    """
    Calls val.cuda(device,**kwargs) and recurses on lists/tuples
    If device is None we default to the current device
    We call val.to(device,**kwargs) if a `cuda` function is not found.
    If a `to` function is also not found then we just return `val` without doing anything to it.
    """
    if device is None:
        device = torch.cuda.current_device()
    if type(val) == list:
        return [recursive_cuda(item,device,**kwargs) for item in val]
    elif type(val) == tuple:
        return tuple([recursive_cuda(item,device,**kwargs) for item in val])
    elif hasattr(val,'cuda'):
        result = val.cuda(device,**kwargs)
    elif hasattr(val,'to'):
        result = val.to(device,**kwargs)
    assert result is not None, "Your .to() or .cuda() implementation should return `self` at the end."
    return result

#def class_recursive_cuda(val,device=None,deep=False,**kwargs):
#    """
#    Like recursive_cuda but recurses into all fields of classes if the overall class didn't have a .cuda or .to method
#    If deep is True then this will recurse into classes within fields!
#    """
#    if isinstance(val,(list,tuple)):
#        return recui
#        for k,v in vars(self).items():
#            setattr(self,k,recursive_cuda(v,device,**kwargs)
#        return self


# a decorator that takes an argument
def tp_collate(ndevices):
    assert isinstance(ndevices,int)
    def decorator(collate_fn):
        assert callable(collate_fn)
        def wrapper(getitem_list):
            as_ndarray = np.array(getitem_list,dtype=np.object)
            sublists = [sublist.tolist() for sublist in np.array_split(as_ndarray,ndevices)]
            return [collate_fn(sublist) for sublist in sublists]
        return wrapper
    return decorator

class Batch:
    def cuda(self,device=None,**kwargs):
        for k,v in vars(self).items():
            setattr(self,k,recursive_cuda(v,device,**kwargs))
        return self


# a version of default_collate that works with ndevices
def default_collate(ndevices):
    assert isinstance(ndevices,int)
    return tp_collate(ndevices)(torch.utils.data.dataloader.default_collate)


