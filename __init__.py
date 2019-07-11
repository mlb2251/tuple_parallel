import torch
from torch import nn
import time
import threading
#from mlb_py.util import breaker


class TupleParallel(nn.DataParallel):
    def __init__(self,*args,transfer_data=True,**kwargs):
        """
        transfer_data: Use this if your data is not on the correct devices already. Does a nonblocking (if pinned memory) transfer of data to each gpu (small speedup).
        """
        super().__init__(*args,**kwargs)
        self.transfer_data = transfer_data
    def forward(self,input_tuple,**kwargs_shared):
        """
        IMPORTANT:
            -`input_tuple` must be a tuple of length equal to the number of GPUS TupleParallel was initialized with. Each element of the tuple should be a minibatch that would normally be sent as input to whatever module TupleParallel is wrapping.
            -whatever kwargs you pass into this function will be SHARED by each device
            -Note on minibatch sizes: use a batch of size batch_size/num_gpus if you want results equivalent to running a batch on a single gpu of size batch_size).
            -Returns results in tuple form just like input. The results will be on their respective devices.

        Conveniences:
            -Attribute references will be forwarded to the wrapped module so you can do tuplemodule.x and it will forward it to whatever module tuplemodule is wrapped around.
            -timers are built in. print(tuple_parallel.

        Implementation:
            -Transfer data to proper devices
            -If single GPU, just run the wrapped module as normal
            -If multiple GPUs, use DataParallel.replicate() to copy the model to each device
            -Use DataParallel.parallel_apply() (or a custom version of it) to spawn one CPU thread for each GPU to run the underlying module.
            -return results as a tuple

        Timing notes: 1% spent on each transfer, 9% spent on replicate, 89% spent on parallel_apply. Depends on model size and computation complexity of course.
        """

        # TODO should i make sure the model is on self.device_ids[0]? Should i move it there automatically?

        if len(input_tuple) != len(self.device_ids):
            raise ValueError(f"Expected input of length {len(self.device_ids)} and got {len(input_tuple)}")

        if self.parallel_transfer:
            # transfer to gpu nonblocking
            for i in range(len(input_tuple)):
                with torch.cuda.device(self.device_ids[i]):
                    input_tuple[i] = to_gpu(input_tuple[i],self.device_ids[i])
        else:
            if not all([input.device.index==d for input,d in zip(input_tuple,self.device_ids)]):
                raise ValueError(f"At least one input is not on the correct device. Improper inputs at tuple indices: {[i for i,(input,d) in enumerate(zip(input_tuple,self.device_ids)) if input.device.index != d]}")

        # single GPU case
        if len(self.device_ids) == 1:
            return [self.module(input_tuple[0], **kwargs_shared)]

        replicas = self.replicate(self.module, self.device_ids)
        outputs = self.parallel_apply(replicas, input_tuple, [kwargs_shared for _ in input_tuple])

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

def to_gpu(val,dev):
    """
    Calls val.cuda(non_blocking=True) or val.to(dev,non_blocking=True) and recurses on lists/tuples
    """
    if type(val) == list:
        return [to_gpu(item,dev) for item in val]
    if type(val) == tuple:
        return tuple([to_gpu(item,dev) for item in val])
    if hasattr(val,'cuda'):
        try:
            return val.cuda(non_blocking=True)
        except TypeError:
            raise TypeError(f"{val.__class__} .cuda() implementation must allow the call .cuda(non_blocking=True)")
    if hasattr(val,'to'):
        try:
            return val.to(dev,non_blocking=True)
        except TypeError:
            raise TypeError(f"{val.__class__} .to() implementation must allow the call .to(device,non_blocking=True)")
    raise TypeError(f"Type {val.__class__} must implement a .to(device, non_blocking=False) or .cuda() function for your type")

