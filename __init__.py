import torch
import numpy as np
from torch import nn
import itertools

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
    def forward(self,*args,**kwargs):
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

        # TODO should i make sure the model is on self.device_ids[0]? Should i move it there automatically?

        ndevices = len(self.device_ids)

        # make sure args is a list of N-tuples and kwargs is a dict with N-tuples where N == ndevices
        for arg in itertools.chain(args,kwargs.values()):
            if not isinstance(arg,(list,tuple)):
                raise ValueError(f"Expected each argument to be a list or tuple not a {arg.__class__}")
            if len(arg) != ndevices:
                raise ValueError(f"Expected input of length {ndevices} and got {len(arg)}")

        # convert to lists for mutability
        args = [list(arg) for arg in args]
        kwargs = {k:list(v) for k,v in kwargs.items()}


        # transfer to gpu if requested (default yes)
        # args example: for model(x,y) it might be like ((xs,xs,xs),(ys,ys,ys))
        # so we step thru and send the first xs and first ys to the first device, etc
        if self.transfer_data:
            for i,device in enumerate(self.device_ids):
                with torch.cuda.device(device):
                    for arg in args:
                        arg[i] = recursive_cuda(arg[i])
                    for key,arg in kwargs.items():
                        kwargs[key][i] = recursive_cuda(arg[i])

        # go from ((xs,xs,xs),(ys,ys,ys)) back to ((xs,ys),(xs,ys),(xs,ys)) ie one arg list per gpu
        args_tup = list(zip(*args))
        if args_tup == []: # TODO this is a quick fix. Unclear if similar issue for kwargs by the way. Worth seeing why this happens, unclear.
            args_tup = [[]]*ndevices
        # go from dict of k:(v1,v2,v3) to tuple of dicts [{k:v1}, {k:v2}, {k:v3}]
        kwargs_tup = [{k:v[i] for k,v in kwargs.items()} for i in range(ndevices)]

        # single GPU case
        if ndevices == 1:
            assert len(args_tup) == len(kwargs_tup) == 1
            return list(zip(*[self.module(*args_tup[0], **kwargs_tup[0])]))

        # copy the network to each gpu
        replicas = self.replicate(self.module, self.device_ids)

        # run the network on each gpu in parallel
        outputs = self.parallel_apply(replicas, args_tup, kwargs_tup)

        if not isinstance(outputs[0],(list,tuple)):
            # if forward() only returns a single thing not a tuple then just return the list of that value
            return outputs

        # go from ((xs,ys),(xs,ys),(xs,ys)) back to ((xs,xs,xs),(ys,ys,ys)) 
        # but we dont do this if outputs = (1,2,3) for example, we only do it when the output is a tuple/list
        outputs = [list(x) for x in list(zip(*outputs))]
        return outputs

    def __getattr__(self, key):
        """
        Forward attribute requests to inner module
        """
        try:
            return super().__getattr__(key)
        except AttributeError:
            pass # pass so that error messages become more readable if the second getattr fails
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
    else:
        return val
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
    """
    This decorator makes the collate_fn it wraps work with tupleparallel.
    Step by step example of what's happening:
        Lets say getitem returns x,y where x and y are tensors, and we're using the default collate function (tupleparallel.default_collate, which is just tp_collate decorating torch.utils.data.dataloader.default_collate).
        wrapper() is the collate_fn according to DataLoader so it gets called with a getitem list like (batch size 5):
            [(x,y),(x,y),(x,y),(x,y),(x,y)]
        This gets split using np.array_split to yield a tuple of 3 lists (we have 3 GPUs):
            ([(x,y),(x,y)], [(x,y),(x,y)], [(x,y)])
        We then call the underlying collate_fn, in this case dataloader.default_collate, on each of the two sublists, yielding:
            ((xs,ys),(xs,ys),(xs,ys)) # 3 lists each length 2
        We then use the list(zip(*input)) idiom, which can be thought of like a sort of transpose on nested lists:
            ((xs,xs,xs),(ys,ys,ys)) # 2 lists each length 3

        Thus without this wrapper collate function would return:
            (xs,ys)
        And with this wrapper collate function returns:
            ((xs,xs,xs),(ys,ys,ys))
    """
    assert isinstance(ndevices,int)
    def decorator(collate_fn):
        assert callable(collate_fn)
        def wrapper(getitem_list):
            # turn list -> ndarray(dtype=object) so we can do array_split() to approximately
            # evenly split the list into sublists, then we do .tolist() to turn the sublists
            # from ndarrays(dtype=object) back into lists.
            as_ndarray = np.array(getitem_list,dtype=np.object)
            sublists = [sublist.tolist() for sublist in np.array_split(as_ndarray,ndevices)]
            # call collate_fn on each sublist
            collate_results = [collate_fn(sublist) for sublist in sublists]
            if isinstance(collate_results[0],dict):
                # go from [{k:v1}, {k:v2}, {k:v3}] to {k:(v1,v2,v3)}
                #assert all([all([key in c_res for c_res in collate_results]) for key in set(sum([list(c_res.keys()) for c_res in collate_results]))])
                return {k:[cres[k] for cres in collate_results] for k in collate_results[0]}

            if not isinstance(collate_results[0],(list,tuple)):
                # if getitem just returns a single thing `x` then we will returns (xs,xs) for a 2-gpu output rather than something like ((xs,xs),(ys,ys)) notice the latter has an outer tuple layer. This gets destructured during `for x,y in loader` whereas it is not wanted in `for x in loader` since if it was like [(xs,xs)] it would need to be written as `for (x,) in loader`.
                # Note that both cases will get wrapped back into args=[(xs,xs)] and args=[(xs,xs),(ys,ys)] when passed as *args or **kwargs to TupleParallel's forward().
                return collate_results

            # go from ((xs,ys),(xs,ys),(xs,ys)) -> ((xs,xs,xs),(ys,ys,ys))
            result = list(zip(*collate_results))
            result = [list(res) for res in result] # so it's mutable
            return result
        return wrapper
    return decorator

class Batch:
    """
    Just a helpful class you can subclass that will make each attr you assign get .cuda()'d automatically
    """
    def cuda(self,device=None,**kwargs):
        for k,v in vars(self).items():
            setattr(self,k,recursive_cuda(v,device,**kwargs))
        return self

def tuple_fn(fn):
    """
    A decorator that takes a function that takes inputs like foo(x,y) and converts them to take a tuple of inputs
    works with args and kwargs alike
    """
    assert callable(fn)
    def wrapper(*args,**kwargs):
        ndevices = None
        for arg in itertools.chain(args,kwargs.values()):
            if ndevices is None:
                ndevices = len(arg)
            if not isinstance(arg,(list,tuple)):
                raise ValueError(f"Expected each argument to be a list or tuple not a {arg.__class__}")
            if len(arg) != ndevices:
                raise ValueError(f"Expected input tuples to all be the same length")

        # go from ((xs,xs,xs),(ys,ys,ys)) back to ((xs,ys),(xs,ys),(xs,ys)) ie one arg list per gpu
        args_tup = list(zip(*args))
        # go from dict of k:(v1,v2,v3) to tuple of dicts [{k:v1}, {k:v2}, {k:v3}]
        kwargs_tup = [{k:v[i] for k,v in kwargs.items()} for i in range(ndevices)]

        assert len(args_tup) == len(kwargs_tup)
        results = [fn(*args_in,**kwargs_in) for args_in,kwargs_in in zip(args_tup,kwargs_tup)]

        if not isinstance(results[0],(tuple,list)):
            return results

        return [list(x) for x in list(zip(*results))]
    return wrapper
#
#def varwise_to_gpuwise(*,args=None,kwargs=None):
#
#def gpuwise_to_varwise(*,tuple=None,dict=None):
#    res = []
#    if tuple is not None:
#        # go from ((xs,xs,xs),(ys,ys,ys)) back to ((xs,ys),(xs,ys),(xs,ys)) ie one arg list per gpu
#        res.append(list(zip(*tuple)))
#    if dict is not None:
#        # go from dict of k:(v1,v2,v3) to tuple of dicts [{k:v1}, {k:v2}, {k:v3}]
#        res.append([{k:v[i] for k,v in dict.items()} for i in range(len(dict.values()[0]))])
#    if len(res) == 2:
#        return res
#    return res[0]
#
#    kwargs = {k:[cres[k] for cres in collate_results] for k in collate_results[0]}

# a version of default_collate that works with ndevices
def default_collate(ndevices):
    assert isinstance(ndevices,int)
    return tp_collate(ndevices)(torch.utils.data.dataloader.default_collate)

class Tuplewise:
    def __init__(self,tup):
        assert isinstance(tup,(list,tuple))
        self.tup = tup
    def __getattribute__(self,k):
        if k == 'tup':
            return object.__getattribute__(self,'tup')
        return [object.__getattribute__(v,k) for v in self.tup]
    def __getitem__(self,k):
        return [v[k] for v in self.tup]


def tuplewise(tup):
    return Tuplewise(tup)

