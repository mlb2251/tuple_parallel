import torch
import numpy as np
from torch import nn
import itertools

def cat(returns):
    """
    return cat([x,y,z]) where x,y,z are each tuples of tensors will yield x,y,z where each is a tensor that results from cating
    """
    if not isinstance(returns[0],(list,tuple)):
        # e.g. if the function just returned a single value then returns is a tuple of that value
        return torch.cat(returns)

    result = []
    for ret in returns:
        if len(ret) > 0 and isinstance(ret[0],torch.Tensor):
            result.append(torch.cat(ret))
        else:
            result.append(ret)
    if len(result) == 1:
        result = result[0]
    return result

class TupleParallel(nn.DataParallel):
    def __init__(self,module,device_ids,transfer_output=True,transfer_input=True,reduction=cat):
        """
        `device_ids`: List of device ids (ints) for gpus to use. The length of this is the number of gpus that we parallelize over.
        `transfer_output`: If True, all outputs will be sent to the home_device (device_ids[0]) at the end.
        `reduction`: a callable, and if provided this will be called on the output of TupleParallel's forward right before returning.
            tuple_parallel.cat is the default. None means no reduction.
        `transfer_input`: Use this if your data is not on the correct devices already. Transfers data to proper gpus at start of forward().
        TupleParallel will automatically move `module` to the home device, which is device_ids[0]
        """
        nn.Module.__init__(self) # don't do DataParallel's __init__, we only subclass it to steal its functions
        assert all([isinstance(dev,int) for dev in device_ids])
        self.module = module
        self.device_ids = device_ids
        self.ndevices = len(device_ids)
        self.home_device = device_ids[0]
        self.transfer_output = transfer_output
        self.transfer_input = transfer_input
        self.reduction = reduction if (reduction is not None) else (lambda x:x)

        #torch.nn.parallel.data_parallel._check_balance(self.device_ids)
        self.module = self.module.cuda(self.home_device)


    def forward(self,*args,**kwargs):
        """
        Conveniences:
            -Attribute references will be forwarded to the wrapped module so you can do
                tuplemodule.x and it will forward it to whatever module tuplemodule is
                wrapped around.

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

        # make sure args is a list of N-tuples and kwargs is a dict with N-tuples where N == ndevices
        for arg in itertools.chain(args,kwargs.values()):
            if not isinstance(arg,(list,tuple)):
                raise ValueError(f"Expected each argument to be a list or tuple not a {arg.__class__}")
            if len(arg) != self.ndevices:
                raise ValueError(f"Expected input of length {self.ndevices} and got {len(arg)}")

        # convert to lists for mutability
        args = [list(arg) for arg in args]
        kwargs = {k:list(v) for k,v in kwargs.items()}

        # transfer to gpu if requested (default yes)
        # args example: for model(x,y) it might be like ((xs,xs,xs),(ys,ys,ys))
        # so we step thru and send the first xs and first ys to the first device, etc
        if self.transfer_input:
            for i,device in enumerate(self.device_ids):
                with torch.cuda.device(device):
                    for arg in args:
                        arg[i] = recursive_cuda(arg[i])
                    for key,arg in kwargs.items():
                        kwargs[key][i] = recursive_cuda(arg[i])

        # go from ((xs,xs,xs),(ys,ys,ys)) back to ((xs,ys),(xs,ys),(xs,ys)) ie one arg list per gpu
        args_tup = list(zip(*args))

        # go from dict of k:(v1,v2,v3) to tuple of dicts [{k:v1}, {k:v2}, {k:v3}]
        kwargs_tup = [{k:v[i] for k,v in kwargs.items()} for i in range(self.ndevices)]

        if args_tup == []: # TODO this is a quick fix. Unclear if similar issue for kwargs by the way. Worth seeing why this happens, unclear. Also maybe this isn't a real problem idk it came up once though in a case with no args as input only kwargs
            args_tup = [[]]*self.ndevices

        assert len(args_tup) == len(kwargs_tup)

        # single GPU case
        if self.ndevices == 1:
            return self.reduction(list(zip(*[self.module(*args_tup[0], **kwargs_tup[0])])))

        # copy the network to each gpu
        replicas = self.replicate(self.module, self.device_ids)

        # run the network on each gpu in parallel
        outputs = self.parallel_apply(replicas, args_tup, kwargs_tup)

        if self.transfer_output is True:
            with torch.cuda.device(self.home_device):
                outputs = recursive_cuda(outputs)

        if not isinstance(outputs[0],(list,tuple)):
            # if forward() only returns a single thing not a tuple then just return the list of that value
            return self.reduction(outputs)

        # go from ((xs,ys),(xs,ys),(xs,ys)) back to ((xs,xs,xs),(ys,ys,ys)) 
        # but we dont do this if outputs = (1,2,3) for example, we only do it when the output is a tuple/list
        outputs = [list(x) for x in list(zip(*outputs))]
        return self.reduction(outputs)

    def __getattr__(self, key):
        """
        Forward attribute requests to inner module
        """
        try:
            return super().__getattr__(key)
        except AttributeError:
            pass # pass so that error messages become more readable if the second getattr fails
        return getattr(self.module, key)

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
    elif type(val) == dict:
        return {k:recursive_cuda(item,device,**kwargs) for k,item in val.items()}
    elif hasattr(val,'cuda'):
        result = val.cuda(device,**kwargs)
    elif hasattr(val,'to'):
        result = val.to(device,**kwargs)
    else:
        return val
    assert result is not None, "Your .to() or .cuda() implementation should return `self` at the end."
    return result

def cuda_attrs(obj,*args,**kwargs):
    for k,v in vars(obj).items():
        setattr(obj,k,recursive_cuda(v,*args,**kwargs))
    return obj

class Batch:
    """
    Just a helpful class you can subclass that will make each attr you assign get .cuda()'d automatically
    """
    def cuda(self,*args,**kwargs):
        return cuda_attrs(self,*args,**kwargs)


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

# a version of default_collate that works with ndevices
def default_collate(ndevices):
    assert isinstance(ndevices,int)
    return tp_collate(ndevices)(torch.utils.data.dataloader.default_collate)

