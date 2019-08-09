# TupleParallel

### `tuple_parallel.TupleParallel`:
    test

#### `tuple_parallel.recursive_cuda(val,device=None,**kwargs)`:
    returns val.cuda(device,**kwargs), recursing on lists, tuples, and dicts. If val doesn't have a .cuda implementation then .to is used, and if .to isn't implemented then the value is returned as is.
    If no device is provided the torch.cuda.current_device() is used.

##### `tuple_paralle.tp_collate(ndevices)`:
    This is a decorator taking an int `ndevices`.
    If `collate_fn` worked with some model Model(), then tp_collate(ndevices)(collate_fn) works with TupleParallel(Model()). It's basically an easy way to make your collate function work with TupleParallel without any extra work.
    See also tuple_parallel.default_collate

### `tuple_parallel.default_collate(ndevices)`:
    This returns tp_collate(ndevices)(torch.utils.data.dataloader.default_collate)
    So basically it's just tp_collate applied to the default collate function
    If the module you're wrapping in TupleParallel didn't need a special collate function, then you want to do DataLoader(collate_fn=tuple_parallel.default_collate). This is an important step!

### `tuple_parallel.tuple_fn(fn)`:
    This is a decorator that takes a function of arbitrary args/kwargs and returns a function that takes the same arguments in tuple form. Basically it's what TupleParallel does for the forward() function, but for any function. So if youd normally do a,b = foo(x,y) you'd do (a1,a1),(b1,b2) = foo((x1,x2),(y1,y2)).
    Note that this does NOT run the inputs in parallel. That extension could be easily added as an option.

`tuple_parallel.cuda_attrs(obj,*args,**kwargs)`:
    Calls recursive_cuda() on every attr in obj and returns obj. This is a nice generic implementation for .cuda in most custom classes you might write.

`tuple_parallel.Batch`:
    This is a class that implements .cuda() to simply call tuple_parallel.cuda_fields. Meant to be subclassed though you can also just use tuple.paralle.cuda_fields yourself for the same result.

`tuple_parallel.Tuplewise`:
    This is a class that's initialized with a list/tuple. Once initialized, any getattribute or getitem calls will be applied to each item in the underlying tuple.
    For example: Tuplewise([(0,1),(2,3),(4,5)])[0] == [0,2,4]
    So effectively it's just sugar on list comprehension

`tuple_parallel.cat`:
    A common case used like: TupleParallel(transfer_output=True, reduction=tuple_parallel.cat)
    This takes a tuple of return values and calls torch.cat() on each return value if the return value is a tuple of tensors. Note it's assumed that all return values are themselves tensors

One thing anyone working with tuples should know is this:
    list(zip(*some_list)) is effectively like a matrix transpose if we consider a list of equal-length lists like a matrix. so if some_list is ((1,2),(3,4),(5,6)) this yields ((1,3,5),(2,4,6)). This operation is its own inverse. Note that if the sublists aren't all the same length things will get truncated silently with no error, so watch out for that (mlb.zip() throws an error in this case).



## Quick Summary
The `TupleParallel` module extends the `torch.nn.DataParallel` module to work with input of a much more general form. Wrapping a module in `TupleParallel` is done in the same way as DataParallel (and any extra args and kwargs are forwarded to `DataParallel.__init__`). Unlike `DataParallel`, the `forward()` function for `TupleParallel` takes a tuple of length N as input, where N is the number of GPUs that `TupleParallel` was initialized with. Each element of the tuple should be a batch that would normally be sent as input to the module that `TupleParallel` is warpping. To achieve the same results for a given batch size as running a single batch on one gpu, you can split that batch into `N` mini batches and send the tuple of minibatches to `TupleParallel`. `TupleParallel` will transfer the data to the GPUs for you by calling .`to(non_blocking=True)` on them (for custom datatypes this means you should implement `.to()`). `forward()` will return a tuple of outputs, still on their respective GPUs. See comment at the start of `forward()` for more. Also see example.py.

## Excerpts from `example.py`
```python

# Basic model initialization
from tuple_parallel import TupleParallel
DEVICES = [1,2]
model = Model() # this is the original model
model = TupleParallel(model,device_ids=DEVICES)
model.to(DEVICES[0]) # we put the module on first device to start

# custom collate_fn. Same as normal collate fn but does .chunk() at the end to return a tuple of tensors (one per GPU) rather than a single tensor
# This is the main complicated part of TupleParallel but it's not that bad!
class CollateFunction:
    def __init__(self,ndevices):
        self.ndevices = ndevices
    def __call__(self,data_list):
        """
        Takes a list of Dataset.__getitem__ results as input
        Output will be returned from DataLoader

        For reference the default for a tensor list input is is torch.stack(data_list) I believe
        """
        return torch.stack(data_list).chunk(self.ndevices)


# DataLoader initialized with collate_fn
rand_loader = DataLoader(
    dataset=RandomDataset(input_size, data_size),
    pin_memory=True,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=CollateFunction(len(DEVICES)))


# training loop
# now thanks to collate_fn rand_loader will return tuples
for data in rand_loader: 
    output = model(data)

```



