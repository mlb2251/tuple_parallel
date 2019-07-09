# TupleParallel

##Quick Summary
The `TupleParallel` module extends the `torch.nn.DataParallel` module to work with input of a much more general form. Wrapping a module in `TupleParallel` done in the same way as DataParallel (and any extra args and kwargs are forwarded to DataParallel). Unlike DataParallel, the `forward()` function for `TupleParallel` takes a tuple of length N as input, where N is the number of GPUs that `TupleParallel` was initialized with. Each element of the tuple should be a batch that would normally be sent as input to the module that `TupleParallel` is warpping. To achieve the same results for a given batch size as running a single batch on one gpu, you can split that batch into N mini batches and send the tuple of minibatches to `TupleParallel`. `TupleParallel` will transfer the data to the GPUs for you by calling .`to(non_blocking=True)` on them (for custom datatypes this means you should implement `.to()`). `forward()` will return a tuple of outputs, still on their respective GPUs. See comment at the start of `forward()` for more. Also see example.py.

##Excerpts from `example.py`
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



