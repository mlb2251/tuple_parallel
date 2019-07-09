The TupleParallel module extends the DataParallel module to work with input of a much more general form. Wrapping a module in TupleParallel done in the same way as DataParallel (and any extra args and kwargs are forwarded to DataParallel). Unlike DataParallel, the forward() function for TupleParallel takes a tuple of length N as input, where N is the number of GPUs that TupleParallel was initialized with. Each element of the tuple should be a batch that would normally be sent as input to the module that TupleParallel is warpping. To achieve the same results for a given batch size as running a single batch on one gpu, you can split that batch into N mini batches and send the tuple of minibatches to TupleParallel. TupleParallel will transfer the data to the GPUs for you by calling .to(non_blocking=True) on them (for custom datatypes this means you should implement .to()). forward() will return a tuple of outputs, still on their respective GPUs. See comment at the start of forward() for more.
