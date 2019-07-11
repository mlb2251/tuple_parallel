import torch
from torch import nn
import time
import threading
#from mlb_py.util import breaker


class TupleParallel(nn.DataParallel):
    def __init__(self,*args,parallel_transfer=True,custom_parallel_apply=True,**kwargs):
        """
        parallel_transfer: Use this if your data is not on the correct devices already. Does a nonblocking (if pinned memory) transfer of data to each gpu (small speedup).
        custom_parallel_apply: Use my version of parallel_apply which gives timing info and removes locks that I think are unnecessary (very small speedup)
        """
        super().__init__(*args,**kwargs)
        self.parallel_transfer = parallel_transfer
        self.custom_parallel_apply = custom_parallel_apply

        # Timers for analysis
        self._first = True
        self.timer = Timer() # TupleParallel timer
        self.partime = Timer() # parallel_apply timer
        self.thread_timers = [Timer() for _ in range(len(self.device_ids))] # one per gpu
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

        self.timer.start('total')
        assert len(input_tuple) == len(self.device_ids)


        if self.parallel_transfer:
            # transfer to gpu nonblocking
            for i in range(len(input_tuple)):
                self.timer.start(f'nonblocking transfer {self.device_ids[i]}')
                with torch.cuda.device(self.device_ids[i]):
                    input_tuple[i] = to_gpu(input_tuple[i],self.device_ids[i])
                self.timer.stop(f'nonblocking transfer {self.device_ids[i]}')
        else:
            assert all([input.device.index==d for input,d in zip(input_tuple,self.device_ids)]), "`devices` does not match with the devices that the actual inputs are on. Use TupleParallel(parallel_transfer=True) to do this transfer for you"

        # single GPU case
        if len(self.device_ids) == 1:
            return [self.module(input_tuple[0], **kwargs_shared)]

        self.timer.start('replicate')
        replicas = self.replicate(self.module, self.device_ids)
        self.timer.stop()

        self.timer.start('par_apply')
        if self.custom_parallel_apply:
            outputs = self.parallel_apply_timed(replicas, input_tuple, kwargs_shared)
        else:
            outputs = self.parallel_apply(replicas, input_tuple, [kwargs_shared for _ in input_tuple])
        self.timer.stop('par_apply')
        self.timer.stop('total')

        if self._first:
            self.timer.clear()
            self.partime.clear()
            [t.clear() for t in self.thread_timers]
            self._first = False

        #breaker('par')
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
    def parallel_apply_timed(self,modules, inputs, kwargs_shared={}):
        """
        This is a slightly modified version of DataParallel.parallel_apply. Using the original works as well and you should try it if you have issue with this.

        Timing notes: ~100% of time should be spent in the module() call for each thread. Additionally ~7% of total time will be spent on launching threads.
        """
        self.partime.start('total')
        assert len(modules) == len(inputs) == len(self.device_ids), ""
        assert isinstance(kwargs_shared,dict)
        results = [None]*len(self.device_ids)
        grad_enabled = torch.is_grad_enabled()

        def _worker(i, module, input, device):
            self.thread_timers[i].start(f'total')
            torch.set_grad_enabled(grad_enabled)
            try:
                #t.start(f'{i}.to')
                #input = input.to(device) ## ADDED
                #t.stop(f'{i}.to')
                with torch.cuda.device(device): # negligible
                    self.thread_timers[i].start(f'module()')
                    output = module(input, **kwargs_shared) # 94%
                    self.thread_timers[i].stop(f'module()')
                #with lock:
                #    results[i] = output
                results[i] = output
            except Exception as e:
                #with lock:
                results[i] = e
            self.thread_timers[i].stop(f'total')

        self.partime.start('threads') # 100%
        if len(modules) > 1:
            threads = [threading.Thread(target=_worker,
                                        args=(i, module, input, device))
                       for i, (module, input, device) in
                       enumerate(zip(modules, inputs, self.device_ids))]

            self.partime.start(f'launch_join')
            for i,thread in enumerate(threads):
                self.thread_timers[i].start(f'launch_time') # abt 7-16% of the time spent inside is spent on this start() call
                thread.start()
                self.thread_timers[i].stop(f'launch_time')
            for i,thread in enumerate(threads):
                thread.join()
            self.partime.stop(f'launch_join')
        else:
            _worker(0, modules[0], inputs[0], self.device_ids[0])
        self.partime.stop('threads')

        outputs = []
        for i in range(len(inputs)):
            output = results[i]
            if isinstance(output, Exception):
                raise output
            outputs.append(output)
        self.partime.stop('total')
        return outputs


def to_gpu(val,dev):
    """
    Calls val.to(dev,non_blocking=True) but recurses on lists/tuples
    """
    if type(val) == list:
        return [to_gpu(item,dev) for item in val]
    if type(val) == tuple:
        return tuple([to_gpu(item,dev) for item in val])
    if hasattr(val,'cuda'):
        try:
            return val.cuda(non_blocking=True)
        except TypeError:
            raise TypeError(".cuda() implementation must allow the call .cuda(non_blocking=True)")
    if hasattr(val,'to'):
        try:
            return val.to(dev,non_blocking=True)
        except TypeError:
            raise TypeError(".to() implementation must allow the call .to(device,non_blocking=True)")
    raise TypeError("You must implement a .to(device, non_blocking=False) or .cuda() function for your type")


# helper class for timing things.
class Time():
    def __init__(self,name,parent,cumulative=False):
        self.name = name
        self.cumulative = cumulative
        self.count = 0
        self._start = None
        self.elapsed = 0
        self.avg = None
        self.parent=parent
    def start(self):
        self._start = time.time()
    def stop(self):
        self.count += 1
        dt = time.time() - self._start
        self.elapsed = (self.elapsed + dt) if self.cumulative else dt
        if self.cumulative:
            self.avg = self.elapsed/self.count
    def percent(self):
        assert self.name != 'total' and 'total' in self.parent.timers and self.parent.timers['total'].elapsed != 0
        return self.elapsed/self.parent.timers['total'].elapsed*100
    def __repr__(self):
        if self.name != 'total' and 'total' in self.parent.timers and self.parent.timers['total'].elapsed != 0:
            percent = self.percent()
            return f'{self.name+":":<{self.parent.longest_name+1}} tot:{str(self.elapsed)+",":<23} avg:{self.avg}, {percent:.3f}%'
        return f'{self.name+":":<{self.parent.longest_name+1}} tot:{str(self.elapsed)+",":<23} avg:{self.avg}'
class Timer:
    def __init__(self,cumulative=True):
        self.timers = {}
        self.most_recent=None
        self.cumulative = cumulative
        self.longest_name = 5
    def start(self,name='timer',cumulative=None):
        if len(name) > self.longest_name:
            self.longest_name = len(name)
        if cumulative is None:
            cumulative=self.cumulative
        self.most_recent = name
        if name not in self.timers:
            self.timers[name] = Time(name,self,cumulative)
        if not hasattr(self,name):
            setattr(self,name,self.timers[name])
        self.timers[name].start()
        return self
    def clear(self):
        for name in self.timers:
            if isinstance(getattr(self,name),Time):
                delattr(self,name)
        self.timers = {}
    def stop(self,name=None):
        """
        Convenience: if you dont provide `name` it uses the last one that `start` was called with. stop() also returns the elapsed time and also does a setattr to set the field with the name `name` to the elapsed time.
        """
        if name is None:
            name = self.most_recent
        assert name in self.timers, f"You need to .start() your timer '{name}'"
        self.timers[name].stop()
        return self.timers[name]
    def print(self,name=None):
        if name is None:
            name = self.most_recent
        print(self.stop(name))
    def __repr__(self):
        body = []
        for timer in self.timers.values():
            body.append(repr(timer))
        return 'Timers:\n'+',\n'.join(body)


