from __future__ import annotations
from os import path
from sys import stderr
from importlib import import_module
from time import time
from datetime import timedelta
from functools import partial
import asyncio
import typing as tp

from .directory import Directory


class Workspace(Directory):
    """A directory with a task."""
    # workspace task
    task: Task

    # task progress prober
    prober: Prober

    # whether child workspaces are executed concurrently
    concurrent: tp.Optional[bool]

    # workspace to inherit properties from
    inherit: tp.Optional[Workspace]

    # display name
    _name: tp.Optional[str] = None

    # initial data passed to self.__init__
    _init: dict

    # data modified by self.task
    _data: dict

    # parent workspace
    _parent: tp.Optional[Workspace]

    # time when task started
    _starttime: tp.Optional[float] = None

    # time when task ended
    _endtime: tp.Optional[float] = None

    # exception raised from self.task
    _err: tp.Optional[Exception] = None

    # child workspaces
    _ws: tp.List[Workspace]

    @property
    def name(self) -> str:
        """Directory name."""
        if self._name is not None:
            return self._name

        func = self.task
        
        if func:
            # use task function name as workspace name
            if isinstance(func, tuple) or isinstance(func, list):
                return func[1]

            while isinstance(func, partial):
                func = func.func
            
            if hasattr(func, '__name__'):
                return func.__name__.lstrip('_')

        # use directory name as workspace name
        return path.basename(self.path(abs=True))

    @property
    def parent(self) -> Workspace:
        """Parent workspace."""
        return tp.cast(Workspace, self._parent)

    @property
    def done(self) -> bool:
        """Main function and child workspaces executed successfully."""
        if self._endtime:
            return all(ws.done for ws in self._ws)

        return False
    
    @property
    def elapsed(self) -> tp.Optional[float]:
        """Total walltime."""
        if self.done:
            delta = self._endtime - self._starttime # type: ignore
            delta_ws = tp.cast(tp.List[float], [ws.elapsed for ws in self._ws])

            if self.concurrent and len(delta_ws) > 1:
                return delta + max(*delta_ws)

            return delta + sum(delta_ws)
    
    def __init__(self, cwd: str, data: dict, parent: tp.Optional[Workspace]):
        super().__init__(cwd)
        self._init = data
        self._data = {}
        self._parent = parent
        self._ws = []
    
    def __getattr__(self, key: str):
        """Get workspace data (including parent data)."""
        if key.startswith('_'):
            return object.__getattribute__(self, key)

        if key in self._data:
            return self._data[key]

        if key in self._init:
            return self._init[key]
        
        if key not in tp.get_type_hints(Workspace):
            if self.inherit:
                return self.inherit.__getattr__(key)

            elif self._parent:
                return self._parent.__getattr__(key)
        
        return None
    
    def __setattr__(self, key: str, val):
        """Set workspace data."""
        if key.startswith('_'):
            object.__setattr__(self, key, val)
        
        else:
            self._data[key] = val
    
    def __getstate__(self):
        """Items to be saved when pickled."""
        state = {}
        
        for key in tp.get_type_hints(Workspace):
            if key.startswith('_'):
                state[key] = getattr(self, key)
        
        return state
    
    def __setstate__(self, state):
        """Restore from saved state."""
        for key, val in state.items():
            setattr(self, key, val)

    def __getitem__(self, key: int) -> Workspace:
        """Get child workspace."""
        return self._ws[key]

    def __len__(self):
        return len(self._ws)
    
    def __str__(self):
        from .root import root
        
        name = self.name

        if self._err:
            name += ' (failed)'
        
        elif self._starttime:
            if self._endtime:
                if elapsed := self.elapsed:
                    # task done
                    delta = str(timedelta(seconds=int(round(elapsed))))

                    if delta.startswith('0:'):
                        delta = delta[2:]
                    
                    name += f' ({delta})'

            else:
                # task started but not finished
                if root.job_paused:
                    name += ' (terminated)'
                
                else:
                    if self.prober:
                        try:
                            state = self.prober(self)

                            if isinstance(state, float):
                                name += f' ({int(state*100)}%)'
                            
                            else:
                                name += f' ({state})'

                        except:
                            pass
                    
                    if name == self.name:
                        name += ' (running)'
        
        return name


    def __repr__(self):
        return self.stat(False)
    
    async def run(self):
        """Execute task and child tasks."""
        await self.run_task()
        await self.run_ws()
    
    async def run_task(self):
        """Execute self.task."""
        from .root import root

        if self._endtime:
            return
        
        # save whether previous run failed
        err = self._err

        # backup data and reset state before execution
        self._starttime = time()
        self._endtime = None
        self._err = None
        self._data.clear()
        root.save()

        try:
            # import task
            task = self.task

            if isinstance(task, tuple) or isinstance(task, list):
                path = task[0]

                if path.startswith('module:'):
                    # choice of sebox module
                    path = f'sebox.{path[7:]}.' + getattr(self, f'module_{path[7:]}')

                task = getattr(import_module(path), task[1])

            # call task function
            if task and (result := task(self)) and asyncio.iscoroutine(result):
                await result
        
        except Exception as e:
            from traceback import format_exc
            
            self._starttime = None
            self._err = e

            print(format_exc(), file=stderr)

            if err or root.job_debug:
                # job failed twice or job in debug mode
                root.job_aborted = True
            
            else:
                # job failed in its first attempt
                root.job_failed = True
        
        else:
            self._endtime = time()
        
        root.save()
    
    async def run_ws(self):
        """Execute self._ws."""
        if not self._endtime:
            return
        
        from .root import root
        
        # skip executed nodes
        exclude = []

        def get_unfinished():
            wss: tp.List[Workspace] = []
            
            for ws in self._ws:
                if ws not in exclude and not ws.done:
                    wss.append(ws)
            
            return wss

        while len(wss := get_unfinished()):
            if self.concurrent:
                # execute nodes concurrently
                exclude += wss
                await asyncio.gather(*(ws.run() for ws in wss))

            else:
                # execute nodes in sequence
                exclude.append(wss[0])
                await wss[0].run()

            # exit if any error occurs
            if root.job_failed or root.job_aborted:
                break
    
    def update(self, items: dict):
        """Update properties from dict."""
        self._data.update(items)

    def add(self, name: tp.Union[str, Task[tp.Any]] = None, /,
        task: Task[tp.Any] = None, *,
        concurrent: tp.Optional[bool] = None, prober: Prober = None,
        inherit: tp.Optional[Workspace] = None, **data) -> Workspace:
        """Add a child workspace or a child task."""
        if name is not None and not isinstance(name, str):
            if task is not None:
                raise TypeError('duplicate task when adding workspace')

            task = name
        
        if task is not None:
            data['task'] = task
        
        if prober is not None:
            data['prober'] = prober
        
        if concurrent is not None:
            data['concurrent'] = concurrent
        
        if inherit is not None:
            data['inherit'] = inherit

        if isinstance(name, str):
            ws = Workspace(self.path(name), data, self)
            ws._name = name
        
        else:
            ws = Workspace(self.path(), data, self)
        
        self._ws.append(ws)

        return ws
    
    def add_mpi(self, cmd: tp.Union[str, tp.Callable], 
        nprocs: tp.Optional[int] = None, per_proc: tp.Union[int, tp.Tuple[int, int]] = (1, 0), *,
        name: tp.Optional[str] = None, arg: tp.Any = None, arg_mpi: tp.Optional[list] = None,
        check_output: tp.Optional[tp.Callable[[str], None]] = None):
        """Run MPI task."""
        from .mpiexec import mpiexec, getname

        if nprocs is None:
            from sebox import root
            nprocs = root.task_nprocs

        if isinstance(per_proc, int):
            per_proc = (per_proc, per_proc)
        
        ws = self.add(partial(mpiexec, self, cmd, nprocs, per_proc[0], per_proc[1], name, arg, arg_mpi, check_output))

        if name is not None:
            ws._name = name

        else:
            ws._name = getname(cmd)
        
        return ws
    
    def reset(self):
        """Reset workspace (including child workspaces)."""
        self._starttime = None
        self._endtime = None
        self._err = None
        self._data.clear()
        self._ws.clear()
    
    def stat(self, verbose: bool = False):
        """Structure and execution status."""
        stat = str(self)

        if not verbose:
            stat = stat.split(' ')[0]

        def idx(j):
            if self.concurrent:
                return '- '

            return '0' * (len(str(len(self) - 1)) - len(str(j))) + str(j) + ') '
            
        collapsed = False

        for i, node in enumerate(self._ws):
            stat += '\n' + idx(i)

            if not verbose and (node.done or (collapsed and node._starttime is None)):
                stat += str(node)
        
            else:
                collapsed = True
                
                if len(node):
                    stat += '\n  '.join(node.stat(verbose).split('\n'))
                
                else:
                    stat += str(node)
        
        return stat


# type annotation for a workspace task function
T = tp.TypeVar('T', bound=Workspace)
Task = tp.Optional[tp.Union[tp.Callable[[T], tp.Optional[tp.Coroutine]], tp.Tuple[str, str]]]
Prober = tp.Optional[tp.Callable[[Workspace], tp.Union[float, str, None]]]
