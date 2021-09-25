from __future__ import annotations
from os import path
from sys import stderr
from importlib import import_module
from time import time
import asyncio
import typing as tp

from .directory import Directory


# properties that should not inherit from parent workspace
_locals = ('task', 'prober', 'concurrent')

# root properties that can be set
_root = ('job_paused', 'job_failed', 'job_aborted')


class Workspace(Directory):
    """A directory with a task."""
    # workspace task
    task: Task

    # task progress prober
    prober: tp.Optional[tp.Callable]

    # whether child workspaces are executed concurrently
    concurrent: bool

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
        return path.basename(self.abs())

    @property
    def parent(self) -> tp.Optional[Workspace]:
        """Parent workspace."""
        return self._parent

    @property
    def done(self) -> bool:
        """Main function and child workspaces executed successfully."""
        if self._endtime:
            return all(ws.done for ws in self._ws)

        return False
    
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
        
        if self._parent and key not in _locals:
            return self._parent.__getattr__(key)
        
        return None
    
    def __setattr__(self, key: str, val):
        """Set workspace data."""
        if key.startswith('_'):
            object.__setattr__(self, key, val)
        
        elif (self._endtime or not self._starttime) and key not in _root:
            raise AttributeError('workspace property can only be set by its task')
        
        else:
            self._data[key] = val
    
    def __getstate__(self):
        """Items to be saved when pickled."""
        state = {}
        
        for key in tp.get_type_hints(Workspace):
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

        if self.done:
            name += ' (done)'
        
        elif self._err:
            name += ' (failed)'
        
        elif self._starttime and not self._endtime:
            if root.job_paused:
                name += ' (terminated)'
            
            else:
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

        try:
            # import task
            task: tp.Any = self.task

            if isinstance(task, tuple) or isinstance(task, list):
                module = import_module(task[0] + '.' + getattr(self, 'module_' + task[0].split('.')[-1]))
                task = getattr(module, task[1])

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

    @tp.overload
    def add(self, name: str, data: tp.Optional[tp.Union[bool, dict]]) -> Workspace:
        """Add a child workspace."""
    
    @tp.overload
    def add(self, name: tp.Callable[..., tp.Optional[tp.Coroutine]]) -> Workspace:
        """Add a child task."""
    
    @tp.overload
    def add(self, name: tp.Tuple[str, str]) -> Workspace:
        """Add a child task (imported from a module)."""

    def add(self, name: tp.Union[str, Task], data: tp.Optional[tp.Union[bool, dict]] = None) -> Workspace:
        """Add a child workspace or a child task."""
        if isinstance(name, str):
            # create a new workspace
            if isinstance(data, bool):
                data = { 'concurrent': data }

            ws = Workspace(self.rel(name), data or {}, self)
        
        else:
            # add a task to current workspace
            ws = Workspace(self.rel(), { 'task': name }, self)
        
        self._ws.append(ws)
        return ws
    
    def stat(self, verbose: bool = False):
        """Structure and execution status."""
        stat = str(self)

        if not verbose:
            stat = stat.split(' ')[0]

        def idx(j):
            if self._concurrent:
                return '- '

            return '0' * (len(str(len(self) + 1)) - len(str(j + 1))) + str(j + 1) + ') '
            
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
