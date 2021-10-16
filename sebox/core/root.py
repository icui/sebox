from __future__ import annotations
from importlib import import_module
import typing as tp
import signal
import asyncio

from .node import Node

if tp.TYPE_CHECKING:
    from sebox.core.mpi import MPI
    from sebox.typing.modules import System


class Root(Node):
    """Root node with job configuration."""
    # job name
    job_name: str

    # number of nodes to request
    job_nnodes: int

    # account to submit the job
    job_account: str

    # amount of walltime to request
    job_walltime: float

    # submit to debug queue and do not requeue if job fails
    job_debug: bool

    # avoid calling new MPI tasks if remaining walltime is less than certain minutes
    job_gap: float

    # any task failed during execution
    job_failed: bool

    # any task failed twice during execution
    job_aborted: bool

    # paused due to insuffcient time
    job_paused: bool

    # number of CPUs per node
    cpus_per_node: tp.Optional[int]

    # number of GPUs per node
    gpus_per_node: tp.Optional[int]

    # module for job scheduler
    module_system: str

    # default number of nodes to run MPI tasks (if task_nnodes is None, task_nprocs must be set)
    task_nnodes: tp.Optional[int]

    # MPI workspace (only available with __main__ from sebox.core.mpi)
    _mpi: tp.Optional[MPI] = None

    # runtime global cache (use underscore to avoid being saved by __getstate__)
    _cache: tp.Dict[str, tp.Any] = {}

    # module of job scheduler
    _sys: System

    # currently being saved
    _saving = False
    
    @property
    def cache(self):
        return self._cache

    @property
    def sys(self) -> System:
        return self._sys
    
    @property
    def mpi(self) -> MPI:
        return tp.cast('MPI', self._mpi)

    @property
    def task_nprocs(self) -> int:
        """Number of processors to run MPI tasks."""
        if 'task_nprocs' in self._init:
            return self._init['task_nprocs']

        if 'task_nprocs' in self._data:
            return self._data['task_nprocs']

        return tp.cast(int, self.task_nnodes) * (self.cpus_per_node or self.sys.cpus_per_node)
    
    async def execute(self):
        """Execute main task."""
        self.restore()

        # reset execution state
        self.job_failed = False
        self.job_aborted = False
        self.job_paused = False

        # requeue before job gets killed
        if not self.job_debug:
            signal.signal(signal.SIGALRM, self._signal)
            signal.alarm(int((self.job_walltime - self.job_gap) * 60))

        await super().execute()

        # requeue job if task failed
        if self.job_failed and not self.job_aborted and not self.job_debug:
            self.sys.requeue()
    
    def save(self):
        """Save state."""
        if self.job_paused or self._saving:
            # job is already being saved
            return

        if self.mpi:
            # root can only be saved from main process
            raise RuntimeError('cannot save root from MPI process')
        
        self._saving = True
        asyncio.create_task(self._save())
    
    def restore(self, node: tp.Optional[Node] = None):
        """Restore state."""
        if hasattr(self, '_sys'):
            return
        
        if node:
            # restore from a saved workspace (e.g. pickle file from mpiexec)
            while node.parent is not None:
                node = node.parent
            
            self.__setstate__(node.__getstate__())

        elif self.mpi is None and self.has('root.pickle'):
            # restore previous state
            self.__setstate__(self.load('root.pickle'))
        
        elif self.has('config.toml'):
            # load configuration
            self._init.update(root.load('config.toml'))

        # load module of job scheduler
        self._sys = tp.cast('System', import_module(f'sebox.system.{self.module_system}'))

    async def _save(self):
        """Save to root.pickle"""
        if self._saving:
            self.dump(self.__getstate__(), 'root.pickle')

            if self.job_paused and not self.job_aborted:
                self.sys.requeue()
            
            else:
                self._saving = False
    
    def _signal(self, *_):
        """Requeue due to insufficient time."""
        if not self.job_aborted:
            self.job_paused = True
            self.save()


# create root node
root = Root('.', {}, None)
