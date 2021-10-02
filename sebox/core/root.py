from __future__ import annotations
from importlib import import_module
import typing as tp
import signal

from .workspace import Workspace

if tp.TYPE_CHECKING:
    from sebox.typing import SystemModule


class Root(Workspace):
    """Root workspace with job configuration."""
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

    # number of nodes to run MPI tasks (if task_nnodes is None, task_nprocs must be set)
    task_nnodes: tp.Optional[int]

    # runtime global cache
    _cache: tp.Dict[str, tp.Any] = {}

    # module of job scheduler
    _sys: SystemModule
    
    @property
    def cache(self):
        return self._cache

    @property
    def sys(self) -> SystemModule:
        return self._sys

    @property
    def task_nprocs(self) -> int:
        """Number of processors to run MPI tasks."""
        if 'task_nprocs' in self._data:
            return self._data['task_nprocs']

        return tp.cast(int, self.task_nnodes) * (self.cpus_per_node or self.sys.cpus_per_node)

    def submit(self):
        """Submit job to scheduler."""
    
    async def run(self):
        """Run main task."""
        self.restore()
        self.job_failed = False
        self.job_aborted = False

        # requeue before job gets killed
        if not self.job_debug:
            signal.signal(signal.SIGALRM, self._signal)
            signal.alarm(int((self.job_walltime - self.job_gap) * 60))

        await super().run()

        # requeue job if task failed
        if self.job_failed and not self.job_aborted and not self.job_debug:
            self.sys.requeue()
    
    def save(self):
        """Save state."""
        self.dump(self.__getstate__(), 'root.pickle')
    
    def restore(self, ws: tp.Optional[Workspace] = None):
        """Restore state."""
        if hasattr(self, '_sys'):
            return
        
        if ws:
            # restore from a saved workspace
            while ws.parent is not None:
                ws = ws.parent
            
            self.__setstate__(ws.__getstate__())

        elif self.has('root.pickle'):
            # restore previous state
            print('?')
            exit()
            self.__setstate__(self.load('root.pickle'))
        
        elif self.has('config.toml'):
            # load configuration
            self._init.update(root.load('config.toml'))
        
        self.job_paused = False

        # load module of job scheduler
        self._sys = tp.cast('SystemModule', import_module(f'sebox.system.{self.module_system}'))
        
    
    def _signal(self, *_):
        """Requeue due to insufficient time."""
        if not self.job_aborted:
            self.job_paused = True
            self.save()
            self.sys.requeue()


# create root workspace
root = Root('.', {}, None)
