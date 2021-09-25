from __future__ import annotations
from importlib import import_module
import typing as tp
import signal

from .workspace import Workspace

if tp.TYPE_CHECKING:
    from sebox.system import System


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

    # number of CPUs per node
    cpus_per_node: int

    # number of GPUs per node
    gpus_per_node: int

    # module for job scheduler
    module_system: str
    
    # runtime global cache
    _cache: tp.Dict[str, tp.Any] = {}

    # module of job scheduler
    _sys: System
    
    @property
    def cache(self):
        return self._cache

    @property
    def sys(self) -> System:
        return self._sys

    def submit(self):
        """Submit job to scheduler."""
    
    async def run(self):
        """Run main task."""
        self.restore()

        # requeue before job gets killed
        signal.signal(signal.SIGALRM, self._check_requeue)
        signal.alarm(int((self.job_walltime - self.job_gap) * 60))

        await super().run()

        # requeue job if task failed
        if self.job_failed:
            self._check_requeue()
    
    def save(self):
        """Save state."""
        self.dump(self.__getstate__(), 'root.pickle')
    
    def restore(self):
        """Restore state."""
        if hasattr(self, '_sys'):
            return

        if self.has('root.pickle'):
            # restore previous state
            self.__setstate__(self.load('root.pickle'))
        
        elif self.has('config.toml'):
            # load configuration
            self._data.update(root.load('config.toml'))
        
        # load module of job scheduler
        self._sys = tp.cast(tp.Any, import_module(f'sebox.system.{self.module_system}'))

    def _check_requeue(self, *_):
        """Requeue job if necessary."""
        if not self.job_aborted and not self.job_debug:
            self.sys.requeue()


# create root workspace
root = Root('.', False, None, None)
