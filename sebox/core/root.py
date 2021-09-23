from __future__ import annotations
from importlib import import_module
import typing as tp
import signal

from .directory import Directory
from .workspace import Workspace

if tp.TYPE_CHECKING:
    from sebox.system import System


# root directory
_rootdir = Directory('.')


class Job(tp.Protocol):
    """Job configuration for submitting to scheduler."""
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


class Root(Workspace, Job):
    """Root workspace with job configuration."""
    # runtime global cache
    _cache: tp.Dict[str, tp.Any] = {}

    @property
    def sys(self) -> System:
        if 'sys' not in self._cache:
            self._cache['sys'] = import_module(f'sebox.system.{root.module_system}')

        return tp.cast(tp.Any, self._cache['sys'])
    
    @property
    def cache(self):
        return self._cache

    def submit(self):
        """Submit job to scheduler."""
    
    async def run(self):
        """Run main task."""
        # requeue before job gets killed
        signal.signal(signal.SIGALRM, _check_requeue)
        signal.alarm(int((self.job_walltime - self.job_gap) * 60))

        await super().run()

        # requeue job if task failed
        if self.job_failed:
            _check_requeue()


if _rootdir.has('root.pickle'):
    # load saved root workspace
    root = tp.cast(Root, _rootdir.load('root.pickle'))

else:
    # create new root workspace
    root = Root('.', False, None, None)
    
    if root.has('config.toml'):
        root._data.update(root.load('config.toml'))


def _check_requeue(*_):
    """Requeue job if necessary."""
    if not root.job_aborted and not root.job_debug:
        root.sys.requeue()
