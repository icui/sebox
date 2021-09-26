from os import path, fsync
from functools import partial
from subprocess import check_call
from glob import glob
import pickle
import toml
import typing as tp


# supported types for directory.load() and directory.dump()
DumpType = tp.Literal['pickle', 'npy', 'toml', None]


class Directory:
    """Directory related operations."""
    # relative path
    _cwd: str

    @property
    def cwd(self) -> str:
        return self._cwd
    
    def __init__(self, cwd: str):
        self._cwd = cwd
    
    def path(self, *paths: str, abs: bool = False) -> str:
        """Get relative path of a sub directory."""
        src = path.normpath(path.join(self.cwd, *paths))
        return path.abspath(src) if abs else src
    
    def rel(self, *paths: str) -> str:
        """Convert from a path relative to root directory to a path relative to current directory."""
        return path.relpath(path.join('.', *paths), self.path())
    
    def has(self, src: str = '.'):
        """Check if a file or a directory exists."""
        return path.exists(self.path(src))
    
    def rm(self, src: str = '.'):
        """Remove a file or a directory."""
        check_call('rm -rf ' + self.path(src), shell=True)
    
    def cp(self, src: str, dst: str = '.'):
        """Copy file or a directory."""
        self.mkdir(path.dirname(dst))

        check_call(f'cp -r {self.path(src)} {self.path(dst)}', shell=True)
    
    def mv(self, src: str, dst: str = '.'):
        """Move a file or a directory."""
        self.mkdir(path.dirname(dst))

        check_call(f'mv {self.path(src)} {self.path(dst)}', shell=True)
    
    def ln(self, src: str, dst: str = '.'):
        """Link a file or a directory."""
        self.mkdir(path.dirname(dst))

        # relative source path
        if not path.isabs(src):
            src = self.path(src, abs=True)

            if not path.isabs(dst):
                if not path.isdir(dstdir := self.path(dst, abs=True)):
                    dstdir = path.dirname(dstdir)

                src = path.join(path.relpath(path.dirname(src), dstdir), path.basename(src))

        # delete existing target file
        dst = self.path(dst)

        if path.isdir(dst):
            self.rm(path.join(dst, path.basename(src)))
        else:
            self.rm(dst)

        check_call(f'ln -s {src} {dst}', shell=True)
    
    def mkdir(self, dst: str = '.'):
        """Create a directory recursively."""
        check_call('mkdir -p ' + self.path(dst), shell=True)
    
    def ls(self, src: str = '.', grep: str = '*', isdir: tp.Optional[bool] = None) -> tp.List[str]:
        """List items in a directory."""
        entries: tp.List[str] = []

        for entry in glob(self.path(path.join(src, grep))):
            # skip non-directory entries
            if isdir is True and not path.isdir(entry):
                continue
            
            # skip directory entries
            if isdir is False and path.isdir(entry):
                continue
            
            entries.append(entry.split('/')[-1])

        return entries

    def read(self, src: str) -> str:
        """Read text file."""
        with open(self.path(src), 'r') as f:
            return f.read()

    def write(self, text: str, dst: str, mode: str = 'w'):
        """Write text and wait until write is complete."""
        self.mkdir(path.dirname(dst))

        with open(self.path(dst), mode) as f:
            f.write(text)
            f.flush()
            fsync(f.fileno())
    
    def readlines(self, src: str) -> tp.List[str]:
        """Read text file lines."""
        return self.read(src).split('\n')
    
    def writelines(self, lines: tp.Iterable[str], dst: str, mode: str = 'w'):
        """Write text lines."""
        self.write('\n'.join(lines), dst, mode)
    
    def load(self, src: str, ext: DumpType = None) -> tp.Any:
        """Load a pickle / toml file."""
        if ext is None:
            ext = src.split('.')[-1] # type: ignore
        
        if ext == 'pickle':
            with open(self.path(src), 'rb') as fb:
                return pickle.load(fb)
        
        elif ext == 'toml':
            with open(self.path(src), 'r') as f:
                return toml.load(f)
        
        elif ext == 'npy':
            import numpy as np
            return np.load(self.path(src))
        
        else:
            raise TypeError(f'unsupported file type {ext}')
    
    def dump(self, obj, dst: str, ext: DumpType = None):
        """Save a pickle / toml file."""
        self.mkdir(path.dirname(dst))

        if ext is None:
            ext = dst.split('.')[-1] # type: ignore

        if ext == 'pickle':
            with open(self.path(dst), 'wb') as fb:
                pickle.dump(obj, fb)
        
        elif ext == 'toml':
            with open(self.path(dst), 'w') as f:
                toml.dump(obj, f)
        
        elif ext == 'npy':
            import numpy as np
            return np.save(self.path(dst), obj)
        
        else:
            raise TypeError(f'unsupported file type {ext}')
    
    async def mpiexec(self, cmd: tp.Union[str, tp.Callable], 
        nprocs: int = 1, per_proc: tp.Union[int, tp.Tuple[int, int]] = (1, 0), *,
        name: tp.Optional[str] = None, arg: tp.Any = None, arg_mpi: tp.Optional[list] = None):
        """Run MPI task."""
        from .mpiexec import mpiexec

        if isinstance(per_proc, int):
            per_proc = (per_proc, per_proc)
        
        await mpiexec(self, cmd, nprocs, per_proc[0], per_proc[1], name, arg, arg_mpi)
