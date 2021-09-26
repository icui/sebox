from __future__ import annotations
import typing as tp


if tp.TYPE_CHECKING:
    from sebox import Task, Workspace


    class DatasetModule(tp.Protocol):
        """Required functions in a dataset module."""
        # extract data from archive format to MPI format
        scatter: Task[Convert]

        # bundle data from MPI format to archive format
        gather: Task[Convert]


    class Convert(Workspace):
        """A workspace to convert dataset from / to MPI format."""
        # path to bundled data file
        path_bundle: str

        # path to MPI data file
        path_mpi: str

        # tag of bundled data file
        tag_bundle: tp.Optional[str]

        # tag of MPI data file
        tag_mpi: tp.Optional[str]

        # collective data
        stats: tp.Optional[dict]
