import typing as tp

from sebox import root, Directory


def getdir() -> Directory:
    return Directory(tp.cast(str, root.path_catalog))


def getevents(group: tp.Optional[int] = None) -> tp.List[str]:
    return []


def getstations(event: tp.Optional[str] = None, group: tp.Optional[int] = None) -> tp.List[str]:
    return []
