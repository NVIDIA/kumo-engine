"""Local or S3 output via pyarrow.fs. One S3 client per process (each writer is a process)."""
import os

import pyarrow.fs as pafs

_FS = {}


def filesystem(root):
    """Returns (fs, base path) for a local directory or s3://bucket/prefix."""
    if root.startswith("s3://"):
        bucket = root[5:].split("/", 1)[0]
        if root not in _FS:
            _FS[root] = pafs.S3FileSystem(region=pafs.resolve_s3_region(bucket), request_timeout=60,
                                          connect_timeout=10)
        return _FS[root], root[5:].rstrip("/")
    return pafs.LocalFileSystem(), os.path.abspath(root)


def write_bytes(fs, path, data):
    if isinstance(fs, pafs.LocalFileSystem):  # on S3 this would leave empty "directory" objects
        fs.create_dir(path.rsplit("/", 1)[0], recursive=True)
    with fs.open_output_stream(path) as f:
        f.write(data)


def read_bytes(fs, path):
    with fs.open_input_stream(path) as f:
        return f.read()


def exists(fs, path):
    return fs.get_file_info(path).type != pafs.FileType.NotFound


def list_files(fs, path):
    sel = pafs.FileSelector(path, recursive=True, allow_not_found=True)
    return sorted((i.path, i.size) for i in fs.get_file_info(sel) if i.type == pafs.FileType.File)
