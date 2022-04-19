import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..utilities.decorators import read_files
from ..utilities.logging import logger

STRUCT_NAME_RE = re.compile(r"^struct (\w+) \{$")
ARRAY_RE = re.compile(r"^\s*(\w+) (\w+)\((\d+)\);$")
VAR_RE = re.compile(r"^\s*(\w+) (\w+);$")
INFO_RE = re.compile(
    r" array\((char|short|int|long|float|double|complex|string)((?:,\w+)*)\)"
)


class PDBReader:
    _known_types = {
        **{k: int for k in ("char", "short", "int", "long")},
        **{k: float for k in ("float", "double")},
        "complex": complex,
        "string": str,
        "pointer": "yorick_pointer",
    }

    @read_files(1)
    def __init__(self, fname):
        try:
            import pyorick
        except ModuleNotFoundError as e:
            logger.error(
                "Reading PDB files requires the pyorick package. "
                "Install it via `pip install pyorick` along with Yorick."
            )
            raise e

        if not self.check(fname):
            raise Exception(
                "Unrecognized file format for file %s, "
                "could not read as PDB file." % fname
            )

        self._known_types = self._known_types.copy()
        self.yo = pyorick.Yorick()
        self.yo(f'f=openb("{fname}")')

        self._variables = self._get_vars()
        self._structure = {}
        for v in self._variables:
            self._structure[v] = self._parse_struct(v)

        logger.debug("File structure: %s", self._structure)

        self._data = {}

    def check(self, fname):
        # Test that the file starts with the magic string
        with open(fname, "br") as f:
            return f.readline().decode() == "!<<PDB:II>>!\n"

    def _get_vars(self):
        self.yo("ptrs=get_vars(f)")

        length = int(self.yo.e("numberof(ptrs)")) - 1
        variables = []
        for i in range(length):
            variables.extend(self.yo.e(f"*ptrs({i+1})"))
        return variables

    def _parse_struct(self, var_name):
        v = f"f.{var_name}"

        (info,) = self.yo(f"=info({v})")

        match = INFO_RE.match(info)
        if match:
            type_str, shape_str = match.groups()
            if shape_str:
                shape = tuple(int(s) for s in shape_str[1:].split(","))
            else:
                shape = (1,)

            return (self._known_types[type_str], shape)

        lines = self.yo.e(f"print(structof({v}))")

        # Now parse the structure
        remaining = lines[1:-1]  # last line is just }

        variables: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
        for line in remaining:
            tmp = ARRAY_RE.match(line)
            if tmp:
                type_name, name, length = tmp.groups()
                length = int(length)
            else:
                type_name, name = VAR_RE.match(line).groups()
                length = 0

            if type_name not in self._known_types:
                new_type = self._parse_struct(f"{var_name}.{name}")
                self._known_types[type_name] = new_type
                type_name = new_type
            else:
                type_name = self._known_types[type_name]

            if length == 0:
                spec = type_name
            else:
                spec = (type_name, length)
            variables[name] = spec
        return variables

    @property
    def structure(self):
        return self._structure

    def get(self, path):
        if path in self._data:
            return self._data[path]

        # Check the patch does exist
        keys = path.split("/")
        node = self.structure
        for k in keys:
            node = node.get(k, None)
            if node is None:
                raise KeyError("Provided path does not exist on file: %s" % path)
        cmd = "f.%s" % path.replace("/", ".")

        value = self.yo(f"={cmd}")
        self._data[path] = value
        return value

    def __getitem__(self, key):
        return self.get(key)

    def walk_structure(self, node: dict = None, prefix=None) -> Iterable[List[str]]:
        if node is None:
            node = self.structure
        if prefix is None:
            prefix = []
        for k, v in node.items():
            if isinstance(v, dict):
                yield from self.walk_structure(v, prefix + [k])
                continue

            if isinstance(v, tuple):
                typ, _ = v
            else:
                typ = v
            if typ in (int, float):
                yield prefix + [k]

    def to_hdf(self, filename: str, verbose=False):
        import h5py

        with h5py.File(filename, mode="w") as f:
            for key in ("/".join(_) for _ in self.walk_structure()):
                data = self.get(key)
                logger.info("Reading %s shape=%s", key, data.shape)

                f.create_dataset(key, data=data)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser("Convert yorick db to hdf5")
    parser.add_argument("fname", help="Yorick file to read")
    parser.add_argument("hdfname", help="HDF5 file to write")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    if args.verbose:
        logger.setLevel("DEBUG")
    p = PDBReader(args.fname)
    p.to_hdf(args.hdfname)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))  # pragma: no cover
