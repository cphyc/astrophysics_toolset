import re
from ..utilities.logging import logger

STRUCT_NAME_RE = re.compile('^struct (\w+) \{$')
ARRAY_RE = re.compile('^\s*(\w+) (\w+)\((\d+)\);$')
VAR_RE = re.compile('^\s*(\w+) (\w+);$')


class PDBReader:
    _known_types = {
        'int': int,
        'float': float,
        'pointer': 'yorick_pointer'
    }

    def __init__(self, fname):
        try:
            import pyorick
        except ModuleNotFoundError:
            logger.error('Reading PDB files requires the pyorick package. Install it via `pip install pyorick` along with Yorick.')

        if not self.check(fname):
            raise Exception('Unrecognized file format for file %s, could not read as PDB file.' % fname)

        self._known_types = self._known_types.copy()
        self.yo = pyorick.Yorick()
        self.yo(f'f=openb("{fname}")')

        self._variables = self._get_vars()
        self._structure = {}
        for v in self._variables:
            self._structure[v] = self._parse_struct(v)

    def check(self, fname):
        # Test that the file starts with the magic string
        with open(fname, 'br') as f:
            return f.readline().decode() == '!<<PDB:II>>!\n'

    def _get_vars(self):
        self.yo('ptrs=get_vars(f)')

        length = int(self.yo.e('numberof(ptrs)')) - 1
        variables = []
        for i in range(length):
            variables.append(*self.yo.e(f'*ptrs({i+1})'))
        return variables

    def _parse_struct(self, var_name):
        v = f'f.{var_name}'
        lines = self.yo.e(f'print(structof({v}))')

        # Now parse the structure
        remaining = lines[1:-1]  # last line is just }

        variables = {}
        for line in remaining:
            tmp = ARRAY_RE.match(line)
            if tmp:
                type_name, name, length = tmp.groups()
                length = int(length)
            else:
                type_name, name = VAR_RE.match(line).groups()
                length = 0

            if type_name not in self._known_types:
                new_type = self._parse_struct(f'{var_name}.{name}')
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
        # Check the patch does exist
        keys = path.split('/')
        node = self.structure
        for k in keys:
            node = node.get(k, None)
            if node is None:
                raise KeyError('Provided path does not exist on file: %s' % path)
        cmd = 'f.%s' % path.replace('/', '.')
        return self.yo(f'={cmd}')

    def __getitem__(self, key):
        return self.get(key)