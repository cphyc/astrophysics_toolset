from ..yorick import PDBReader


def test_reading():
    # This should work
    pdb = PDBReader('data/galaxies-040.pdb')

    def walker(keys, root):
        for key, val in root.items():
            if isinstance(val, dict):
                for _ in walker(keys + [key], val):
                    yield _
            else:
                yield keys + [key]

    # This should work
    for keys in walker([], pdb.structure):
        # Bake access path as "a/b/c"
        path = '/'.join(keys)
        pdb[path]
