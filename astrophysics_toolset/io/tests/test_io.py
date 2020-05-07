import pytest
import numpy as np

from ..yorick import PDBReader


@pytest.fixture
def pdb():
    return PDBReader('data/galaxies-040.pdb')


def walker_helper(keys, root):
    for key, val in root.items():
        if isinstance(val, dict):
            for _ in walker_helper(keys + [key], val):
                yield _
        else:
            if isinstance(val, tuple):
                exp_type = val[0]
            else:
                exp_type = val
            yield keys + [key], exp_type


def deep_dict_compare(a, b, prefix=[]):
    assert len(a) == len(b)

    key_a = set(a.keys())
    key_b = set(b.keys())

    if len(key_a.difference(key_b)) > 0:
        raise AssertionError("Found different set of keys at %s" % prefix)

    for key, va in a.items():
        vb = b[key]
        if type(va) != type(vb):
            raise AssertionError('Type differ at %s[%s]' % (prefix, key))

        if isinstance(va, dict):
            deep_dict_compare(va, vb, prefix + [key])
        else:
            if va != vb:
                raise AssertionError(
                    "Value differ at %s[%s], %s ≠ %s" % (prefix, key, va, vb))


def test_structure(pdb):
    expected_structure = {
        'catalog': {
            'vir': {'rvir': float,
                    'mvir': float,
                    'tvir': float,
                    'cvel': float
            },
            'shape': (float, 3),
            'pos': (float, 3),
            'vel': (float, 3),
            'L': (float, 3),
            'profile': (float, 2),
            'age': float,
            'aexp': float,
            'num': int,
            'slice': (int, 2),
            'level': int,
            'hosthalo': int,
            'hostsub': int,
            'nbsub': int,
            'nextsub': int,
            'mass': float,
            'rad': float,
            'spin': float,
            'ek': float,
            'ep': float,
            'et': float,
            'macc': float,
            'tree': {
                'nbfather': int,
                'father': 'yorick_pointer',
                'mfrac': 'yorick_pointer',
                'nbson': int,
                'son': 'yorick_pointer'
            },
            'npart': int,
            'index': 'yorick_pointer',
            'bulge': (float, 3)
        }
    }
    deep_dict_compare(pdb.structure, expected_structure)


def test_access(pdb):
    # This should work
    for keys, expected_type in walker_helper([], pdb.structure):
        # Bake access path as "a/b/c"
        path = keys
        res = pdb[path]

        if expected_type in (int, float):
            assert isinstance(res, np.ndarray)


def test_access_incorrect(pdb):
    with pytest.raises(KeyError):
        pdb['this/does/not/exists']
