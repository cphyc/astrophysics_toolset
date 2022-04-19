from shutil import which

import numpy as np
import pytest

from ..yorick import PDBReader

SKIP_YORICK = which("yorick") is None
SKIP_YORICK_REASON = "Could not find yorick executable. Skipping tests."


@pytest.fixture
def pdb():
    return PDBReader("data/galaxies-040.pdb")


@pytest.fixture
def pdb_array():
    return PDBReader("data/toto.pdb")


def walker_helper(keys, root):
    for key, val in root.items():
        if isinstance(val, dict):
            yield from walker_helper(keys + [key], val)
        else:
            if isinstance(val, tuple):
                exp_type = val[0]
            else:
                exp_type = val
            yield keys + [key], exp_type


def deep_dict_compare(a, b, prefix=None, print_pre=""):
    if prefix is None:
        prefix = []
    assert len(a) == len(b)

    key_a = set(a.keys())
    key_b = set(b.keys())

    if len(key_a.difference(key_b)) > 0:
        raise AssertionError("Found different set of keys at %s" % prefix)

    for key, va in a.items():
        print(print_pre, end="")
        vb = b[key]
        if type(va) != type(vb):
            raise AssertionError(f"Type differ at {prefix}[{key}]")

        if isinstance(va, dict):
            print("checking key «%s»" % key)
            deep_dict_compare(va, vb, prefix + [key], print_pre + "\t")
        else:
            print(f"comparing {prefix} {key}", end="...")
            if va != vb:
                raise AssertionError(f"Value differ at {prefix}[{key}], {va} ≠ {vb}")
            print("ok!")


@pytest.mark.skipif(SKIP_YORICK, reason=SKIP_YORICK_REASON)
def test_structure(pdb):
    expected_structure = {
        "catalog": {
            "vir": {"rvir": float, "mvir": float, "tvir": float, "cvel": float},
            "shape": (float, 3),
            "pos": (float, 3),
            "vel": (float, 3),
            "L": (float, 3),
            "profile": (float, 2),
            "age": float,
            "aexp": float,
            "num": int,
            "slice": (int, 2),
            "level": int,
            "hosthalo": int,
            "hostsub": int,
            "nbsub": int,
            "nextsub": int,
            "mass": float,
            "rad": float,
            "spin": float,
            "ek": float,
            "ep": float,
            "et": float,
            "macc": float,
            "tree": {
                "nbfather": int,
                "father": "yorick_pointer",
                "mfrac": "yorick_pointer",
                "nbson": int,
                "son": "yorick_pointer",
            },
            "npart": int,
            "index": "yorick_pointer",
            "bulge": (float, 3),
        }
    }
    deep_dict_compare(pdb.structure, expected_structure)


@pytest.mark.skipif(SKIP_YORICK, reason=SKIP_YORICK_REASON)
def test_access(pdb):
    # This should work
    for keys, expected_type in walker_helper([], pdb.structure):
        # Bake access path as "a/b/c"
        path = "/".join(keys)
        res = pdb[path]

        if expected_type in (int, float):
            assert isinstance(res, np.ndarray)


@pytest.mark.skipif(SKIP_YORICK, reason=SKIP_YORICK_REASON)
def test_access_incorrect(pdb):
    with pytest.raises(KeyError):
        pdb["this/does/not/exists"]


@pytest.mark.skipif(SKIP_YORICK, reason=SKIP_YORICK_REASON)
def test_read_arrays(pdb_array):
    assert pdb_array["rg"].shape == (151,)
    assert pdb_array["dens"].shape == (4, 151)
