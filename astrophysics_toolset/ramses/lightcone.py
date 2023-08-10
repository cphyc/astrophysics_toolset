import argparse
import os
import sys
from collections import defaultdict
from math import ceil

import numpy as np
import pandas as pd
import yt
from cython_fortran_file import FortranFile


def read_lightcones(
    input_folder: str, iout: int, ncpu: int, *, kind: str, metal: bool, nchem: int
):
    data = defaultdict(list)
    for icpu in range(ncpu):
        filename = os.path.join(
            input_folder, f"cone_{iout:05d}", f"cone_{kind}_{iout:05d}.out{icpu:05d}"
        )

        if not os.path.exists(filename + ".txt"):
            continue

        with open(filename + ".txt") as f:
            ncpu2 = int(f.readline())
            nstride = int(f.readline())
            npart = int(f.readline())

            if ncpu != ncpu2:
                raise RuntimeError(
                    f"Number of CPUs mismatch: {ncpu} != {ncpu2} for {filename}"
                )

        nbloc = int(ceil(npart / nstride))
        # print(npart, nstride, ncpu2, nbloc)

        with FortranFile(filename, "r") as f:
            # Skip headers
            f.read_int()
            f.read_int()
            f.read_int()

            records = [
                "x",
                "vx",
                "_skip",
                "y",
                "vy",
                "_skip",
                "z",
                "vz",
                "_skip",
                "redshift",
                "mass",
            ]
            if kind == "stars":
                records += ["age"]
            elif kind == "gas":
                records += ["dens", "temp"]

            if metal:
                records += ["metal", *(f"chem{i}" for i in range(nchem))]

            for _bloc in range(nbloc):
                for key in records:
                    value = f.read_vector("f")
                    if key.startswith("_"):
                        continue
                    data[key].append(value)

            inow = f.tell()
            # Seek to the end of the file
            f.seek(0, 2)
            iend = f.tell()

            if not inow == iend:
                raise RuntimeError(f"File {filename} not fully read: {inow} != {iend}")
            yt.mylog.debug(filename)
            yt.mylog.debug("xp(1:5, 1)=%s", data["x"][-1][:5])
            yt.mylog.debug("xp(1:5, 2)=%s", data["y"][-1][:5])
            yt.mylog.debug("xp(1:5, 3)=%s", data["z"][-1][:5])
            yt.mylog.debug("vp(1:5, 1)=%s", data["vx"][-1][:5])
            yt.mylog.debug("vp(1:5, 2)=%s", data["vy"][-1][:5])
            yt.mylog.debug("vp(1:5, 3)=%s", data["vz"][-1][:5])
            yt.mylog.debug("zp(1:5)=%s", data["redshift"][-1][:5])
            yt.mylog.debug("mp(1:5)=%s", data["mass"][-1][-5:])
            yt.mylog.debug("age(1:5)=%s", data["age"][-1][:5])
            yt.mylog.debug("Zmet(1:5)=%s", data["metal"][-1][:5])

    return {k: np.concatenate(v) for k, v in data.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read lightcone data")

    parser.add_argument("input", type=str, help="Input folder")
    parser.add_argument("output", type=str, help="Output folder")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--star", action="store_true", help="Read star data")
    grp.add_argument("--gas", action="store_true", help="Read gas data")

    parser.add_argument("--metal", action="store_true", help="Read metallicity")
    parser.add_argument(
        "--nchem", type=int, help="Number of chemical species", default=0
    )

    parser.add_argument("--ncpu", type=int, help="Number of CPUs to use", required=True)
    parser.add_argument("--nmin", type=int, help="Minimum output", required=True)
    parser.add_argument("--nmax", type=int, help="Maximum output", required=True)

    args = parser.parse_args(argv)

    yt.enable_parallelism()

    all_data = {}
    from yt.utilities.parallel_tools.parallel_analysis_interface import (
        communication_system,
    )

    comm_size = communication_system.communicators[-1].size
    prog = yt.get_pbar(
        "Reading lightcone data", (args.nmax - args.nmin + 1) // comm_size
    )
    for sto, iout in yt.parallel_objects(
        range(args.nmin, args.nmax + 1), storage=all_data
    ):
        if args.star:
            data = read_lightcones(
                args.input,
                iout,
                args.ncpu,
                kind="stars",
                metal=args.metal,
                nchem=args.nchem,
            )
        elif args.gas:
            data = read_lightcones(
                args.input,
                iout,
                args.ncpu,
                kind="gas",
                metal=args.metal,
                nchem=args.nchem,
            )

        df = pd.DataFrame(data)

        sto.result_id = iout
        sto.result = df

        prog.update()

    if not yt.is_root():
        return 0

    df = pd.concat(list(all_data.values()), ignore_index=True).loc[
        :, "x y z vx vy vz redshift mass age metal".split()
    ]
    df.to_hdf(
        args.output,
        "data",
    )

    yt.mylog.info(f"Successfully wrote {len(df)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
