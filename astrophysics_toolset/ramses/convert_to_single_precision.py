from argparse import ArgumentParser
import os
from pathlib import Path
import numpy as np
import yt
from yt.config import ytcfg

from astrophysics_toolset.ramses._convert_to_single_precision import convert_grav, convert_hydro, convert_part

yt.enable_parallelism()

def get_unique_locations(field_offsets: dict[tuple[str, str], int]) -> list[tuple[str, str]]:
    unique_locs: list[tuple[int, tuple[str, str]]] = []
    seen_offsets = set()
    for field, offset in field_offsets.items():
        if offset not in seen_offsets:
            seen_offsets.add(offset)
            unique_locs.append((offset, field))

    unique_locs = sorted(unique_locs, key=lambda x: x[0])
    return [field for (_offset, field) in unique_locs]


def convert_file_descriptor(src: Path, dst: Path, blacklist: list[str]) -> None:
    with src.open("r") as fin, dst.open("w") as fout:
        # Copy two first lines
        fout.write(fin.readline())
        fout.write(fin.readline())

        for line in fin.readlines():
            ivar, var_name, dtype = (_.strip() for _ in line.split(","))
            if var_name in blacklist:
                out_dtype = dtype
            elif dtype == "d":
                out_dtype = "f"
            else:
                out_dtype = dtype

            fout.write(f"{ivar:3d}, {var_name}, {out_dtype}\n")


def convert(input_folder: Path, output_folder: Path, include_tracers: bool = False, verbose: bool = False) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    ds = yt.load(input_folder)
    dom = ds.index.domains[0]

    # List of files
    all_files = set(input_folder.glob("*"))

    # Convert hydro files
    hydro_files = sorted(input_folder.glob("hydro_*.out*"))
    hydro_handler = next(
        handler for handler in dom.field_handlers if handler.ftype == "ramses"
    )
    hydro_fields, _ = hydro_handler.get_detected_fields(ds)
    nvar = len(hydro_fields)
    for hydro_file in yt.parallel_objects(hydro_files):
        output_file = output_folder / hydro_file.name
        if not output_file.exists():
            convert_hydro(str(hydro_file), str(output_file), verbose=verbose, nvar_manual=nvar)
    all_files.difference_update(hydro_files)

    # Convert grav files
    grav_files = sorted(input_folder.glob("grav_*.out*"))
    grav_handler = next(
        handler for handler in dom.field_handlers if handler.ftype == "gravity"
    )
    grav_fields, _ = grav_handler.get_detected_fields(ds)
    nvar = len(grav_fields)
    for grav_file in yt.parallel_objects(grav_files):
        output_file = output_folder / grav_file.name
        if not output_file.exists():
            convert_grav(str(grav_file), str(output_file), verbose=verbose, nvar_manual=nvar)
    all_files.difference_update(grav_files)

    # Convert part files
    part_files = sorted(input_folder.glob("part_*.out*"))
    handler = next(handler for handler in ds.index.domains[0].particle_handlers if handler.ptype == "io")
    part_fields = get_unique_locations(handler.field_offsets)

    input_types = [handler.field_types[field] for field in part_fields]
    output_types = [
        "f"
        if handler.field_types[field] == "d" and not field[1].startswith("particle_position")
        else handler.field_types[field]
        for field in part_fields
    ]
    for part_file in yt.parallel_objects(part_files):
        output_file = output_folder / part_file.name
        if not output_file.exists():
            convert_part(str(part_file), str(output_file), include_tracers, input_types, output_types, verbose=verbose)
    all_files.difference_update(part_files)

    # Special treatment for the file descriptor.txt files
    hydro_file_desc = input_folder / "hydro_file_descriptor.txt"
    if hydro_file_desc.exists():
        all_files.remove(hydro_file_desc)
        tgt = output_folder / hydro_file_desc.name
        if yt.is_root():
            if verbose:
                print(f" Converting file descriptor: {hydro_file_desc} to {tgt}")
                convert_file_descriptor(hydro_file_desc, tgt, blacklist=[])

    part_file_desc = input_folder / "part_file_descriptor.txt"
    if part_file_desc.exists():
        all_files.remove(part_file_desc)
        tgt = output_folder / part_file_desc.name
        if yt.is_root():
            if verbose:
                print(f" Converting file descriptor: {part_file_desc} to {tgt}")
                convert_file_descriptor(part_file_desc, tgt, blacklist=[f"position_{k}" for k in "xyz"])

    # Hard link remaining files
    for remaining_file in yt.parallel_objects(sorted(all_files)):
        tgt = output_folder / remaining_file.name

        if tgt.exists():
            raise RuntimeError(f"{tgt} already exists! Aborting")
        if verbose:
            print(f" Copying file: {remaining_file} to {tgt}")
        os.link(remaining_file, tgt)


def verify(input_folder: Path, output_folder: Path):
    """Verify that the conversion from double to single precision was successful.

    This function checks that the hydro, grav, and part files in the output folder
    have the expected single precision formats compared to the input folder.

    Parameters
    ----------
    input_folder : Path
        Path to the original RAMSES output directory (double precision)
    output_folder : Path
        Path to the converted RAMSES output directory (single precision)

    Returns
    -------
    bool
        True if all files are verified successfully, False otherwise.
    """
    # Verification logic to be implemented
    yt.set_log_level(50)
    ds_old = yt.load(input_folder)
    for fh in ds_old.index.domains[0].field_handlers:
        fields, _single_precision = fh.get_detected_fields(ds_old)
        ytcfg[fh.config_field, "fields"] = [f"{field},f" for field in fields]
    ds_new = yt.load(output_folder)

    ad_old = ds_old.all_data()
    ad_new = ds_new.all_data()

    fields = [field for field in ds_old.field_list if field[0] != "all"]

    for field in yt.parallel_objects(sorted(fields)):
        try:
            np.testing.assert_allclose(ad_new[field], ad_old[field], rtol=1e-5, atol=1e-30)
            print(f"✓ Field {str(field):50s} matches between old and new datasets.")
        except AssertionError as e:
            print(f"✗ Field {str(field):50s} does not match: {e}")


def main(args=None):
    parser = ArgumentParser("Convert RAMSES output files to single precision")
    parser.add_argument("input", type=Path, help="Path to the RAMSES output directory")
    parser.add_argument("output", type=Path, help="Path to the output directory for single precision files")
    parser.add_argument("--include-tracers", action="store_true", help="Output contains tracer particles")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument("--verify", action="store_true", help="Verify that the conversion was successful")

    args = parser.parse_args(args)

    if args.verify:
        verify(args.input, args.output)
    else:
        convert(args.input, args.output, args.include_tracers, args.verbose)

    return 0


if __name__ == "__main__":
    args = [
        "/home/cphyc/Documents/prog/yt-data/output_00080",
        "/home/cphyc/Documents/prog/yt-data/single_precision/output_00080",
        "--verify",
    ]
    main(args)
