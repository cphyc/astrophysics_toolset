import yt

import argparse
import xarray as xr
from pathlib import Path
import numpy as np


def tracer(pfilter, data):
    return data[pfilter.filtered_type, "particle_family"] <= 0


yt.add_particle_filter(
    "tracer",
    function=tracer,
    filtered_type="io",
    requires=["particle_family"],
)

unit_mapping = {
    # Gas fields
    "x": "unitary",
    "y": "unitary",
    "z": "unitary",
    "velocity_x": "km/s",
    "velocity_y": "km/s",
    "velocity_z": "km/s",
    "mass": "Msun",
    "density": "mp/cm**3",
    "temperature": "K",
    # Particle fields
    "particle_position_x": "unitary",
    "particle_position_y": "unitary",
    "particle_position_z": "unitary",
    "particle_velocity_x": "km/s",
    "particle_velocity_y": "km/s",
    "particle_velocity_z": "km/s",
    "particle_mass": "Msun",
    "cell_gas_density": "mp/cm**3",
    "cell_gas_temperature": "K",
    "cell_gas_velocity_x": "km/s",
    "cell_gas_velocity_y": "km/s",
    "cell_gas_velocity_z": "km/s",
}


def main():
    yt.enable_parallelism()
    parser = argparse.ArgumentParser(
        description="Extract particle data from a yt dataset."
    )
    parser.add_argument("filename", type=str, nargs="+", help="Path to the dataset(s).")
    parser.add_argument(
        "output_folder",
        type=str,
        help="Folder where to save the extracted particle data.",
    )

    parser.add_argument(
        "--particle-type",
        default="tracer",
        choices=["star_tracer", "gas_tracer", "DM", "star", "tracer", "all"],
    )
    parser.add_argument(
        "--particle-fields",
        type=str,
        nargs="+",
        default=[
            "particle_position_x",
            "particle_position_y",
            "particle_position_z",
            "particle_velocity_x",
            "particle_velocity_y",
            "particle_velocity_z",
            "particle_mass",
        ],
        help="List of particle fields to extract.",
    )
    parser.add_argument(
        "--gas-fields",
        type=str,
        nargs="+",
        default=["density", "temperature", "velocity_x", "velocity_y", "velocity_z"],
        help="List of gas fields to extract.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    args = parser.parse_args()

    output = Path(args.output_folder)

    ts = yt.data_objects.api.DatasetSeries(args.filename)

    if args.verbose:
        yt.mylog.setLevel(10)

    for ds in ts.piter():
        fields = [(args.particle_type, f) for f in args.particle_fields]
        fields.append((args.particle_type, "particle_family"))
        fields.append((args.particle_type, "particle_identity"))

        # Add particle filter (if required)
        if args.particle_type == "tracer":
            ds.add_particle_filter("tracer")

        # Add mesh sampling particle fields
        for gas_field in args.gas_fields:
            ds.add_mesh_sampling_particle_field(
                ("gas", gas_field), ptype=args.particle_type
            )
            fields.append((args.particle_type, f"cell_gas_{gas_field}"))

        # Extract the fields
        ad = ds.all_data()

        ad.get_data(fields)

        data = {}
        order = np.argsort(ad[args.particle_type, "particle_identity"].astype("int64"))
        for ftype, fname in fields:
            if fname == "particle_identity":
                # Special case for particle identity
                data["idp"] = ad[ftype, fname][order].astype("int64").d
                continue

            # Get custom unit if available
            unit = unit_mapping.get(fname, str(ad[ftype, fname].units))
            dtype = ad[ftype, fname].dtype

            # Convert to float32
            if fname in (
                "x",
                "y",
                "z",
                "particle_position_x",
                "particle_position_y",
                "particle_position_z",
            ):
                dtype = "float64"
            elif fname in ("particle_family",):
                dtype = "int32"
            else:
                dtype = "float32"

            data[f"{ftype}_{fname}"] = (
                "idp",
                ad[ftype, fname][order].to(unit).astype(dtype).d,
                {"units": unit},
            )

        data["t"] = float(ds.current_time.to("Gyr"))
        data["redshift"] = float(ds.current_redshift)

        xr_ds = xr.Dataset(data)

        output_XXXXX = Path(ds.filename).parent.name
        nc_filename = output / f"{output_XXXXX}_particles.nc"

        xr_ds.to_netcdf(nc_filename)
