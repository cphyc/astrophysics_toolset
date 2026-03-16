from collections import defaultdict
import h5py
import numpy as np
import yt
import argparse


def main():
    parser = argparse.ArgumentParser(description="Convert yt-supported dataset to Gadget HDF5 format")
    parser.add_argument("input", help="Input dataset (any supported by yt)")
    parser.add_argument("output", help="Output gadget dataset")
    parser.add_argument(
        "--mapping",
        type=str,
        nargs="+",
        default=["gas:PartType0", "DM_hires:PartType1", "star:PartType4"],
        help=(
            "Mapping from yt fields to Gadget particle types. "
            "Format is 'yt_field:PartTypeX'. Default is %(default)s."
        )
    )

    args = parser.parse_args()

    ds = yt.load(args.input)
    ad = ds.all_data()

    pos_units = "kpc/h/a"
    vel_units = "km/s*a**0.5"
    mass_units = "1e10*Msun/h"

    data_out = defaultdict(lambda: defaultdict(list))
    for ftype, ftype_out in (_.split(":") for _ in args.mapping):
        if ftype == "DM_hires":
            ftype = "DM"

            def mask_gen(pos, vel, ids, mass):
                mask = mass == mass.min()
                print(f"NDM={len(mask)}, NDM_hires={mask.sum()}")
                return mask
        else:

            def mask_gen(pos, vel, ids, mass):
                return slice(None)

        if ftype in ds.particle_types:
            # We have particles
            pos = ad[ftype, "particle_position"].to(pos_units).value
            vel = ad[ftype, "particle_velocity"].to(vel_units).value
            ids = ad[ftype, "particle_index"].astype(np.int64).value
            mass = ad[ftype, "particle_mass"].to(mass_units).value
            hsml = None
            try:
                hsml = ad[ftype, "smoothing_length"].to(pos_units).value
            except yt.utilities.exceptions.YTFieldNotFound:
                pass
        else:
            # We have gas
            pos = np.stack([ad[ftype, k] for k in "xyz"], axis=-1).to(pos_units).value
            vel = np.stack([ad[ftype, f"velocity_{k}"] for k in "xyz"], axis=-1).to(vel_units).value
            ids = np.arange(len(pos), dtype=np.int64)
            mass = ad[ftype, "cell_mass"].to(mass_units).value
            hsml = ad[ftype, "dx"].to(pos_units).value

        mask = mask_gen(pos, vel, ids, mass)
        data_out[ftype_out] = {
            "Coordinates": pos[mask],
            "Velocities": vel[mask],
            "ParticleIDs": ids[mask],
            "Masses": mass[mask],
        }
        if hsml is not None:
            data_out[ftype]["SmoothingLength"] = hsml[mask]

    with h5py.File(args.output, "w") as f:
        header = f.create_group("Header")

        counts = np.array(
            [len(data_out[f"PartType{i}"]["ParticleIDs"]) for i in range(6)],
            dtype=np.int32,
        )
        aexp = 1 / (1 + ds.current_redshift)

        mass_table = np.zeros(6)
        for ftype, data in data_out.items():
            if "Masses" in data:
                if np.ptp(data["Masses"]) == 0:
                    # If all masses are the same, store in the MassTable
                    # and remove from the data array
                    print(f"Detected uniform mass for {ftype=}")
                    mass_table[int(ftype[-1])] = data.pop("Masses")[0]

        header.attrs["NumPart_ThisFile"] = counts
        header.attrs["NumPart_Total"] = counts
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.int32)
        header.attrs["MassTable"] = mass_table
        header.attrs["Time"] = aexp
        header.attrs["Redshift"] = ds.current_redshift
        header.attrs["BoxSize"] = ds.domain_width[0].to("kpccm/h")
        header.attrs["Omega0"] = ds.omega_matter
        header.attrs["OmegaLambda"] = ds.omega_lambda
        header.attrs["HubbleParam"] = ds.hubble_constant
        header.attrs["NumFilesPerSnapshot"] = 1

        for ftype, data in data_out.items():
            g = f.create_group(ftype)
            for k, v in data.items():
                g.create_dataset(k, data=v)


if __name__ == "__main__":
    main()
