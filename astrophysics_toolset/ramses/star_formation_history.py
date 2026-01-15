import argparse
from pathlib import Path
from textwrap import indent
import yt
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("inputs", type=str, nargs="+", help="Input file path")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--iout", type=int, help="Target output index")
    grp.add_argument("--redshift", type=float, help="Target redshift")
    grp.add_argument("--time", type=float, help="Target time in Gyr")

    parser.add_argument("--output", type=str, default="sfh.pdf", help="Output plot filename")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--dt-SFR", type=float, default=5, help="Time bin size for SFR calculation in Myr")

    args = parser.parse_args()

    if args.verbose:
        yt.set_log_level(10)  # INFO

    all_ts = [yt.load(Path(inp) / "output_?????") for inp in args.inputs]

    if args.iout is not None:
        target_ds = []
        for ts in all_ts:
            _outputs = [Path(_).name for _ in ts.outputs]
            _out = f"output_{args.iout:05d}"

            if _out not in _outputs:
                my_str = indent("\n".join(_outputs), "    ")
                raise IndexError(f"Output '{_out}' not found in {ts} outputs. Possible values are: {my_str}")
            ind = _outputs.index(_out)
            target_ds.append(ts[ind])
    elif args.redshift is not None:
        target_ds = [ts.find_closest("redshift", args.redshift) for ts in all_ts]
    elif args.time is not None:
        target_ds = [ts.find_closest("time", args.time) for ts in all_ts]

    # Now get SFH
    tborn, masses = {}, {}
    for ds, inp in zip(target_ds, args.inputs):
        ad = ds.all_data()
        ad.get_data([("star", "particle_birth_mass"), ("star", "particle_birth_time")])
        masses[inp] = ds.arr(ad["star", "particle_birth_mass"].d, "code_mass").to("Msun").d
        tborn[inp] = ad["star", "particle_birth_time"].to("Gyr").d

    dt = args.dt_SFR / 1000.0  # Convert Myr to Gyr
    tmin, tmax = min(tborn[inp].min() for inp in args.inputs), max(tborn[inp].max() for inp in args.inputs)
    tmin = float(np.floor(tmin / dt) * dt)  # Round down to nearest dt
    tmax = float(np.ceil(tmax / dt) * dt)  # Round up to nearest dt
    bins = np.arange(tmin, tmax + dt, dt)

    fig, ax = plt.subplots(constrained_layout=True)
    for inp in args.inputs:
        hist, bin_edges = np.histogram(tborn[inp], bins=bins, weights=masses[inp])
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        sfr = hist / dt / 1e9  # SFR in Msun per year
        ax.plot(bin_centers, sfr, label=Path(inp).name)

    ax.set(
        xlabel=r"Cosmic time [$\rm Gyr$]",
        ylabel=rf"$\mathrm{{SFR}}_{{{dt * 1000:.0f}\rm Myr}}$ [$\rm M_{{\odot}}/yr$]",
        yscale="log",
    )

    ax.legend()
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
