import re

import matplotlib.pyplot as plt
import pandas as pd

# from astrophysics_toolset.utilities.logging import logger
from yt import mylog as logger

# Match 'Restarting at t=  -5.29607640528895       nstep_coarse=       46763'
RESTART_RE = re.compile(
    r"Restarting at t=\s*(?P<t>[-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]+)\s*"
    r"nstep_coarse=\s*(?P<nstep_coarse>\d+)"
)

# Match 'Level 18 has     495862 grids (       0,     880,     121,)'
LEVEL_RE = re.compile(
    r"Level\s*(?P<level>\d+) has\s+(?P<grids>\d+) grids "
    r"\(\s*(?P<Nmin>\d+),\s*(?P<Nmax>\d+),\s*(?P<Nmean>\d+),\)"
)

# Match 'Fine step=531884 t=-5.29608E+00 dt= 4.621E-06 a= 1.456E-01 mem=10.3%  7.4%'
FINE_RE = re.compile(
    r"Fine step=\s*(?P<fine_step>\d+) t=\s*(?P<t>[-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]+) "
    r"dt=\s*(?P<dt>[-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]+) "
    r"a=\s*(?P<a>[-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]+) "
    r"mem=\s*(?P<mem_amr>[-+]?[0-9]*\.?[0-9]+)%\s*(?P<mem_part>[-+]?[0-9]*\.?[0-9]+)%"
)

# Match 'Time elapsed since last coarse step:  398.91 s     1526.97 mus/pt  [...]'
TIME_RE = re.compile(
    r"Time elapsed since last coarse step:\s*(?P<time_elapsed>\d+\.\d+) s"
    r"\s*(?P<micros_per_pt>\d+\.\d+) mus/pt\s*"
    r"\s*(?P<micros_per_pt_av>\d+\.\d+) mus/pt \(av\)"
)


def read_log_file(log_file: str):
    logger.info("Reading log file %s", log_file)
    current_coarse_timestep: int
    current_fine_timestep: int

    level_stats: dict[tuple[int, int], dict[str, int]] = {}
    fine_step_stats: dict[tuple[int, int], dict[str, float]] = {}
    coarse_step_stats: dict[int, dict[str, float]] = {}

    with open(log_file) as f:
        for line in f:
            line = line.strip()
            # Initial timestep
            if match := RESTART_RE.match(line):
                data = match.groupdict()
                current_coarse_timestep = int(data["nstep_coarse"])
            elif match := LEVEL_RE.match(line):
                data = match.groupdict()
                level = int(data.pop("level"))
                level_stats[current_coarse_timestep, level] = {
                    k: int(v) for k, v in data.items()
                }
            elif match := FINE_RE.match(line):
                data = match.groupdict()
                current_fine_timestep = int(data["fine_step"])
                fine_step_stats[current_coarse_timestep, current_fine_timestep] = {
                    k: float(v) for k, v in data.items()
                }
            elif match := TIME_RE.match(line):
                data = match.groupdict()
                current_coarse_timestep += 1
                coarse_step_stats[current_coarse_timestep] = {
                    k: float(v) for k, v in data.items()
                }

    level_stats_df = pd.DataFrame(level_stats).T
    level_stats_df.index.names = ["nstep_coarse", "level"]

    fine_step_stats_df = pd.DataFrame(fine_step_stats).T
    fine_step_stats_df.index.names = ["nstep_coarse", "fine_step"]

    coarse_step_stats_df = pd.DataFrame(coarse_step_stats).T
    coarse_step_stats_df.index.names = ["nstep_coarse"]

    return level_stats_df, fine_step_stats_df, coarse_step_stats_df


def plot_level_stats(level_stats: pd.DataFrame):
    logger.info("Plotting level stats")

    with plt.style.context("paper-twocolumns"):
        fig, axes = plt.subplots(ncols=2, constrained_layout=True, figsize=(6, 2))
        level_max = level_stats.index.get_level_values("level").max()
        for level, sub in level_stats.reset_index().groupby("level"):
            if level < level_max - 7:
                continue
            axes[0].plot(sub["nstep_coarse"], sub["grids"], label=f"Level {level}")

            # Plot memory imbalance
            mem_imbalance = sub["Nmax"] / sub["Nmean"]
            axes[1].plot(sub["nstep_coarse"], mem_imbalance)

        axes[0].set(
            xlabel="Coarse timestep",
            ylabel="Number of grids",
            yscale="log",
        )
        axes[1].set(
            xlabel="Coarse timestep",
            ylabel="Memory imbalance",
            yscale="log",
        )
        axes[0].legend(fontsize="x-small", ncol=2)

        fig.savefig("level_stats.pdf")


def plot_time_per_timestep(coarse_step_stats: pd.DataFrame):
    logger.info("Plotting time per timestep")
    with plt.style.context("paper-onecolumn"):
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(
            coarse_step_stats.index,
            coarse_step_stats["time_elapsed"],
            label="Time elapsed",
        )
        ax.set(
            xlabel="Coarse timestep",
            ylabel="Time per coarse timestep (s)",
        )

        fig.savefig("time_per_timestep.pdf")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read RAMSES log files")
    parser.add_argument(
        "log_file",
        type=str,
        help="Path(s) to the log file(s)",
        nargs="+",
    )
    parser.add_argument(
        "--plot-level-stats",
        action="store_true",
        help="Plot the number of grids for each level",
    )
    parser.add_argument(
        "--plot-time-per-timestep",
        action="store_true",
        help="Plot the time per timestep",
    )

    args = parser.parse_args()

    logger.setLevel("DEBUG")

    level_stats, find_step_stats, coarse_step_stats = (
        pd.concat(_)
        for _ in zip(*(read_log_file(log_file) for log_file in sorted(args.log_file)))
    )

    if args.plot_level_stats:
        plot_level_stats(level_stats)

    if args.plot_time_per_timestep:
        plot_time_per_timestep(coarse_step_stats)
