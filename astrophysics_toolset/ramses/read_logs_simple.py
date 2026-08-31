import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from labellines import labelLines

parser = argparse.ArgumentParser(description="Read logs from a file.")

parser.add_argument("logfile", type=str, nargs="+", help="Path to the log file(s) to read.")

args = parser.parse_args()

data = [("", 0, 0)]
for logfile in args.logfile:
    with open(logfile) as f:
        aexp = -1
        ttot = -1
        for line in f:
            line = line.strip()

            if line.startswith("Total running time"):
                ttot = float(line.split(":")[1].strip().split()[0])

            if " a= " in line:
                aexp = float(line.split(" a=")[1].split()[0])

    data.append((logfile, aexp, ttot))
    print(".", end="", flush=True)

df = pd.DataFrame(data, columns=["logfile", "aexp", "ttot"]).sort_values(by="aexp")

df["walltime_cumulative"] = df["ttot"].cumsum()

df["walltime_cumulative_days"] = df["walltime_cumulative"] / (3600 * 24)

print(df)

plt.style.use("paper-onecolumn")

fig, ax = plt.subplots()
ax.plot(df["walltime_cumulative_days"], df["aexp"], marker=".", c="k")

# Interpolate all but first point
popt, pcov = np.polyfit(df["walltime_cumulative_days"][2:], df["aexp"][2:], 1, cov=True)

xlines, ylines = [], []
# Find intersection with aexp=0.2 (z=4)
for target in (
    1 / (1 + 4),
    1 / (1 + 3),
):
    intersection_x = (target - popt[1]) / popt[0]
    z = (1 / target) - 1
    xlines.append(
        ax.axvline(
            intersection_x,
            color="gray",
            linestyle=":",
            zorder=-1,
            label=f"$t={intersection_x:.1f}$",
            lw=0.5,
        )
    )
    ylines.append(
        ax.axhline(
            target,
            color="gray",
            linestyle=":",
            zorder=-1,
            label=f"$z={z:.0f}$",
            lw=0.5,
        )
    )
labelLines(xlines, align=False)
labelLines(ylines, xvals=[1, 1], align=False)


# Plot the linear fit
xx = np.linspace(1, intersection_x + 0.8, 20)
ax.plot(xx, np.polyval(popt, xx), color="gray", linestyle="--", alpha=0.5, lw=1)


ax.set(ylabel="aexp", xlabel="Walltime [day]", xlim=(-0.1, xx.max()), ylim=(-0.01, None))

fig.savefig("aexp_vs_time.pdf")
