# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] Collapsed="false"
# Here, I try to setup some way of measuring the intrinsic number of dimensions in some data. The data resides on a manifold of dimension $N$, embedded in a continuous $M$-dimensional space. The question is: how to find $N$?

# + Collapsed="false" tags=[]
from typing import Optional

import black
import joblib
import jupyter_black
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from astropy.io import fits
from IPython.display import Markdown as md
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.spatial import cKDTree as KDTree
from sklearn import datasets
from tqdm.auto import tqdm

from astrophysics_toolset.io.yorick import PDBReader
from astrophysics_toolset.utilities.logging import logger

# from sklearn.neighbors import KDTree


jupyter_black.load(
    lab=True,
    line_length=100,
    verbosity="DEBUG",
    target_version=black.TargetVersion.PY310,
)

plt.rcParams["figure.dpi"] = 180


# + Collapsed="false" tags=[]
def _find_dimensionality_helper(
    *, inode: int, G: nx.Graph, distances: np.ndarray
) -> np.ndarray:
    N = np.zeros(len(distances))
    Gprev = G
    for i, d in enumerate(distances):
        Gprev = nx.ego_graph(Gprev, inode, radius=d, distance="weight")
        N[i] = len(Gprev.nodes)

    return N


def find_dimensionality(
    data: np.ndarray,
    Nsample: int = 100,
    bins: Optional[np.ndarray] = None,
    use_MST: bool = False,
    min_neigh: int = 5,
    n_jobs=-1,
):
    """Find the dimensionality of the manifold on which data reside.

    Parameters
    ----------
    data : ndarray, shape (n, m)
        n samples of m features
    Nsample : int
        The number of samples to draw to estimate the dimensionality.
    bins : array
        The distance arrays to use when computing the distance
    use_MST : bool
        If True, use a Minimum Spanning Tree instead of the full graph. May be faster, but potentially lead
        to wrong answer
    min_neigh : int
        The minimum number of nearest-neighbours to build the graph.

    Returns
    -------
    N : array (Nsample, Nbins)
        The number of elements in the graph
    distances : array (Nbins, )
        The distances.

    Notes
    -----
    1. The dimensionality can be obtained by computing dlog N/dlog distances, which should be an integer number.
    2. It is highly advised to scale the input data so that the dynamical range of all features is the same, preferably
       between 0 and 1.

    The algorithm works as follows:
    1. A nearest-neighbour graph is computed, with a number of neighbours large enough so that the graph is fully connected.
    2. Optionaly, the graph is reduced using a minimum-spanning tree approach.
    3. Randomly selected points are drawn from the graph and the number of connected points within
       some given distance (along graph edges) is computed. This number scales as r^dimensionality.

    """
    logger.info("Building KD-tree")
    tree = KDTree(data)

    logger.info("Finding nearest neighbours")
    n_neigh_max = 30

    d, ineigh = tree.query(
        tree.data, k=n_neigh_max + 1, workers=-1
    )  # +1 since the 0-th is itself

    logger.info("Building graph")
    G = nx.Graph()
    G.add_nodes_from((i, {"pos": data[i]}) for i in ineigh[:, 0])

    logger.info("Adding edges until graph is connected")
    connected = False
    do_once_more = min_neigh == 1
    i = min_neigh
    while not connected or do_once_more:
        edges = (
            (a, b, d) for a, b, d in zip(ineigh[:, 0], ineigh[:, i + 1], d[:, i + 1])
        )
        G.add_weighted_edges_from(edges)

        if do_once_more:
            logger.debug("n_neigh%s: ✔", i)
            break

        if nx.connected.number_connected_components(G) < 5:
            logger.debug("n_neigh%s: ✔", i)
            connected = True
            do_once_more = True
            break
            # i *= 2
        else:
            logger.debug("n_neigh%s: ✖", i)
            i += 1

    if use_MST:
        logger.info("Computing minimum spanning tree")
        GMST = nx.algorithms.minimum_spanning_tree(G)
        G = GMST

    logger.info("Computing number of point as a function of distance")
    # Use decreasing distance, so that the ego graph at step i+1 is fully contained in the one at step i (huge improvement!)
    distances = bins if bins is not None else np.geomspace(0.01, 2, 40)

    # Sort by decreasing distances
    distances = np.sort(distances)[::-1]

    N = np.zeros((len(distances), Nsample))

    data = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_find_dimensionality_helper)(
            inode=inode,
            G=G,
            distances=distances,
        )
        for inode in tqdm(np.random.choice(len(data), size=Nsample, replace=False))
    )

    for istep, dt in enumerate(data):
        N[:, istep] += dt

    return N, distances


# + Collapsed="false" tags=[]
def normalize_log(a):
    b = np.log10(a)
    return normalize_lin(b)


def normalize_lin(a):
    mask = np.isfinite(a)
    amin, amax = np.min(a[mask]), np.max(a[mask])
    tmp = (a - amin) / (amax - amin)
    return np.where(np.isfinite(tmp), tmp, 0)


# + [markdown] Collapsed="true" jp-MarkdownHeadingCollapsed=true tags=[]
# # Simple case: the U-shape

# + Collapsed="false" tags=[]
s = np.random.rand(100_000) * np.pi
x = np.cos(s)
y = 1 - np.sin(s)
z = np.cos(2 * np.pi * s)

# Add some noise
mu = [0, 0]
ss = 0.0001
Sigma = np.array([ss, ss / 4, ss / 4, ss]).reshape(2, 2)
noise = np.random.multivariate_normal(mu, Sigma, size=x.size)
x += noise[:, 0]
y += noise[:, 1]

data = np.stack([x, y, z], axis=-1)
plt.scatter(x, y, s=1)
# -

# ## Without using minimum-spanning tree

# + Collapsed="false" tags=[]
N, distances = find_dimensionality(data, Nsample=20)
dN_dd = np.gradient(np.log(N), axis=0) / np.gradient(np.log(distances))[:, None]

# + Collapsed="false"
mu = dN_dd.mean(axis=1)
std = dN_dd.std(axis=1)
fig = plt.figure(figsize=(10, 5), dpi=180)

ax = fig.add_subplot(121, projection="3d")
ax.scatter(data[::100, 0], data[::100, 1], data[::100, 2], s=1, marker="o")
ax.view_init(30, 60)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
for k in "xyz":
    ticks = np.array([-1, -0.5, 0, 0.5, 1])
    vmin, vmax = getattr(ax, f"get_{k}lim")()
    getattr(ax, f"set_{k}ticks")(ticks[(ticks >= vmin) & (ticks <= vmax)])
plt.title("1% of the datapoints")

fig.add_subplot(122)
# plt.axvline(np.sqrt(Sigma[0,0]))
plt.plot(distances, mu, "-+")
plt.fill_between(distances, mu - std, mu + std, alpha=0.1)
plt.grid(True)

plt.xlabel("Graph distance")
plt.ylabel("Dimensionality")
plt.tight_layout()

# S-curve (from scikit)
# -

# ## Using minimum spanning tree

logger.setLevel(20)

# + Collapsed="false" tags=[]
N, distances = find_dimensionality(data, Nsample=20, use_MST=True)
dN_dd = np.gradient(np.log(N), axis=0) / np.gradient(np.log(distances))[:, None]

# + Collapsed="false"
mu = dN_dd.mean(axis=1)
std = dN_dd.std(axis=1)
fig = plt.figure(figsize=(10, 5), dpi=180)

ax = fig.add_subplot(121, projection="3d")
ax.scatter(data[::100, 0], data[::100, 1], data[::100, 2], s=1, marker="o")
ax.view_init(30, 60)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
for k in "xyz":
    ticks = np.array([-1, -0.5, 0, 0.5, 1])
    vmin, vmax = getattr(ax, f"get_{k}lim")()
    getattr(ax, f"set_{k}ticks")(ticks[(ticks >= vmin) & (ticks <= vmax)])
plt.title("1% of the datapoints")

fig.add_subplot(122)
# plt.axvline(np.sqrt(Sigma[0,0]))
plt.plot(distances, mu, "-+")
plt.fill_between(distances, mu - std, mu + std, alpha=0.1)
plt.grid(True)

plt.xlabel("Graph distance")
plt.ylabel("Dimensionality")
plt.tight_layout()

# S-curve (from scikit)

# + nteract={"transient": {"deleting": false}} tags=[]
# !mkdir -p plots/

# + nteract={"transient": {"deleting": false}} tags=[]
X, color = datasets.make_s_curve(n_samples=200_000, noise=0.01)
x, y, z = X.T

# + Collapsed="false" tags=[]
data = np.stack([x, y, z], axis=-1)
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(121, projection="3d")
ax.scatter(x[::10], y[::10], z[::10], s=1, c=color[::10])

ax = plt.subplot(122)
ax.hist2d(x[::1], y[::1], bins=128)
plt.savefig("plots/s-curve_2panels.pdf")

# + Collapsed="false"
N, distances = find_dimensionality(data, Nsample=10, n_jobs=3)
dN_dd = np.gradient(np.log(N), axis=0) / np.gradient(np.log(distances))[:, None]

# + Collapsed="false"
mu = dN_dd.mean(axis=1)
std = dN_dd.std(axis=1)
fig = plt.figure(figsize=(10, 5), dpi=180)

ax = fig.add_subplot(121, projection="3d")
sl = slice(None, None, 10)
ax.scatter(data[sl, 0], data[sl, 1], data[sl, 2], s=1, marker="o", c=color[sl])
ax.view_init(30, 60)
ax.set(
    xlabel="$x$",
    ylabel="$y$",
    zlabel="$z$",
)
for k in "xyz":
    ticks = np.array([-1, -0.5, 0, 0.5, 1])
    vmin, vmax = getattr(ax, f"get_{k}lim")()
    getattr(ax, f"set_{k}ticks")(ticks[(ticks >= vmin) & (ticks <= vmax)])
plt.title("10% of the datapoints")

ax = fig.add_subplot(122)
# plt.axvline(np.sqrt(Sigma[0,0]))
ax.plot(distances, mu, "-+")
ax.fill_between(distances, mu - std, mu + std, alpha=0.1)
ax.grid(True)

ax.set(xlabel="Graph distance", ylabel="Dimensionality", xscale="log")
plt.tight_layout()

# + [markdown] Collapsed="true" jp-MarkdownHeadingCollapsed=true tags=[]
# # Data from Horizon-AGN

# + Collapsed="false"
# !rsync -azz --progress infinity:"/data65/laigle/Hz-AGNCub/galaxies-782.pdb" tmp

# + Collapsed="false"
p = PDBReader("tmp/galaxies-782.pdb")


# + Collapsed="false"
def walk(node, prefix=None):
    if prefix is None:
        prefix = []
    for k, v in node.items():
        if isinstance(v, dict):
            yield from walk(v, prefix + [k])
            continue

        if isinstance(v, tuple):
            typ, _ = v
        else:
            typ = v
        if typ in (int, float):
            yield prefix + [k]


# + Collapsed="false"
dt = {}
for k in ("/".join(_) for _ in walk(p.structure)):
    tmp = p[k]
    if p[k].ndim > 1 and p[k].shape[1] == 3:
        for idim, dim in enumerate("xyz"):
            dt[f"{k}_{dim}"] = p[k][:, idim]
    elif p[k].ndim == 1:
        dt[k] = p[k]

# + Collapsed="false"
df = pd.DataFrame(dt)
ok_columns = df.max() != df.min()

# + Collapsed="false"
all_data = df.loc[:, ok_columns]

q1, q5, q9 = all_data.quantile(q=[0.1, 0.5, 0.9]).values

# + Collapsed="false"
all_data.apply(normalize_lin, axis=0).describe()

# + Collapsed="false"
data = np.stack(
    [
        normalize_log(np.abs(dt[f"catalog/{key}"]))
        for key in (
            "vir/rvir",
            "vir/mvir",
            "vir/tvir",
            "vir/cvel",
            "spin",
            "ek",
            "ep",
            "et",
        )
    ],
    axis=-1,
)

# + Collapsed="false"
plt.hist(
    data,
    bins=200,
    histtype="step",
    label=r"$R_{\mathrm{vir}}$ $M_{\mathrm{vir}}$ $T_{\mathrm{vir}}$ $v_{\mathrm{vir}}$ $\lambda$ $E_\mathrm{k}$ $E_\mathrm{p}$ $E_\mathrm{t}$".split(),
)
plt.legend()
plt.xlabel("Normalized data")

# + Collapsed="false"
N, d = find_dimensionality(data, Nsample=99, bins=np.geomspace(8e-3, 0.5, 10), n_jobs=8)

dN_dd = np.gradient(np.log(N), axis=0) / np.gradient(np.log(d))[:, None]

# + Collapsed="false"
ylow, ymed, yup = np.percentile(dN_dd, q=(10, 50, 90), axis=1)
plt.plot(d, ymed)
plt.fill_between(d, ylow, yup, alpha=0.1)

plt.xscale("log")
plt.grid(True)

# + Collapsed="false"
md(
    f"The initial dimensionality is {data.shape[1]}. From the look of the plot below, it seems that the intrinsic dimensionality is 3-4."
)

# + [markdown] Collapsed="false" jp-MarkdownHeadingCollapsed=true tags=[]
# # Data from H-AGN lightcone

# + Collapsed="false"
# !rsync -azzh --progress infinity:"/data75/laigle/CONE/Galaxies/catalogs/fits/Galaxies_0-1.d.fits" tmp/
# !rsync -azzh --progress infinity:"/data75/laigle/CONE/Galaxies/catalogs/pdb/full_catalog_0-1.d.pdb" tmp/

# + Collapsed="false"
p = PDBReader("tmp/full_catalog_0-1.d.pdb")
# -

p.structure

# + Collapsed="false"
columns = [
    "catalog/" + _
    for _ in ("sSFR", "spin", "vir/rvir", "vir/mvir", "vir/tvir", "met", "agegal", "z")
]
df = pd.DataFrame({k.replace("catalog/", ""): p[k] for k in columns})
df["Ltot"] = np.linalg.norm(p["catalog/L"], axis=1)
for i, chem in enumerate(p["catalog/chem"].T):
    df[f"chem_{i}"] = chem

# + Collapsed="false"
new_data = {}

mask = df["vir/mvir"] > 2e-3

for column in df:
    tmp = df.loc[mask, column].copy()
    q1, q5, q9 = tmp.quantile(q=[0.01, 0.5, 0.99])
    if q1 == 0:
        vmin = tmp[tmp > 0].min() / 2
        tmp += vmin
        q1, q5, q9 = (_ + vmin for _ in (q1, q5, q9))

    if q5 / q1 > 5 or q9 / q5 > 5:
        print(f"{column} ⇒ log")
        new_data[column] = normalize_log(tmp)
    else:
        print(f"{column} ⇒ lin", q1, q5, q9)
        new_data[column] = normalize_lin(tmp)
df_norm = pd.DataFrame(new_data)

# + Collapsed="false"
_ = plt.hist(
    df_norm.values, bins=np.linspace(0, 1, 100), histtype="step", label=df_norm.columns
)
plt.yscale("log")
# plt.loglog()
plt.legend(fontsize=6, ncol=3, loc="lower center")

# + Collapsed="false"
sample = slice(None)
# sample = np.random.choice(len(df_norm), size=500_000, replace=False)
data = np.ascontiguousarray(df_norm.values[sample])
N, d = find_dimensionality(
    data, bins=np.geomspace(2e-2, 4e0, 20), Nsample=50, use_MST=True, min_neigh=15
)

dN_dd = np.gradient(np.log(N), axis=0) / np.gradient(np.log(d))[:, None]

# + Collapsed="false"
x = d
y = dN_dd
ylow, ymed, yup = np.percentile(dN_dd, q=[10, 50, 90], axis=1)
fig, ax = plt.subplots()
ax.plot(d, ymed)
ax.fill_between(d, ylow, yup, alpha=0.1)
ax.set(
    xscale="log",
    xlabel="Distance",
    ylabel="Dimensionality",
)
plt.grid(True)

# + Collapsed="false"
md(
    f"The initial dimensionality is {data.shape[1]}. From the look of the plot below, it seems that the intrinsic dimensionality is (SEE ABOVE)."
)

# + [markdown] Collapsed="false"
# # Data from mock catalogues
# Using data from [Laigle et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.5104L/abstract) and [Davidzon et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.4817D/abstract). Description of columns available at https://www.horizon-simulation.org/PHOTOCAT/README_HZAGN_LAIGLE-DAVIDZON_2019.
# -

# !(cd tmp/ && wget -c https://www.horizon-simulation.org/PHOTOCAT/HorizonAGN_LAIGLE-DAVIDZON+2019_COSMOS_v1.6.fits)

# +
hdul = fits.open("tmp/HorizonAGN_LAIGLE-DAVIDZON+2019_COSMOS_v1.6.fits")
hdr = hdul[1].header
hdd = hdul[1].data

df = pd.DataFrame(
    {
        col.name: hdd.field(col.name).astype(
            str(hdd.field(col.name).dtype).replace(">", "=")
        )
        for col in tqdm(hdd.columns)
    },
)
# -

df

# + Collapsed="false"
new_data = {}

for column in (
    col
    for col in df.columns
    if col.endswith(("sedf", "TOT")) and "UP" not in col and "DOWN" not in col
):
    tmp = df[column].copy()
    if len(np.unique(tmp)) < 100:
        continue

    q1, q5, q9 = tmp.quantile(q=[0.01, 0.5, 0.99])
    if q1 == 0:
        vmin = tmp[tmp > 0].min() / 2
        tmp += vmin
        q1, q5, q9 = (_ + vmin for _ in (q1, q5, q9))

    if q5 / q1 > 5 or q9 / q5 > 5:
        print(f"{column} ⇒ log")
        tmp = normalize_log(tmp)
    else:
        print(f"{column} ⇒ lin", q1, q5, q9)
        tmp = normalize_lin(tmp)

    if (np.histogram(tmp, bins=np.linspace(0, 1, 25))[0] == 0).sum() > 10:
        continue
    new_data[column] = tmp
df_norm = pd.DataFrame(new_data)

# + Collapsed="false"
_ = plt.hist(
    df_norm.values, bins=np.linspace(0, 1, 100), histtype="step", label=df_norm.columns
)
plt.yscale("log")
# plt.loglog()
plt.legend(fontsize=6, ncol=3, loc="lower center")
# -

logger.setLevel(10)

# + Collapsed="false"
sample = slice(None)
sample = np.random.choice(len(df_norm), size=100_000, replace=False)
data = np.ascontiguousarray(df_norm.values[sample])
N, d = find_dimensionality(
    data, bins=np.geomspace(2e-2, 4e0, 20), Nsample=50, use_MST=True, min_neigh=10
)

dN_dd = np.gradient(np.log(N), axis=0) / np.gradient(np.log(d))[:, None]

# + Collapsed="false"
x = d
y = dN_dd
# ylow, ymed, yup = np.percentile(dN_dd, q=[5, 50, 95], axis=1)

fig, ax = plt.subplots()
ymed = np.mean(dN_dd, axis=1)
ystd = np.std(dN_dd, axis=1)
ylow, yup = ymed - ystd / 2, ymed + ystd / 2
ax.plot(d, ymed, label=r"$\pm \sigma$")
ax.fill_between(d, ylow, yup, alpha=0.1)

ylow, ymed, yup = np.percentile(dN_dd, q=[68 / 2, 50, 100 - 68 / 2], axis=1)
ax.plot(d, ymed, label="68%")
ax.fill_between(d, ylow, yup, alpha=0.1)

ax.legend()
ax.set(
    xscale="log",
    xlabel="Distance",
    ylabel="Dimensionality",
)
plt.grid(True)
# -
