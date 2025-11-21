import yt
from scipy.io import FortranFile
from pathlib import Path
from itertools import chain

ds = yt.load("output_00080")
ad = ds.all_data()

root = Path(ds.filename).parent

for domain in ds.index.domains:
    # Convert hydro files to single precision
    for fh in domain.field_handlers:
        p = Path(fh.fname)

        new_p = root / "single_precision" / p.parent.name / p.name
        new_p.parent.mkdir(parents=True, exist_ok=True)

        print(f"{str(p)[-40:]:>40s} → {str(new_p)[-40:]:>40s}")

        with FortranFile(p, "r") as fin, FortranFile(new_p, "w") as fout:
            header = {}
            for attr, count, dtype in fh.attrs:
                tmp = fin.read_record(dtype)
                fout.write_record(tmp)

                if count == 1:
                    header[attr] = tmp[0]
                else:
                    header[attr] = tmp

            for _ilevel in range(header["nlevelmax"]):
                for _ibound in range(header["nboundary"] + header["ncpu"]):
                    ilvl = fin.read_ints()
                    fout.write_record(ilvl)
                    ncell = fin.read_ints()
                    fout.write_record(ncell)
                    ncell = ncell[0]
                    if ncell == 0:
                        continue

                    for _ind in range(2**ds.dimensionality):
                        for _ivar in range(header["nvar"]):
                            data = fin.read_reals(dtype="float64")
                            if data.size != ncell:
                                raise OSError("Data size does not match ncell!")

                            fout.write_record(data.astype("float32"))

            # Make sure we reached the end of the file
            ipos = fin._fp.tell()
            fin._fp.seek(0, 2)
            eos = fin._fp.tell()
            if ipos != eos:
                raise OSError("Did not reach the end of the file!")

    # Convert particles files to single precision
    for fh in domain.particle_handlers:
        p = Path(fh.fname)

        new_p = root / "single_precision" / p.parent.name / p.name
        new_p.parent.mkdir(parents=True, exist_ok=True)

        print(f"{str(p)[-40:]:>40s} → {str(new_p)[-40:]:>40s}")

        with FortranFile(p, "r") as fin, FortranFile(new_p, "w") as fout:
            header = {}
            for attr, count, dtype in fh.attrs:
                tmp = fin.read_record(dtype)
                fout.write_record(tmp)

                if count == 1:
                    header[attr] = tmp[0]
                else:
                    header[attr] = tmp

            # Read until we reach eof
            for field in fh.field_types.keys():
                offset = fh.field_offsets[field]
                if fin._fp.tell() != offset:
                    raise OSError("File pointer is not at expected offset!")
                data = fin.read_record(dtype=fh.field_types[field])

                if data.size != header["npart"]:
                    raise OSError("Data size does not match npart!")

                if fh.field_types[field] == "float64":
                    fout.write_record(data.astype("float32"))
                else:
                    fout.write_record(data)

            # Make sure we reached the end of the file
            ipos = fin._fp.tell()
            fin._fp.seek(0, 2)
            eos = fin._fp.tell()
            if ipos != eos:
                raise OSError("Did not reach the end of the file!")

# Copy .nml and info_*.txt files
for p in chain(root.glob("*.nml"), root.glob("info_*.txt")):
    new_p = root / "single_precision" / p.name
    new_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"{str(p)[-40:]:>40s} → {str(new_p)[-40:]:>40s}")
    with open(p) as fin, open(new_p, "w") as fout:
        fout.write(fin.read())
