from pydeltamesh.fileio.poserFile import PoserFile

import sys

inpath = sys.argv[1]
outpath = sys.argv[2]

with open(inpath) as fp:
    source = PoserFile(fp)

root = source.root

for ch in list(root.select('actor', 'channels', 'valueParm|targetGeom')):
    if ch.rest.startswith('PBMCC'):
        ch.unlink()

with open(outpath, "w") as fp:
    source.writeTo(fp)
