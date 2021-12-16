import codecs
import struct


def read_pmd(fp):
    fp.seek(0)

    magic = fp.read(4)
    if magic != b"PZMD":
        raise IOError("Not a Poser PMD file")

    fp.seek(16)
    nrTargets = read_uint(fp)

    morphnames = []
    actornames = []
    nrDeltas = []
    dataStarts = []
    uuids = []

    for i in range(nrTargets):
        morphnames.append(read_str(fp))
        actornames.append(read_str(fp))
        nrDeltas.append(read_uint(fp))
        dataStarts.append(read_uint(fp))

    if min(dataStarts) > fp.tell():
        for i in range(nrTargets):
            uuids.append(read_str(fp))

    deltaIndices = []
    deltaVectors = []

    for pos in dataStarts:
        fp.seek(pos)
        indices, vectors = read_deltas(fp)
        deltaIndices.append(indices)
        deltaVectors.append(vectors)

    for i in range(len(dataStarts)):
        print(actornames[i])
        print(morphnames[i])
        print(uuids[i] if i < len(uuids) else "[no uuid]")
        print(nrDeltas[i])

        for j in range(len(deltaIndices[i])):
            k = deltaIndices[i][j]
            x, y, z = deltaVectors[i][j]
            print("%7d   %10.7f %10.7f %10.7f" % (k, x, y, z))

        print()


def read_uint(fp, size=4):
    if size == 1:
        return struct.unpack("!B", fp.read(size))[0]
    elif size == 2:
        return struct.unpack("!H", fp.read(size))[0]
    elif size == 4:
        return struct.unpack("!I", fp.read(size))[0]
    else:
        raise ValueError("invalid int size: " + size)


def read_str(fp):
    strlen = read_uint(fp, 1)
    return codecs.decode(fp.read(strlen))


def read_vector3d(fp):
    return struct.unpack("!fff", fp.read(12))


def read_deltas(fp):
    nrIndices = read_uint(fp)

    indices = []
    vectors = []
    for i in range(nrIndices):
        indices.append(read_uint(fp))
        vectors.append(read_vector3d(fp))

    return indices, vectors


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "rb") as fp:
        read_pmd(fp)
