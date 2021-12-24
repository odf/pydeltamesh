import sys
from lib import obj


def _cyclicPairs(indices):
    return zip(indices, indices[1:] + indices[:1])


for path in sys.argv[1:]:
    try:
        with open(path) as fp:
            data = obj.load(fp, path)
            faces = [ f.vertices for f in data.faces ]
            edgeLocations = {}

            for i, face in enumerate(faces):
                for k, (v, w) in enumerate(_cyclicPairs(face)):
                    edgeLocations.setdefault((v, w), []).append((i, k, False))
                    edgeLocations.setdefault((w, v), []).append((i, k, True))

            if any(len(a) > 2 for a in edgeLocations.values()):
                print(path)
    except:
        pass
