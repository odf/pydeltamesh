from pydeltamesh.makeTexture import *


def maprange(x, from_min, from_max, to_min, to_max):
    t = (to_max - to_min) / (from_max - from_min)
    return (x - from_min) * t + to_min


def threads(x, scale, gap):
    a = maprange(x * scale % 1, gap, 1, 0, 1)
    return (1 - 2 * abs(a - 0.5)).bias(0.9)


u = Input(U(), "u")
v = Input(V(), "v")
scale = Input(10, "scale")
gap = Input(0.2, "gap")

warp = threads(u, scale, gap)
warp.name = "warp"

weft = threads(v, scale, gap)
weft.name = "weft"

with open("weaver_mktx.mt5", "w") as fp:
    write_poser_file(fp, "weaver", [warp, weft])

from PIL import Image
Image.fromarray(weft.data * 256).show()
