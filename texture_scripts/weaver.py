from pydeltamesh.makeTexture import *


def rem(a, b):
    return (a % b + b) % b


def maprange(x, from_min, from_max, to_min, to_max):
    t = (to_max - to_min) / (from_max - from_min)
    return (x - from_min) * t + to_min


def threads(x, scale, gap):
    a = maprange(x * scale % 1, gap / 2, 1 - gap / 2, 0, 1)
    return (a * (1 - a)).sqrt()


u = Input(U(), "u")
v = Input(V(), "v")
scale = Input(10, "scale")
gap = Input(0.2, "gap")
up_count = Input(3, "up_count")
down_count = Input(2, "down_count")
shift = Input(2, "shift")

warp = threads(u, scale, gap)
warp.name = "warp"

weft = threads(v, scale, gap)
weft.name = "weft"

i_warp = (u * scale).floor()
i_weft = (v * scale).floor()
period = up_count + down_count

weft_is_up = rem(i_warp - shift * i_weft, period) < up_count
weft_is_up.name = "weft_is_up"

weave = weft * weft_is_up + warp * (1 - weft_is_up)
weave.name = "weave"

with open("weaver_mktx.mt5", "w") as fp:
    write_poser_file(fp, "weaver", [warp, weft, weft_is_up, weave])

from PIL import Image
Image.fromarray(weave.data * 256).show()
