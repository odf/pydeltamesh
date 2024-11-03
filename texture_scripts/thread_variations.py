from math import pi
from pydeltamesh.makeTexture import *


def random(x):
    t = x.sin() * 99371
    return t - t.floor()


def rnd(x, lo, hi):
    return random(x) * (hi - lo) + lo


def make_val(x, y, scale):
    i = (x * scale).floor()
    a1 = random(i)
    a2 = random(i + scale)
    a3 = random(i + 2 * scale)
    a4 = random(i + 3 * scale)
    a5 = random(i + 4 * scale)

    offset = random(i + 5 * scale)

    arg = 2 * pi * (y + offset)
    return (
        a1 * arg.sin()
        + a2 * (arg * 2).sin()
        + a3 * (arg * 3).sin()
        + a4 * (arg * 4).sin()
        + a5 * (arg * 5).sin()
    ) / 10 + 0.5


u = Input(U(), "u")
v = Input(V(), "v")
scale = Input(720, "scale")

warp_val = make_val(u, v, scale)
warp_val.name = "warp_variation"

weft_val = make_val(v + 1, u, scale)
weft_val.name = "weft_variation"

with open("thread_variations_mktx.mt5", "w") as fp:
    write_poser_file(
        fp,
        name="thread_variations",
        output_nodes=[warp_val, weft_val]
    )

from PIL import Image
Image.fromarray(warp_val.data * 256).show()
