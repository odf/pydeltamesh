from math import pi
from pydeltamesh.makeTexture import *


def random(x):
    t = x.sin() * 99371
    return t - t.floor()


def rnd(x, lo, hi):
    return random(x) * (hi - lo) + lo


def make_val(x, y, scale):
    i = (x * scale).floor()
    amplitude = rnd(i, 0.2, 0.5)
    wavelength = rnd(i + scale, 0.2, 0.4)
    offset = random(i + 2 * scale)

    return 0.5 + amplitude * (2 * pi * (y + offset) / wavelength).sin()


u = Input(U(), "u")
v = Input(V(), "v")
scale = Input(20, "scale")

warp_val = make_val(u, v, scale)
warp_val.name = "warp_value"

weft_val = make_val(v + 1, u, scale)
weft_val.name = "weft_value"

with open("strand_values_mktx.mt5", "w") as fp:
    write_poser_file(
        fp,
        name="strand_values",
        output_nodes=[warp_val, weft_val]
    )

from PIL import Image
Image.fromarray(warp_val.data * 256).show()
