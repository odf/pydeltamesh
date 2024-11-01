from math import pi
from pydeltamesh.makeTexture import *


def random(x):
    t = x.sin() * 99371
    return t - t.floor()


def maprange(x, from_min, from_max, to_min, to_max):
    t = (to_max - to_min) / (from_max - from_min)
    return (x - from_min) * t + to_min


def make_val(x, y, scale):
    i = (x * scale).floor()
    amplitude = maprange(random(i), 0, 1, 0.2, 0.5)
    wavelength = maprange(random(i + scale), 0, 1, 0.2, 0.4)
    offset = maprange(random(i + 2 * scale), 0, 1, 0, wavelength / 2)

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
