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
weft = threads(v, scale, gap)

warp_mask_raw = warp > 0
weft_mask_raw = weft > 0

opacity = warp_mask_raw.max(weft_mask_raw)
opacity.name = "opacity"

i_warp = (u * scale).floor()
i_weft = (v * scale).floor()
period = up_count + down_count

weft_is_up = rem(i_warp - shift * i_weft, period) < up_count

warp_mask = warp_mask_raw * (weft_is_up * (1 - weft_mask_raw)).max(1 - weft_is_up)
warp_mask.name = "warp_mask"

weft_mask = weft_mask_raw.min(1 - warp_mask)
weft_mask.name = "weft_mask"

bump = weft * weft_is_up + warp * (1 - weft_is_up)
bump.name = "bump"

thread_index = 2 * (warp_mask * (i_warp + 0.5) + weft_mask * (i_weft + 1))
thread_index.name = "thread_index"

with open("weaver_mktx.mt5", "w") as fp:
    write_poser_file(
        fp, "weaver", [opacity, bump, warp_mask, weft_mask, thread_index]
    )

from PIL import Image
Image.fromarray((thread_index % 3).data * 256).show()
