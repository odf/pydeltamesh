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
gap = Input(0.4, "gap")
up_count = Input(3, "up_count")
down_count = Input(2, "down_count")
shift = Input(2, "shift")

warp = threads(u, scale, gap)
weft = threads(v, scale, gap)

warp_mask_raw = warp > 0
weft_mask_raw = weft > 0

opacity = warp_mask_raw.max(weft_mask_raw)
opacity.name = "opacity"

idx_warp = (u * scale).floor()
idx_warp.name = "warp_thread_index"

idx_weft = (v * scale).floor()
idx_weft.name = "weft_thread_index"

period = up_count + down_count

next_weft_is_up = rem(idx_warp - shift * (idx_weft + 1), period) < up_count
this_weft_is_up = rem(idx_warp - shift * idx_weft, period) < up_count
previous_weft_is_up = rem(idx_warp - shift * (idx_weft - 1), period) < up_count

warp_mask = warp_mask_raw & ~(weft_mask_raw & this_weft_is_up)
warp_mask.name = "warp_mask"

weft_mask = weft_mask_raw & ~warp_mask
weft_mask.name = "weft_mask"

weft_pos = rem(u * scale - shift * idx_weft, period)
weft_height = 0.5 * (
    this_weft_is_up
    - (0.5 - weft_pos).clamp()
    + (weft_pos + 0.5 - period).clamp()
    - (weft_pos + 0.5 - up_count).clamp() * (weft_pos < up_count)
    + (up_count + 0.5 - weft_pos).clamp() * (weft_pos > up_count)
).smoothstep()

warp_pos = rem(v * scale, 1)
warp_height = 0.5 * (
    ~this_weft_is_up
    + (warp_pos - 0.5) * (
        ((warp_pos > 0.5) & this_weft_is_up & ~next_weft_is_up)
        + ((warp_pos < 0.5) & ~this_weft_is_up & previous_weft_is_up)
        - ((warp_pos > 0.5) & ~this_weft_is_up & next_weft_is_up)
        - ((warp_pos < 0.5) & this_weft_is_up & ~previous_weft_is_up)
    )
).smoothstep()

bump = (weft + weft_height) * weft_mask + (warp + warp_height) * warp_mask
bump.name = "bump"

thread_u = u * weft_mask + v * warp_mask
thread_u.name = "thread_u"

thread_v = v * weft_mask + (1 - u) * warp_mask
thread_v.name = "thread_v"

with open("weaver_mktx.mt5", "w") as fp:
    write_poser_file(
        fp,
        name="weaver",
        output_nodes=[
            opacity, bump,
            warp_mask, weft_mask,
            idx_warp, idx_weft,
            thread_u, thread_v
        ]
    )

from PIL import Image
Image.fromarray(thread_v.data * 256).show()
