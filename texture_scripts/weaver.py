from pydeltamesh.makeTexture import *


def rem(a, b):
    return (a % b + b) % b


def random(x):
    t = x.sin() * 99371
    return t - t.floor()


def maprange(x, from_min, from_max, to_min, to_max):
    t = (to_max - to_min) / (from_max - from_min)
    return (x - from_min) * t + to_min


def threads(x, scale, gap):
    a = maprange(x * scale % 1, gap / 2, 1 - gap / 2, 0, 1)
    return (a * (1 - a)).sqrt() * (1 - gap)


u = Input(U(), "u")
v = Input(V(), "v")
scale = Input(20, "scale")
gap_warp = Input(0.3, "gap_warp")
gap_weft = Input(0.3, "gap_weft")
up_count = Input(1, "up_count")
down_count = Input(1, "down_count")
shift = Input(1, "shift")

warp = threads(u, scale, gap_warp)
weft = threads(v + 1, scale, gap_weft)

warp_mask_raw = warp > 0
weft_mask_raw = weft > 0

opacity = warp_mask_raw | weft_mask_raw
opacity.name = "opacity"

idx_warp_raw = (u * scale).floor()
idx_weft_raw = (v * scale).floor()

period = up_count + down_count

next_weft_is_up = rem(idx_warp_raw - shift * (idx_weft_raw + 1), period) < up_count
this_weft_is_up = rem(idx_warp_raw - shift * idx_weft_raw, period) < up_count
previous_weft_is_up = rem(idx_warp_raw - shift * (idx_weft_raw - 1), period) < up_count

warp_mask = warp_mask_raw & ~(weft_mask_raw & this_weft_is_up)
warp_mask.name = "warp_mask"

weft_mask = weft_mask_raw & ~warp_mask
weft_mask.name = "weft_mask"

weft_pos = rem(u * scale - shift * idx_weft_raw, period)
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

bump_raw = (weft + weft_height) * weft_mask + (warp + warp_height) * warp_mask
bump = bump_raw / scale
bump.name = "bump"

idx_warp = idx_warp_raw * warp_mask
idx_warp.name = "warp_thread_index"

idx_weft = idx_weft_raw * weft_mask
idx_weft.name = "weft_thread_index"

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
Image.fromarray(bump_raw.data * 256).show()
