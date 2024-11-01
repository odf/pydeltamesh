from pydeltamesh.makeTexture import *


def rem(a, b):
    return (a % b + b) % b


def random(x):
    t = x.sin() * 99371
    return t - t.floor()


def maprange(x, from_min, from_max, to_min, to_max):
    t = (to_max - to_min) / (from_max - from_min)
    return (x - from_min) * t + to_min


def threads(x, scale, gap, gap_variation):
    thread_nr = (x * scale).floor()
    gap = gap + gap_variation * (random(thread_nr) - 0.5)
    a = maprange(x * scale % 1, gap / 2, 1 - gap / 2, 0, 1)
    return (a * (1 - a)).sqrt() * (1 - gap)


u = Input(U(), "u")
v = Input(V(), "v")
scale = Input(20, "scale")
gap = Input(0.4, "gap")
gap_variation = Input(0.2, "gap_variation")
up_count = Input(1, "up_count")
down_count = Input(1, "down_count")
shift = Input(1, "shift")

warp = threads(u, scale, gap, gap_variation)
weft = threads(v + 1, scale, gap, gap_variation)

warp_mask_raw = warp > 0
weft_mask_raw = weft > 0

opacity = warp_mask_raw | weft_mask_raw
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
Image.fromarray(bump.data * 256).show()
