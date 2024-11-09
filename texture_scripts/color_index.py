from pydeltamesh.makeTexture import *

thread_index = Input((U() * 256).floor(), "thread_index")

counts = [2, 6, 14, 4, 14, 2, 4, 2]

count_inputs = [
    Input(c, f"count_{i}") for i, c in enumerate(counts)
]

starts = [0]
for c in count_inputs:
    starts.append(starts[-1] + c)

length = starts.pop()

t = thread_index % (2 * length)
index = t * (t < length) + (2 * length - 1 - t) * (t >= length)

out = 0
for i, s in enumerate(starts):
    if i > 0:
        out = out * (index < s) + i * (index >= s)

out.name = "color_index"

with open("color_index_mktx.mt5", "w") as fp:
    write_poser_file(fp, name="color_index", output_nodes=[out])

from PIL import Image
Image.fromarray(out.data * 32).show()
