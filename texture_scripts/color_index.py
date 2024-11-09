from pydeltamesh.makeTexture import *

thread_index = Input((U() * 208).floor(), "thread_index")

counts = [2, 24, 24, 2, 0, 0, 0, 0]

count_inputs = [
    Input(c, f"count_{counts[i]}") for i, c in enumerate(counts)
]

starts = [0]
for c in count_inputs:
    starts.append(starts[-1] + c)
print([s.data for s in starts[1:]])

length = starts.pop()

index = ((thread_index + length) % (2 * length) + 0.5 - length).abs() - 0.5

out = 0
for i, s in enumerate(starts):
    if i > 0:
        out = out * (index < s) + i * (index >= s)

out.name = "color_index"

with open("color_index_mktx.mt5", "w") as fp:
    write_poser_file(fp, name="color_index", output_nodes=[out])

from PIL import Image
Image.fromarray(out.data * 50).show()
