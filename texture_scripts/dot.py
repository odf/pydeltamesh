from pydeltamesh.makeTexture import *

u = Input(U(), "u")
v = Input(V(), "v")
mask = ((u - 0.5)**2 + (v - 0.5)**2).sqrt() < 0.5

mask.name = "mask"

with open("dot_mktx.mt5", "w") as fp:
    write_poser_file(fp, "dot", [mask])


from PIL import Image
Image.fromarray(mask.data * 256).show()
