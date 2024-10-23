from pydeltamesh.makeTexture import *

u = Input(U(), "u")
v = Input(V(), "v")

g1 = (u * 2).bias(1 - v) / 2
g2 = 1 - ((1 - u) * 2).bias(1 - v) / 2
g = (u < 0.5) * g1 + (u >= 0.5) * g2

diff = 1000 * (u.gain(v) - g).abs()
diff.name = "diff"

with open("gain_test_mktx.mt5", "w") as fp:
    write_poser_file(fp, "gain_test", [diff])


from PIL import Image
Image.fromarray(diff.data * 256).show()
