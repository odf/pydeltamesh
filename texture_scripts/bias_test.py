from pydeltamesh.makeTexture import *

u = Input(U(), "u")
v = Input(V(), "v")

diff = 1000 * (u.bias(v) - u**(v.log(0.5))).abs()
diff.name = "diff"

with open("bias_test_mktx.mt5", "w") as fp:
    write_poser_file(fp, "bias_test", [diff])


from PIL import Image
Image.fromarray(diff.data * 256).show()
