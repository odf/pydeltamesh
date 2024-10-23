from pydeltamesh.makeTexture import *

u = Input(U(), "u")
v = Input(V(), "v")
a = u * 10 - 5

diff1 = (a % 1 - a % -1).abs() * 1000
diff2 = (a % 1 + -a % 1).abs() * 1000

with open("mod_test_mktx.mt5", "w") as fp:
    write_poser_file(fp, "mod_test", [diff1, diff2])


from PIL import Image
Image.fromarray(diff2.data * 256).show()
