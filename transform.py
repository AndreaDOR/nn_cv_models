from PIL import Image
import os
import sys

path = r"data"
# 加载GIF图像
# image = Image.open(path)
# image.convert("RGB")
# # 将GIF图像转换为JPG
# image.save("output.png", "PNG", quality=90)


# 遍历文件夹
for a, b, c in os.walk(path):
    for i in range(len(c)):
        img = os.path.join(a, c[i])
        # 将文件里的图片转换成png格式
        image = Image.open(img)
        image.convert("RGB")
        output_file = "output" + str(i) + ".png"
        image.save(output_file, "PNG", quality=90)
