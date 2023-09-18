"""
    功能：按着路径，导入单张图片做预测
    作者： Leo在这

"""
import sys
from torchvision.models import resnet34
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable


def predict(imgPath, model_path):
    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """
        加载模型与参数
    """

    # 加载模型
    model = resnet34(pretrained=False, num_classes=2).to(device)  # 43.6%

    # 加载模型参数
    if device == "cpu":
        # 加载模型参数
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    else:
        model.load_state_dict(torch.load(model_path))

    """
        加载图片与格式转化
    """
    img_path = imgPath

    # 图片标准化
    transform_BZ = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )  # 取决于数据集

    val_tf = transforms.Compose(
        [transforms.Resize(206), transforms.ToTensor(), transform_BZ]  # 标准化操作
    )

    def padding_black(img):  # 如果尺寸太小可以扩充
        w, h = img.size
        scale = 206.0 / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 206
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2, (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    img = Image.open(img_path)  # 打开图片
    img = img.convert("RGB")  # 转换为RGB 格式
    img = padding_black(img)

    img_tensor = val_tf(img)

    # 增加batch_size维度
    img_tensor = Variable(
        torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False
    ).to(device)

    """
        数据输入与模型输出转换
    """
    model.eval()
    with torch.no_grad():
        output_tensor = model(img_tensor)
        # print(output_tensor)

        # 将输出通过softmax变为概率值
        output = torch.softmax(output_tensor, dim=1)

        #
        # 输出可能性最大的那位
        pred_value, pred_index = torch.max(output, 1)

        #
        # 将数据从cuda转回cpu
        if torch.cuda.is_available() == False:
            pred_value = pred_value.detach().cpu().numpy()
            pred_index = pred_index.detach().cpu().numpy()
        #
        # print(pred_value)
        # print(pred_index)
        #
        # 增加类别标签
        classes = ["其他", "露娜"]
        #
        # # result = "预测类别为： " + str(classes[pred_index[0]]) + " 可能性为: " + str(pred_value[0] * 100) + "%"
        #
        return classes[pred_index[0]], pred_value[0].item() * 100
        # print(
        #     "预测类别为： ",
        #     classes[pred_index[0]],
        #     " 可能性为: ",
        #     pred_value[0].item() * 100,
        #     "%",
        # )
