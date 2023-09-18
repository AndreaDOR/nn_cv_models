import torch
from torch import nn
from torch.utils.data import DataLoader
from util import LoadData
from torchvision.models import resnet18, resnet34
import sys


def train(model, train_loader, optimizer, lossf):
    # 计算训练集的大小
    size = len(train_loader.dataset)
    # 遍历训练集
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和目标转换为GPU可用的格式
        data, target = data.cuda(), target.cuda()
        # 梯度清零
        optimizer.zero_grad()
        # 计算模型的输出
        output = model(data)
        # 计算损失
        loss = lossf(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
    # 打印训练结果
    if epoch % 10 == 0:
        current = batch_idx * size
        print(f"损失: {loss.item()}\n [{current:>5d}\{size:>5d}]")


# 定义一个测试函数，用于测试模型
def test(model, test_loader):
    # 计算测试集的大小
    size = len(test_loader.dataset)
    # 初始化测试损失
    test_loss, correct = 0, 0

    # 设置模型为评估模式
    model.eval()
    # 不计算梯度
    with torch.no_grad():
        # 遍历测试集
        for data, target in test_loader:
            # 将数据和标签转换为GPU可用的类型
            data, target = data.cuda(), target.cuda()
            # 计算模型的预测结果
            pred = model(data)
            loss = nn.CrossEntropyLoss()
            # 计算损失
            test_loss += loss(pred, target).item()
            # 计算正确率
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()

    # 将损失减少到最小值
    test_loss /= size
    correct /= size
    # 打印测试平均损失和测试精度
    print(f"\n测试平均损失: {test_loss:.3f}\n测试精度: {correct:.3f}")
    # 打印正确率
    print(f"正确率:{correct*100:.2f}%")

    # 如果调用main函数，则打印batch_size


if __name__ == "__main__":
    batch_size = 8
    output_num = 2
    # 加载训练数据
    train_data = LoadData("train.txt", True)
    valid_data = LoadData("test.txt", False)

    train_dataloader = DataLoader(
        dataset=train_data,
        num_workers=4,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=valid_data, num_workers=4, pin_memory=True, batch_size=batch_size
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 修改最后一层模型
    # pretrain_model = resnet18(pretrained=False)
    pretrain_model = resnet34(pretrained=False)

    fc_feature = pretrain_model.fc.in_features
    # print(fc_feature)

    pretrain_model.fc = nn.Linear(fc_feature, output_num)
    pretrain_dict = torch.load("./resnet34.pth")  # 正确率:73.33%

    pretrain_dict.pop("fc.weight")
    pretrain_dict.pop("fc.bias")
    model_dict = pretrain_model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    pretrain_model.load_state_dict(model_dict)

    for name, value in pretrain_model.named_parameters():
        if (name != "fc.weight") and (name != "fc.bias"):
            value.requires_grad = False
    params_conv = filter(lambda p: p.requires_grad, pretrain_model.parameters())
    # filter 函数将模型中属性 requires_grad = True 的参数选出来
    model = pretrain_model.to(device)
    # 定义损失函数和优化器
    lossfn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params_conv, lr=0.001)
    # 优化器只更新fc层
    epoch = 2
    for t in range(epoch):
        print(f"epoch {t+1}")
        train(model, train_dataloader, optimizer, lossfn)
        test(model, test_dataloader)
    print("训练结束")

    torch.save(model.state_dict(), "resnet34_pretrain.pth")
    print("保存模型")
