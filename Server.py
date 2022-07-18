# -*- coding: utf-8 -*-
# 导入常用的库
import time
import os
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
# 导入flask库的Flask类和request对象
from flask import request, Flask
from matplotlib import pyplot as plt

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # 定义字典className_list，把种类索引转换为种类名称
# className_list = ['__background__', 'wheat_head']

#------------------------------------------------------1.加载模型--------------------------------------------------------------
path_model= "G.pth"
model_loaded=torch.load(path_model)

#------------------------------------------------------2.获取测试图片--------------------------------------------------------------

# img_dir=os.path.join(BASE_DIR ,"data", "global-wheat-detection","test")
# names = [name for name in list(filter(lambda x: x.endswith(".jpg"), os.listdir(img_dir)))]
#path_img = os.path.join(BASE_DIR ,"data", "global-wheat-detection","test",i)

transforms = torchvision.transforms.Compose([

    torchvision.transforms.Grayscale(num_output_channels=3),  # 彩色图像转灰度图像num_output_channels默认1
    torchvision.transforms.Scale(128),
    torchvision.transforms.CenterCrop(128),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])#使用平均值和标准偏差对张量图像进行规格化,消除量纲
#------------------------------------------------------4.模型预测--------------------------------------------------------------
# 使用模型对指定图片文件路径完成图像分类，返回值为预测的种类名称
def predict_image(model, imageFilePath):
    model.eval()  # 参数固化

    dataset = torchvision.datasets.ImageFolder(imageFilePath, transform=transforms)
    dataloader = torch.utils.data.DataLoader(  # 数据取器
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )
    if torch.cuda.is_available():
        model.to('cuda')
    with torch.no_grad():  # 不计算梯度
        for i, (img) in enumerate(dataloader):
            output = model(img[0].to('cuda'))
            plt.imshow(torch.squeeze((output.data[0]/255).cpu()).numpy().reshape(128, 128, 3))
        #print('对此图片路径 %s 的预测结果为 %s' % (output_dict))
        return output

#------------------------------------------------------5.服务返回--------------------------------------------------------------
# 定义回调函数，接收来自/的post请求，并返回预测结果
@app.route("/", methods=['POST'])
def return_result():
    startTime = time.time()
    received_file = request.files['file']
    imageFileName = received_file.filename
    if received_file:
        received_dirPath = './resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('图片文件保存到此路径：%s' % imageFilePath)
        usedTime = time.time() - startTime
        print('接收图片并保存，总共耗时%.2f秒' % usedTime)
        startTime = time.time()
        print(imageFilePath)
        result = predict_image(model_loaded, imageFilePath)
        result = str(result)
        print(result)
        usedTime = time.time() - startTime
        print('完成对接收图片的检测，总共耗时%.2f秒' % usedTime)
        print("testtest",result)
        return result
    else:
        return 'failed'


# 主函数
if __name__ == "__main__":
    imageFilePath = 'data1/'
    model_loaded = torch.load(path_model)
    predict_image(model_loaded,imageFilePath)
    #print('在开启服务前，先测试predict_image函数')
    # imageFilePath = os.path.join(BASE_DIR, "data", "global-wheat-detection", "test", "51b3e36ab.jpg")
    # result = predict_image(model_loaded, 'D:\\PycharmWorkPlaces\\DeepModel_deploy_flask\\data\\global-wheat-detection\\test\\51b3e36ab.jpg')
    # print(result)
    #app.run("127.0.0.1", port=5000)

