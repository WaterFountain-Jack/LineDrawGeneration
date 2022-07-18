# -*- coding: utf-8 -*-
# 导入常用的库
import time
import os
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import model6
import json
# 导入flask库的Flask类和request对象
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # 定义字典className_list，把种类索引转换为种类名称
# className_list = ['__background__', 'wheat_head']

#------------------------------------------------------1.加载模型--------------------------------------------------------------
path_model="./netG_050.pth"
#model_loaded=torch.load(path_model,map_location='cpu') # 只有cpu加载模型

#model_loaded=torch.load(path_model)
model_loaded = model6.Generator()
model_loaded.load_state_dict(torch.load(path_model))
#------------------------------------------------------2.获取测试图片--------------------------------------------------------------

# img_dir=os.path.join(BASE_DIR ,"data", "global-wheat-detection","test")
# names = [name for name in list(filter(lambda x: x.endswith(".jpg"), os.listdir(img_dir)))]
#path_img = os.path.join(BASE_DIR ,"data", "global-wheat-detection","test",i)

# 根据图片文件路径获取图像数据矩阵
def get_imageNdarray(imageFilePath):
    input_image = Image.open(imageFilePath).convert("RGB")
    return input_image


#------------------------------------------------------3.定义图片预处理--------------------------------------------------------------
# 模型预测前必要的图像处理
def process_imageNdarray(input_image):
    preprocess = transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),  # 彩色图像转灰度图像num_output_channels默认1
        torchvision.transforms.Scale(128),
        torchvision.transforms.CenterCrop(128),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img_chw = preprocess(input_image)
    return img_chw.unsqueeze(0)  # chw:channel height width

#------------------------------------------------------4.模型预测--------------------------------------------------------------
# 使用模型对指定图片文件路径完成图像分类，返回值为预测的种类名称
def predict_image(model, imageFilePath):
    model.eval()  # 参数固化
    input_image = get_imageNdarray(imageFilePath)
    img_chw = process_imageNdarray(input_image)
    if torch.cuda.is_available():
        img_chw = img_chw.to('cuda')
        model.to('cuda')
    with torch.no_grad():  # 不计算梯度
        output_list = model(img_chw)
        output_dict = output_list[0]
        #print('对此图片路径 %s 的预测结果为 %s' % (output_dict))
        return output_dict

#------------------------------------------------------5.服务返回--------------------------------------------------------------


# 访问首页时的调用函数
@app.route('/')
def index_page():
    return render_template('index.html')


# 使用predict_image这个API服务时的调用函数
@app.route("/upload_image", methods=['POST'])
def anyname_you_like():
    startTime = time.time()
    received_file = request.files['input_image']
    imageFileName = received_file.filename
    if received_file:
        received_dirPath = './resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('image file saved to %s' % imageFilePath)
        usedTime = time.time() - startTime
        print('接收图片并保存，总共耗时%.2f秒' % usedTime)
        startTime = time.time()
        result = predict_image(model_loaded, imageFilePath)
        vutils.save_image(result.data, './static/output/output.png' ,normalize=True)

        usedTime = time.time() - startTime
        print('完成对接收图片的预测，总共耗时%.2f秒' % usedTime)
        return render_template("result.html",result=result)
    else:
        return 'failed'



# 主函数
if __name__ == "__main__":
    # print('在开启服务前，先测试predict_image函数')
    # imageFilePath = 'D:\\PycharmWorkPlaces\\DeepModel_deploy_flask\\data\\global-wheat-detection\\test\\51b3e36ab.jpg'
    # result = predict_image(model_loaded, imageFilePath)
    app.run("127.0.0.1", port=5000)
