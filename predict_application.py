"""
利用训练好的模型，实现图片识别
"""
from PIL import Image
import numpy as np
import os.path
from tensorflow.keras.models import load_model

"""
功能：1 加载模型
"""
model = load_model(r'save_model_and_weights/retrain_model.h5')  # TODO完成加载模型代码
print(model.summary())  # 打印模型内容


while True:
    img_path = input('请输入需要预测的图片本地路径( 若输入“Q”,则退出 ):  ')
    print('注：该模型可以识别的类型有：(1 飞机；2 汽车；3 鸟；4 猫；5 鹿；6 狗；7 青蛙；8 马；9 船；10 卡车)')
    if img_path == 'Q':
        break
    if not os.path.exists(img_path):
        print("file not exist!")
        continue

    """
    功能：2 图像预处理
    """
    # TODO完成图像预处理代码
    img = Image.open(img_path).resize((32, 32), Image.BILINEAR)
    x = np.array(img, dtype='float32')
    x = np.expand_dims(x, axis=0)


    """
    功能：3 预测
    """
    results = model.predict(x)  # TODO完成模型预测代码

    """
    功能：4 将预测结果转化为类别
    """
    labels = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    label_result = labels[np.argmax(results)]
    print(f'预测结果:{results}')
    print(f'预测最有可能的类别:{label_result},概率为{np.max(results)}',)