"""
基于VGG-19模型的的图像识别
该文件主要包括数据集加载、图像数据预处理、构造数据生成器、构建模型、加载权重、训练模型、评估模型等流程。
"""
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.initializers import he_normal
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

"""
功能：设置参数
"""
num_classes  = 10
batch_size   = 128
epochs       = 30
iterations   = 391
dropout      = 0.5
weight_decay = 0.0001
log_filepath = r'./tensorboard_logs/'
retrain_weight = 'retrain_weights/vgg19_weights_tf_dim_ordering_tf_kernels.h5'

"""
功能：设置GPU防止占满显存
"""
from tensorflow.keras import backend as K
if ('tensorflow' == K.backend()):
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

################################## 数据加载及预处理阶段 ###################################
"""
功能：加载cifar10数据集
"""
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # TODO完成加载cifar10数据集代码

"""
功能：数据预处理
"""
# TODO完成训练集样本和测试集样本的预处理、训练集标签和测试集标签的预处理代码
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)


################################## 构建模型、加载权重、编译模型阶段 ###################################
"""
功能：构建模型
"""
model = Sequential()
# Block 1
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv1', input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# TODO完成Block 2，包含两层卷积、一层池化，每层卷积包含128个卷积核（即输出128维）、卷积核3×3、保留边界处的卷积结果、施加一个正则项、定义一个权值初始化、增加一个批标准化和激活函数，池化层2×2、步长为2
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# TODO完成Block 3，包含四层卷积、一层池化，每层卷积包含256个卷积核（即输出256维）、卷积核3×3、保留边界处的卷积结果、施加一个正则项、定义一个权值初始化、增加一个批标准化和激活函数，池化层2×2、步长为2
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

model.add(Flatten(name='flatten'))
model.add(Dense(4096, use_bias = True, kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_cifar10'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(4096, kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))      
model.add(Dense(10, kernel_regularizer=tensorflow.keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifar10'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

"""
功能：为模型加载预训练权重

"""
model.load_weights(retrain_weight, by_name=True)

"""
功能：编译模型
"""

sgd = optimizers.Adam(lr=.1, momentum=0.9, nesterov=True)
# TODO完成编译模型代码
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


"""
功能：设置TensorBoard,以便可视化某些参数的变化，如loss、accuracy等
"""
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)  # TODO完成设置TensorBoard代码
best_model = ModelCheckpoint('save_model_and_weights/best_weight.h5',monitor='val_loss',verbose=0, save_best_only=True)
cbks = [best_model,tb_cb]

################################## 构造数据生成器阶段 ###################################
"""
功能：设置数据生成器对象参数
"""
datagen = ImageDataGenerator(horizontal_flip=True,
        width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

datagen.fit(x_train)

################################## 训练模型 ###################################
"""
功能：训练模型
"""

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=datagen.flow(x_test, y_test))  # TODO完成训练模型代码

################################### 评估模型 ####################################
"""
功能：评估模型
"""
score = model.evaluate(x_test, y_test, verbose=0)  # TODO完成评估模型代码
# 输出结果
print(f'Test score:{score[0]}')
print(f'Accuracy:{score[1]*100}')
print('完成！')

##################################### 保存模型 #######################################
"""
功能：保存模型
"""
# TODO完成保存模型代码
if score[1] > 0.8:
    model.save('save_model_and_weights/retrain_model.h5')
