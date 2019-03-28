# -*- coding: utf-8 -*-
# @Author: Marte
# @Date:   2019-02-17 15:35:37
# @Last Modified by:   Marte
# @Last Modified time: 2019-03-13 14:53:22
from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
#引入了输入层,全连接层,二维卷积,二维池化,二维上采样
from keras.models import Model, load_model
#引入model,模型
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

def loadNoisy():
    (x_train, _), (x_test, _) = mnist.load_data()
    #有训练集和测试集,x_train是数组存的数据,后面空出来的是图片上表示的数字,因为并不需要分类所以这里空着即可.
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    #进行归一化为0~1
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    #重塑reshape成一个四维的tensor(图片的宽度和高度,个数)
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size =x_train.shape)
    #添加随机白噪声,是一个正态分布的normal,均值是0,方差是1,数组的size保持和原来一样.
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    #clip保证数据的范围在0~1之间,不能有负的.
    return x_test, x_train, x_test_noisy, x_train_noisy

#绘制噪声图
def DrawNoisyImg(x_test_noisy):
    n = 10
    plt.figure(figsize=(20, 2))
    #一共有十个图片,大小是20*2
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
    #一共一行n列,第i+1个子图
        plt.imshow(x_test_noisy[i].reshape(28, 28))
    #把加了噪音的第i个画出来
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    #画出来以后把x,y轴隐藏起来
    plt.show()
    
#绘制训练过程中损失和准确率变化图
def historyDraw(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()

#根据噪声图和原始图训练模型
def TrainModel(x_test, x_train, x_test_noisy, x_train_noisy):
    #引入model,模型
    input_img = Input(shape=(28, 28, 1,))
     # N * 28 * 28 * 1
     # //高度 宽度 深度 以及数据的个数
    #print(input_img)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)#卷积核大小3*3
    #因为padding='same'所以卷积之后图像的尺寸不变  激活函数是relu 使用输入的图片input_img 卷积之后图片变为28*28*32
    x = MaxPooling2D((2, 2), padding='same')(x)
    #使用二维的最大池化函数 池化核大小2*2  池化之后图片变为14*14*32
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    #再经过一次卷积  之后图片变为14*14*32
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    #得到编码后 或者说隐层的表示  池化之后图片变为7*7*32
    # 7 * 7 * 32
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(encoded)
    #进行一次卷积 feature map的个数是32  7 * 7 * 32
    x = UpSampling2D((2, 2))(x)
    #进行上采样 会使得shape变大  14 * 14 * 32
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    #14 * 14 * 32
    x = UpSampling2D((2, 2))(x)
    #28 * 28 * 32
    decoded = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
    #28 * 28 * 1 经过解码之后的表示 变得和刚开始的图片尺寸又一样了
    #print(decoded)

    autoencoder = Model(input_img, decoded)
    #整个模型的输入是input_img 输出是decoded
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy"])
    #对它进行compile编译一下 优化算法采用adadelta 损失函数采用交叉熵
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

    hist = autoencoder.fit(x_train_noisy, x_train,
                    epochs=100,
                    batch_size=128,
                    shuffle=True,
                    callbacks=[tensorboard],
                    validation_data=(x_test_noisy, x_test),
                    validation_split=0.2)
    #使用x_train_noisy做输入,x_train做输出图片,训练一百轮迭代,每128条做一组进行训练,每次训练进行打乱 再加一个测试集
    autoencoder.save('autoencoder1.h5')
    #把模型整个保存下来 可以把模型从服务器拷到本地上
    print(hist.history)
    #绘制训练过程中损失和准确率变化图
    historyDraw(hist)
   


def Npredict(x_test_noisy):
    autoencoder = load_model('autoencoder1.h5')
    #加载这个模型  直接把模型存起来的话 前面那些都不用运行 直接用就可以了
    decoded_imgs = autoencoder.predict(x_test_noisy)
    #对于加了噪音的测试数据进行预测  输出decoded_imgs
    #画十列
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        #两行十列 画第一个
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_noisy[i+10].reshape(28, 28))
        #28*28 画加了噪音的第i个
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        #画第十一个
        plt.imshow(decoded_imgs[i+10].reshape(28, 28))
        #画处理以后去燥的第i个
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def main():
    x_test, x_train, x_test_noisy, x_train_noisy = loadNoisy()
    #DrawNoisyImg(x_test_noisy)
    TrainModel(x_test, x_train, x_test_noisy, x_train_noisy)
    Npredict(x_test_noisy)

if __name__ == '__main__':
    main()