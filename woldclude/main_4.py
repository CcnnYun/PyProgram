# -*- coding: utf-8 -*-
# @Author: Marte
# @Date:   2018-10-06 17:24:55
# @Last Modified by:   Marte
# @Last Modified time: 2019-03-12 17:20:34


# wordcloud 词云
# Matplotlib是一个Python 2D绘图库
# PIL是Python平台上的图像处理标准库
# jieba（结巴）中文分词
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jieba

#打开文本
text = open('md.txt').read()

#中文分词
text = ' '.join(jieba.cut(text))
#print(text[:100])

#生成对象
mask=np.array(Image.open("heart.jpg"))
wc = WordCloud(mask=mask, font_path='Dengl.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text)

#显示词云
plt.imshow(wc,interpolation='bilinear')
plt.axis("off")
plt.show()

#保存文件
wc.to_file('hhhhhh.png')