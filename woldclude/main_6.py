# -*- coding: utf-8 -*-
# @Author: Marte
# @Date:   2018-10-06 17:56:35
# @Last Modified by:   Marte
# @Last Modified time: 2019-03-12 17:52:16
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import jieba

# 打开文本
text = open('md.txt').read()
# 中文分词
text = ' '.join(jieba.cut(text))
#print(text[:100])


# 颜色函数 产生随机的颜色
def random_color(word, font_size, position, orientation, font_path, random_state):
    s = 'hsl(0, %d%%, %d%%)' % (random.randint(60, 80), random.randint(60, 80))
    #print(s)
    return s

# 生成对象
mask = np.array(Image.open("heart1.jpg"))
wc = WordCloud(color_func=random_color, mask=mask, font_path='simhei.ttf', mode= 'RGBA', background_color=None).generate(text)

# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# 保存到文件
wc.to_file('md6.png')