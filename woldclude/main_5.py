# -*- coding: utf-8 -*-
# @Author: Marte
# @Date:   2018-10-06 17:36:27
# @Last Modified by:   Marte
# @Last Modified time: 2019-03-12 17:53:14
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jieba

#打开文本
text = open('md.txt').read()

#中文分词
text = ' '.join(jieba.cut(text))
#print(text[:100])

mask = np.array(Image.open("heart1.jpg"))
wc = WordCloud(mask = mask, font_path='simhei.ttf', mode = 'RGBA', background_color =None ).generate(text)

#从图片中生成颜色
image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

#显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

#保存文件
wc.to_file('md2.png')