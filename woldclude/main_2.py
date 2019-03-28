# -*- coding: utf-8 -*-
# @Author: Marte
# @Date:   2018-10-06 15:43:16
# @Last Modified by:   Marte
# @Last Modified time: 2018-10-06 20:13:05
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#打开文件
text = open('md.txt').read()
wc =  WordCloud(font_path='Dengl.ttf',
  width=800, height=600,
  mode='RGBA',
  background_color=None).generate(text)

#显示词云
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()

#保存文件
wc.to_file('md2.png')