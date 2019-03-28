# -*- coding: utf-8 -*-
# @Author: Marte
# @Date:   2018-10-06 16:50:19
# @Last Modified by:   Marte
# @Last Modified time: 2018-10-06 20:13:36
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba

#打开文本
text = open('md.txt').read()

#中文分词
text = ' '.join(jieba.cut(text))
print(text[:100])

#生成对象
wc = WordCloud(font_path='Dengl.ttf',
  width=800, height=600,
  mode='RGBA',
  background_color=None).generate(text)

#显示词云
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()

#保存文件
wc.to_file('md3.png')