# -*- coding: utf-8 -*-
# @Author: Marte
# @Date:   2019-03-08 10:34:39
# @Last Modified by:   Marte
# @Last Modified time: 2019-03-08 11:16:23
import tensorflow as tf
import NumPy as np
# g1 = tf.Graph()
# with g1.as_default():
#     v = tf.get_variable("v", initializer=tf.zeros_initializer()(shape=[1]))

# g2 = tf.Graph()
# with g2.as_default():
#     v = tf.get_variable("v", initializer=tf.ones_initializer()(shape=[1]))

# with tf.Session(graph = g1) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable("v")))

# with tf.Session(graph = g2) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable("v")))


#张量三个属性 (名字，维度，类型)
g1 = tf.Graph()
with g1.as_default():
    a = tf.constant([1.0, 2.0, 3.0], name = "a")
    b = tf.constant([2.0, 1.0, 0.0], name = "b")
    result = tf.add(a, b, name="add")
    print(result)
with tf.Session(graph = g1) as sess:
    print(result.graph is tf.get_default_graph())
    print(sess.run(result))
    print(result.eval())

g2 = tf.Graph()
with g2.as_default():
    a = tf.constant([1.0, 2.0, 3.0], name = "a")
    b = tf.constant([2.0, 1.0, 0.0], name = "b")
    result = tf.add(a, b, name="add")
    print(result)
with tf.Session(graph = g2) as sess:
    print(result.graph is tf.get_default_graph())
    print(sess.run(result))
    print(result.eval())
