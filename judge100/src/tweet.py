#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tweepy
import urllib.request
import sys
import datetime
import re
#from pillow import Image
import cv2
import sys
import os.path
import numpy as np
#import skimage
import copy
#import dlib
#import scipy
import tensorflow as tf

NUM_CLASSES = 5
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

def inference(images_placeholder, keep_prob):
    """ モデルを作成する関数

    引数:
      images_placeholder: inputs()で作成した画像のplaceholder
      keep_prob: dropout率のplace_holder

    返り値:
      cross_entropy: モデルの計算結果
    """
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(images_placeholder, [-1, 28, 28, 3])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

# mitra_sun22のログイン情報
f = open('config.txt')
data = f.read()
f.close()
lines = data.split('\n')

# 顔検出器
#detector = dlib.simple_object_detector("detector.svm")

images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
keep_prob = tf.placeholder("float")

logits = inference(images_placeholder, keep_prob)
sess = tf.InteractiveSession()

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
saver.restore(sess, "E:\python\pleiades\workspace\machine/model.ckpt")

# エンコード設定
#reload(sys)
#sys.setdefaultencoding('utf-8')

def get_oauth():
    consumer_key = lines[0]
    consumer_secret = lines[1]
    access_key = lines[2]
    access_secret = lines[3]
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    return auth

class StreamListener(tweepy.StreamListener):
    # ツイートされるたびにここが実行される
    def on_status(self, status):
        if status.in_reply_to_screen_name=='*****':
            print(status)
            if 'media' in status.entities :
                text = re.sub(r'@****** ', '', status.text)
                text = re.sub(r'(https?|ftp)(://[\w:;/.?%#&=+-]+)', '', text)
                medias = status.entities['media']
                m =  medias[0]
                media_url = m['media_url']
                print(media_url)
                now = datetime.datetime.now()
                time = now.strftime("%H%M%S")
                filename = '{}.jpg'.format(time)
                try:
                    urllib.request.urlretrieve(media_url, filename)
                except IOError:
                    print("保存に失敗しました")

                frame = cv2.imread(filename)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #顔の検出
                #dets = detector(img)
                #height, width = img.shape[:2]
                #flag = True
                # 形式を変換
                img = cv2.resize(img.copy(), (28, 28))
                ximage = img.flatten().astype(np.float32)/255.0

                pred = np.argmax(logits.eval(feed_dict={
                    images_placeholder: [ximage],
                    keep_prob: 1.0 })[0])
                print(pred)

                if pred==0: #動画工房の場合
                    print("無印カンスト")
                    message = '.@'+status.author.screen_name+'カンストおめでとうございます！'
                elif pred==1:
                    print("無印カンスト失敗")
                    message = '.@'+status.author.screen_name+'頑張ってください'
                elif pred==2:
                    print("ソテカン")
                    message = '.@'+status.author.screen_name+'ソテカンおめでとうございます！'
                elif pred==3:
                    print("ボナカン")
                    message = '.@'+status.author.screen_name+'ボナカンおめでとうございます！'
                elif pred==4:
                    print("カンスト失敗")
                    message = '.@'+status.author.screen_name+'頑張ってください'
                else:
                    print("")
                    message = '.@'+status.author.screen_name+' '
                #message = message.decode("utf-8")
                try:
                    #画像をつけてリプライ
                    api.update_with_media(filename, status=message, in_reply_to_status_id=status.id)
                except tweepy.TweepError as e:
                    print("error response code: " + str(e.response.status))
                    print("error message: " + str(e.response.reason))

# streamingを始めるための準備
auth = get_oauth()
api = tweepy.API(auth)
stream = tweepy.Stream(auth, StreamListener(), secure=True)
print ("Start Streaming!")
stream.userstream()