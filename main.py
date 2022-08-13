#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/06/29 12:24:30
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   数据库主函数
'''

import random
import os
import time
import flask
from flask import Flask, jsonify, request, url_for

from db import add_user, check_user, wait_for_admin, auth_admin, auth_root, query_things, add_thing
from model import inference

app = Flask(__name__, static_folder='static', static_url_path="/static")


@app.route('/')
def index():
    return "hello world"

@app.route('/about/')
def about():
    request_data = request.args.to_dict()
    return "this is %s" % request_data['user'].upper()

# @app.route('/recent', methods=['GET'])
# def recent():
#     """
#         发送数据给前端
#         发送格式: json
#             title: str
#             tags: list
#             content: str
#             image: str
#     """
#     dic = {}
#     res = query_things()
    
#     for i, (_, title, year, content, _, _, image) in enumerate(res):
#         if i >= 10:
#             break
#         dic[i] = {
#             'title': title,
#             'tags': [year, ],
#             'content': content,
#             'image': url_for('static', filename= 'data/' + image, _external=True)
#         }
#     # print(dic)
#     return jsonify(dic)

# @app.route('/gallery', methods=['GET'])
# def gallery():
#     year = request.args.get('year')
#     dic = {}
#     res = query_things(year)
#     for i, (_, title, year, content, _, _, image) in enumerate(res):
#         if i >= 10:
#             break
#         dic[i] = {
#             'title': title,
#             'tags': [year, ],
#             'content': content,
#             'image': url_for('static', filename= 'data/' + image, _external=True)
#         }
#     # print(dic)
#     return jsonify(dic)

@app.route('/uploader', methods=['POST'])
def uploader():
    if request.method == 'POST' and 'file' in request.files:
        
        img = request.files.get('file') # 从post请求中获取图片数据
        # print(type(img))
        # 保存图片
        suffix = '.' + img.filename.split('.')[-1] # 获取文件后缀名
        if suffix not in ['.jpg', '.jpeg', '.png', '.bmp']:
            return jsonify(status = 201, msg = '不支持的数据格式')
        basedir = "G:/weixin_app/temp"
        photo = '/' + str(int(time.time())) + suffix # 拼接相对路径
        img_path = basedir + photo # 拼接图片完整保存路径,时间戳命名文件防止重复
        img.save(img_path) # 保存图片
        print(img_path)
        # 进行图片处理
        res = inference(image_path=img_path)
        if res['status'] != 200:
            return jsonify(res)
        res["rec_pth"] = [url_for('static', filename= i, _external=True, _scheme='https') for i in res["rec_pth"]]
        res['msg'] = '检测到物体'
        res['pred_score'] = [int(i*100) for i in res['pred_score']]

        print(res)
        return jsonify(res)
        
# @app.route('/checkAdmin', methods=['POST'])
# def checkAdmin():
#     username = request.form.get('nickName')
#     res = check_user(username)
#     if res is None:
#         add_user(username)
#         isAdmin = 0
#         isRoot = 0
#         isWaitForAdmin = 0
#         return jsonify(
#             status = 201,
#             isAdmin = isAdmin,
#             isRoot = isRoot,
#             isWaitForAdmin = isWaitForAdmin
#         )
#     else:
#         isAdmin = res[2]
#         isRoot = res[3]
#         isWaitForAdmin = res[4]
#         return jsonify(
#             status = 200,
#             isAdmin = isAdmin,
#             isRoot = isRoot,
#             isWaitForAdmin = isWaitForAdmin
#         )
    
# 申请成为管理员
# @app.route('/applyAdmin', methods=['POST'])
# def applyAdmin():
#     username = request.form.get('nickName')
#     wait_for_admin(username)
#     return jsonify(
#         status = 200
#     )
    

# 添加文物
# @app.route('/addThing', methods=['POST'])
# def addThing():
#     title = request.form.get('title')
#     year = request.form.get('year')
#     author = request.form.get('author')
#     content = request.form.get('content')
#     image = request.files.get('image')
#     add_thing(title, year, author, content, image)
#     return jsonify(
#         status = 200
#     )

def get_bbox(img):
    """获取中心框
    """
    from PIL import Image
    im = Image.open(img)
    im_w, im_h = im.size
    return [int(im_w/4), int(im_h/4), int(3*im_w/4), int(3*im_h/4)]

if __name__ == '__main__':
    app.run(host="0.0.0.0")
    