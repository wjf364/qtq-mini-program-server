#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   db.py
@Time    :   2022/06/28 18:39:19
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   数据库相关操作
'''
import os
import random
import sqlite3
from PIL import Image


from utils import base64_to_img, get_random_file, img_to_base64, save_image

db_path = './database/qtq.sqlite'

def check_user(username):
    '''
    检查是否存在用户
    如果没有录入用户，则返回None
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_table WHERE nickname=?", (username, ))
    res = cursor.fetchone()
    conn.close()
    return res 

def add_user(username):
    '''
    添加用户（无管理权限）
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_table (nickname, isAdmin, isRoot, wait_for_admin) VALUES (?, ?, ?, ?)", (username, 0, 0, 0))
    conn.commit()
    conn.close()

# 授权管理员
def auth_admin(username):
    '''
    授权管理员权限
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE user_table SET isAdmin=1 WHERE nickname=?", (username, ))
    cursor.execute("UPDATE user_table SET wait_for_admin=0 WHERE nickname=?", (username, ))
    conn.commit()
    conn.close()

# 授权ROOT
def auth_root(username):
    '''
    授权ROOT权限
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE user_table SET isRoot=1 WHERE nickname=?", (username, ))
    cursor.execute("UPDATE user_table SET isAdmin=1 WHERE nickname=?", (username, ))
    cursor.execute("UPDATE user_table SET wait_for_admin=0 WHERE nickname=?", (username, ))
    conn.commit()
    conn.close()

# 等待管理员授权
def wait_for_admin(username):
    '''
    等待管理员授权
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE user_table SET wait_for_admin=1 WHERE nickname=?", (username, ))
    conn.commit()
    conn.close()
    
# 添加文物信息
def add_thing(title, year, author, content, image):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    filename = save_image(image, info="{}-{}".format(title, year), path="static/data")
    cursor.execute("INSERT INTO qtq_table (title, year, author, content, image, time) VALUES (?, ?, ?, ?, ?, datetime('now'))", (title, year, author, content, filename))
    conn.commit()
    conn.close()


# 查询文物信息
# 按最新添加的顺序排序
# 年代选项
def query_things(year = None, sort_by_time=True):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if year is None:
        cursor.execute("SELECT * FROM qtq_table ORDER BY time DESC")
        res = cursor.fetchall()
    else:
        cursor.execute("SELECT * FROM qtq_table WHERE year=? ORDER BY time DESC", (year, ))
        res = cursor.fetchall()
    conn.close()
    return res


def test_user_table():
    if check_user('唐川') is None:
        add_user('唐川')
        auth_root('唐川')
    print(check_user('唐川'))

def test_qtq_table():
    # 随机生成一些文物信息
    years = [
            "秦朝",
            "汉朝",
            "春秋",
            "商朝",
            "唐朝",
            "宋朝",
            "元朝",
            "明朝",
            "清朝",
        ]
    for i in range(100):
        year = random.choice(years)
        title = year + "时期" + str(i+1).zfill(5) + '号文物'
        content = "这是一个来自{}的文物, 编号: {}".format(year, str(i+1).zfill(5))
        author = "唐川"
        image = get_random_file(path='./static/images')
        image = Image.open(image)
        add_thing(title, year, author, content, image)
    
    # print(query_things())
    # print(query_things(year='清朝'))
    # print(query_things(year='唐朝', sort_by_time=False))
if __name__ == '__main__':
    # for test
    conn = sqlite3.connect(db_path)
    # create table
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS user_table (id INTEGER PRIMARY KEY AUTOINCREMENT,nickname TEXT UNIQUE,isAdmin INTEGER,isRoot INTEGER, wait_for_admin INTEGER)""")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS qtq_table 
              (id INTEGER PRIMARY KEY AUTOINCREMENT,title TEXT,year TEXT,content TEXT,author TEXT,time TEXT,image TEXT)""")
    conn.close()
    
    # 测试用户管理
    test_user_table()
    
    # 测试文物信息管理
    test_qtq_table()
