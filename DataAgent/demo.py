#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/2/24 22:11
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/2/24 22:11
# @File         : demo.py
from taskweaver.app.app import TaskWeaverApp

# This is the folder that contains the taskweaver_config.json file and not the repo root. Defaults to "./project/"
app_dir = "project/"
app = TaskWeaverApp(app_dir=app_dir)
session = app.get_session()

user_query = "hello, what can you do?"
response_round = session.send_message(user_query)
print(response_round.to_dict())
