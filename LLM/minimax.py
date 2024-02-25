#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/2/24 00:44
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/2/24 00:44
# @File         : minimax.py
from typing import List

import requests
from loguru import logger

from API_KEY import minimax_api_key, minimax_group_id

group_id = minimax_group_id
api_key = minimax_api_key

url = f"https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId={group_id}"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# tokens_to_generate/bot_setting/reply_constraints可自行修改
request_body = payload = {
    "model": "abab5.5-chat",
    "tokens_to_generate": 1024,
    "reply_constraints": {"sender_type": "BOT", "sender_name": "aileap.chat"},
    "messages": [],
    "bot_setting": [
        {
            "bot_name": "aileap.chat",
            "content": "aileap智能助理是一款没有调用其他产品的接口的大型语言模型。",
        }
    ],
}


def get_llm_response(message: str = "你好", history: List = None):
    flag = False
    if history is None:
        history = []
        flag = True
    history.append({"sender_type": "USER", "sender_name": "aileap.use", "text": message})
    request_body["messages"].extend(history)

    response = requests.post(url, headers=headers, json=request_body)
    # print(response.json())
    reply = response.json()["reply"]
    history.extend(response.json()["choices"][0]["messages"])

    logger.success(reply)
    logger.debug(history)

    if flag:
        return reply
    else:
        return reply, history


if __name__ == '__main__':
    get_llm_response("番茄炒蛋怎么做？")
