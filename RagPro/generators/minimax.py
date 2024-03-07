#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/3/7 20:00
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/3/7 20:00
# @File         : minimax.py
import requests
# from loguru import logger
# from pprint import pformat
from typing import Optional, Dict, Any, List

from haystack import component, default_to_dict, default_from_dict
# from haystack.dataclasses import ChatMessage

from API_KEY import minimax_group_id, minimax_api_key


@component
class MiniMaxGenerator:
    def __init__(
            self,
            group_id: str = None,
            api_key: str = None,
            api_base_url: Optional[str] = "https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId=",
            model: str = "abab5.5-chat",
            bot_setting: Optional[dict] = None,
            tokens_to_generate: int = 1024,
            temperature: float = 0.01,
            top_p: float = 0.95,
    ):
        if group_id is None:
            self.group_id = minimax_group_id
        else:
            self.group_id = group_id

        if api_key is None:
            self.api_key = minimax_api_key
        else:
            self.api_key = api_key

        self.api_base_url = api_base_url
        self.model = model

        if bot_setting is None:
            self.bot_setting = [
                {
                    "bot_name": "aileap.chat",
                    "content": "aileap智能助理是一款没有调用其他产品的接口的大型语言模型。",
                }
            ]
        else:
            self.bot_setting = bot_setting

        self.tokens_to_generate = tokens_to_generate
        self.temperature = temperature
        self.top_p = top_p

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            model=self.model,
            api_base_url=self.api_base_url,
            bot_setting=self.bot_setting,
            tokens_to_generate=self.tokens_to_generate,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MiniMaxGenerator":
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str, history: List = None):
        if history is None:
            history = []
        # message = ChatMessage.from_user(prompt)
        # logger.debug(message)

        history.append({"sender_type": "USER", "sender_name": "aileap.use", "text": prompt})

        payload = {
            "bot_setting": self.bot_setting,
            "messages": history,
            "reply_constraints": {"sender_type": "BOT", "sender_name": "aileap.chat"},
            "model": self.model,
            "tokens_to_generate": self.tokens_to_generate,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        url = self.api_base_url + self.group_id
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        response = requests.post(url, headers=headers, json=payload)
        response = response.json()

        # logger.info(pformat(response))

        reply = response["reply"]
        history.extend(response["choices"][0]["messages"])

        response["history"] = history

        return {
            "replies": [reply],
            "meta": [response],
        }


if __name__ == '__main__':
    from rich import print
    client = MiniMaxGenerator()
    res = client.run("What's Natural Language Processing? Be brief.")
    print(res)
