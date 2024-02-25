#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/2/25 16:51
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/2/25 16:51
# @File         : core.py
import datetime
import json
import sys

from loguru import logger

from LLM.minimax import get_llm_response

logger.remove()
logger.add(sys.stderr, level='INFO')


def run(query: str = ""):
    # step1: intent detection
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d")

    logger.info("current time: {}".format(formatted_time))

    with open('PromptLib/intent_detection.json', 'r') as f:
        prompt_task_dict = json.load(f)

    prompt_intent_detection = ""
    for key, value in prompt_task_dict.items():
        prompt_intent_detection += key + ": " + value + '\n\n'

    logger.debug(prompt_intent_detection)
    prompt_intent_detection += '\n\n' + 'Instruction:' + '今天的日期是' + formatted_time + ', ' + query + ' ###New Instruction: '
    logger.debug(prompt_intent_detection)

    intent_detection_res = get_llm_response(message=prompt_intent_detection)
    logger.info(intent_detection_res)

    # step2: task planing stage


if __name__ == '__main__':
    run(query="去年三季度的评分相比于去年二季度如何变化？")
