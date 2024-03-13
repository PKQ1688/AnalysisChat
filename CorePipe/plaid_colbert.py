#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/3/13 17:01
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/3/13 17:01
# @File         : plaid_colbert.py
# import os

from RagPro.stores import PLAIDDocumentStore

# os.environ['KMP_DUPLICATE_LIB_OK'] = True

store = PLAIDDocumentStore(index_path="index/",
                           checkpoint_path="model/ColBERT-NQ",
                           collection_path="demo_data/collection.tsv")
