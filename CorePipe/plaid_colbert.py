#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/3/13 17:01
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/3/13 17:01
# @File         : plaid_colbert.py
# from RagPro.stores import PLAIDDocumentStore
#
# store = PLAIDDocumentStore(index_path="index/test/",
#                            checkpoint_path="model/ColBERT-NQ",
#                            collection_path="demo_data/collection.tsv",
#                            create=True)

from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig

if __name__ == '__main__':
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            nbits=2,
            root="index/test/",
        )
        indexer = Indexer(checkpoint="model/ColBERT-NQ", config=config)
        indexer.index(name="msmarco.nbits=2", collection="demo_data/collection.tsv")
