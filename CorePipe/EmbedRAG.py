#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/3/12 14:07
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/3/12 14:07
# @File         : EmbedRAG.py
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice


embeder = SentenceTransformersTextEmbedder(meta_fields_to_embed=["url"])


def create_indexing_pipeline(document_store, metadata_fields_to_embed=None):
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="sentence", split_length=2)
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="thenlper/gte-large",
        meta_fields_to_embed=metadata_fields_to_embed,
        device=ComponentDevice.from_str("cuda:0"),
    )
    document_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("cleaner", document_cleaner)
    indexing_pipeline.add_component("splitter", document_splitter)
    indexing_pipeline.add_component("embedder", document_embedder)
    indexing_pipeline.add_component("writer", document_writer)

    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    return indexing_pipeline
