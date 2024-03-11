#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/3/7 18:16
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/3/7 18:16
# @File         : SampleRAG.py
import pandas as pd
from haystack import Document
from haystack import Pipeline
from haystack import components
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from RagPro.generators.minimax import MiniMaxGenerator

from pathlib import Path

pd.set_option('display.max_columns', None)

document_store = InMemoryDocumentStore()

dataset = pd.read_parquet("demo_data/qa_data.parquet")
docs = [Document(content=doc["content"], meta=doc["meta"]) for _, doc in dataset.iterrows()]
document_store.write_documents(docs)

retriever = InMemoryBM25Retriever(document_store)

template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)

generator = MiniMaxGenerator()

basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

basic_rag_pipeline.draw(Path("basic-rag-pipeline.png"))

question = "What does Rhodes Statue look like?"

response = basic_rag_pipeline.run({"retriever": {"query": question}, "prompt_builder": {"question": question}})

print(response["llm"]["replies"][0])
