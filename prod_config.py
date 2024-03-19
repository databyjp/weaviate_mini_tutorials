import weaviate
import os
import weaviate.classes.config as wc
import pandas as pd
import requests
from weaviate.util import generate_uuid5
from tqdm import tqdm
import json
from datetime import datetime, timezone

headers = {
    "X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY"),
    "X-Cohere-Api-Key": os.getenv("COHERE_APIKEY")
}  # Replace with your OpenAI API key

client = weaviate.connect_to_local(
    headers=headers,
)

client.collections.delete("Movie")

client.collections.create(
    name="Movie",
    properties=[
        wc.Property(name="title", data_type=wc.DataType.TEXT),
        wc.Property(name="overview", data_type=wc.DataType.TEXT),
        wc.Property(name="vote_average", data_type=wc.DataType.NUMBER),
        wc.Property(name="genre_ids", data_type=wc.DataType.INT_ARRAY),
        wc.Property(name="release_date", data_type=wc.DataType.DATE),
        wc.Property(name="tmdb_id", data_type=wc.DataType.INT),
    ],
    vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(),
    generative_config=wc.Configure.Generative.openai(),
    replication_config=wc.Configure.replication(factor=3),
    vector_index_config=wc.Configure.VectorIndex.hnsw(
        quantizer=wc.Configure.VectorIndex.Quantizer.pq()
    ),
    sharding_config=wc.Configure.sharding(
        desired_count=3,
    )
)