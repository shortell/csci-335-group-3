import os
import httpx
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from google import genai
from google.genai import types as genai_types

BASE_DIR   = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / 'data' / 'cleaned' / 'pipeline_output.csv'
BASE_OUTPUT_PATH = BASE_DIR / 'data' / 'vector_embeddings'  # provider subfolder added dynamically

openAI_client = OpenAI(http_client=httpx.Client())
google_client = genai.Client()

openAI_embedding_model = 'text-embedding-3-small'
google_embedding_model = 'models/text-embedding-004'


def embed_batch_openai(client, model, batch):
    response = client.embeddings.create(input=batch, model=model)
    return [item.embedding for item in response.data]


def embed_batch_google(client, model, batch):
    response = client.models.embed_content(
        model=model,
        contents=batch,
        config=genai_types.EmbedContentConfig(task_type='SEMANTIC_SIMILARITY')
    )
    return [e.values for e in response.embeddings]


EMBED_FN = {
    openAI_embedding_model: (embed_batch_openai, 'open_ai'),
    google_embedding_model:  (embed_batch_google, 'google'),
}


def embed_tweets(client, embedding_model):
    if embedding_model not in EMBED_FN:
        raise ValueError(f"Unknown embedding model '{embedding_model}'. "
                         f"Register it in EMBED_FN first.")

    embed_batch, provider_folder = EMBED_FN[embedding_model]  # unpack both values
    output_path = BASE_OUTPUT_PATH / provider_folder           # e.g. vector_embeddings/google

    df    = pd.read_csv(INPUT_PATH)
    texts = df['cleanText'].fillna('').tolist()
    print(f"Embedding {len(texts)} tweets with model '{embedding_model}' ...")

    BATCH_SIZE     = 256
    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vecs  = embed_batch(client, embedding_model, batch)
        all_embeddings.extend(vecs)
        print(f"  {min(start + BATCH_SIZE, len(texts))} / {len(texts)}")

    embeddings_array = np.array(all_embeddings, dtype=np.float32)

    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f'{embedding_model}.npz'
    np.savez_compressed(
        out_file,
        embeddings=embeddings_array,
        row_ids   =df['row_id'].to_numpy(dtype=np.int64),
        tweet_ids =df['tweet_id'].to_numpy(dtype=np.int64),
        timestamps=df['tweet_timestamp'].to_numpy(dtype=str),
    )
    size_mb = out_file.stat().st_size / 1e6
    print(f"Saved {len(all_embeddings)} embeddings ({embeddings_array.shape[1]}d) "
          f"→ '{out_file}' ({size_mb:.1f} MB)")


if __name__ == '__main__':
    # embed_tweets(openAI_client, openAI_embedding_model)
    embed_tweets(google_client, google_embedding_model)