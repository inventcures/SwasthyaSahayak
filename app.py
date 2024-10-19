# Import necessary libraries and modules
import json
import groq
import cohere
from scipy.spatial.distance import cosine
import numpy as np
import math
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import openai
import psycopg2
from flask import Flask, request, jsonify, send_file, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from gtts import gTTS
import tempfile
import queue
from pinecone import Pinecone
import requests

from dotenv import load_dotenv

# Import FastEmbed for text embedding
from fastembed import TextEmbedding

# Import Qdrant client for vector database operations
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Import WhatsApp integration module
from whatsapp_integration import handle_whatsapp_message

# Import Arize Phoenix and OpenInference
from arize.phoenix.logger import Logger
from openinference.instrumentation.groq import GroqInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

# Initialize FastEmbed model for text embedding
embedding_model = TextEmbedding('BAAI/bge-small-en-v1.5')

# Initialize Qdrant client for vector database operations
qdrant_client = QdrantClient(
    url="https://1a2e5d0e-9a9f-4f7e-8e8e-e1c5d0c7c7c9.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="5cCQ4gIgRIdhr1mZ6coBBOH6pi7RzKptijF-YeKwYmld4E7qrPjwig"
)

# Initialize Arize Phoenix logger
phoenix_logger = Logger(
    api_key="your_api_key_here",
    uri=PHOENIX_COLLECTOR_ENDPOINT,
    project="default"
)

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)

# Initialize Groq Instrumentation
GroqInstrumentor().instrument()

def process_audio_folder(folder_path, embedding_dim=384):
    """
    Process audio files in a given folder, generate embeddings, and store them in Qdrant.
    
    Args:
    folder_path (str): Path to the folder containing audio files
    embedding_dim (int): Dimension of the embedding vector (default: 384)
    
    Returns:
    str: Status message
    """
    logger.info(f"Processing audio files in folder: {folder_path}")
    try:
        # Ensure the Qdrant collection exists
        collection_name = "audio_embeddings"
        try:
            qdrant_client.get_collection(collection_name)
        except:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.mp3'):
                video_id = file_name.split('.')[0]
                logger.info(f"Processing file: {file_name}")
                
                # Check if embeddings exist in Qdrant
                search_result = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=[0] * embedding_dim,  # Dummy vector for existence check
                    filter={"must": [{"key": "video_id", "match": {"value": video_id}}]},
                    limit=1
                )
                
                if search_result:
                    logger.info(f"Skipping {video_id}, as embeddings already exist in Qdrant")
                    continue
                
                file_path = os.path.join(folder_path, file_name)
                
                # Generate embedding
                with open(file_path, 'rb') as audio_file:
                    audio_content = audio_file.read()
                embedding = embedding_model.embed(audio_content)
                
                # Store embedding in Qdrant
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[PointStruct(
                        id=video_id,
                        vector=embedding.tolist(),
                        payload={"video_id": video_id}
                    )]
                )
                
                logger.info(f"Processed and stored embeddings for {video_id}")
        
        logger.info("All audio files in the folder have been processed")
        return "All audio files processed successfully"
    except Exception as e:
        logger.error(f"Error processing audio folder: {str(e)}", exc_info=True)
        raise
