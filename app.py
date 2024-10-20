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
import google.generativeai as genai
from PIL import Image
import io

from dotenv import load_dotenv

# Import FastEmbed for text embedding
from fastembed import TextEmbedding

# Import Qdrant client for vector database operations
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Import WhatsApp integration module
from whatsapp_integration import handle_whatsapp_message

# Initialize FastEmbed model for text embedding
embedding_model = TextEmbedding('BAAI/bge-small-en-v1.5')

# Initialize Qdrant client for vector database operations
qdrant_client = QdrantClient(
    url="https://1a2e5d0e-9a9f-4f7e-8e8e-e1c5d0c7c7c9.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="5cCQ4gIgRIdhr1mZ6coBBOH6pi7RzKptijF-YeKwYmld4E7qrPjwig"
)

# Initialize Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-pro-vision-latest')

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
        create_qdrant_collection_if_not_exists(collection_name, embedding_dim)

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.mp3'):
                video_id = file_name.split('.')[0]
                logger.info(f"Processing file: {file_name}")
                
                if embedding_exists_in_qdrant(collection_name, video_id, embedding_dim):
                    logger.info(f"Skipping {video_id}, as embeddings already exist in Qdrant")
                    continue
                
                file_path = os.path.join(folder_path, file_name)
                process_audio_file(file_path, video_id, collection_name, embedding_dim)
        
        logger.info("All audio files in the folder have been processed")
        return "All audio files processed successfully"
    except Exception as e:
        logger.error(f"Error processing audio folder: {str(e)}", exc_info=True)
        raise

def create_qdrant_collection_if_not_exists(collection_name, embedding_dim):
    """Create Qdrant collection if it doesn't exist."""
    try:
        qdrant_client.get_collection(collection_name)
    except:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )

def embedding_exists_in_qdrant(collection_name, video_id, embedding_dim):
    """Check if embedding exists in Qdrant."""
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=[0] * embedding_dim,  # Dummy vector for existence check
        filter={"must": [{"key": "video_id", "match": {"value": video_id}}]},
        limit=1
    )
    return bool(search_result)

def process_audio_file(file_path, video_id, collection_name, embedding_dim):
    """Process a single audio file and store its embedding in Qdrant."""
    with open(file_path, 'rb') as audio_file:
        audio_content = audio_file.read()
    embedding = embedding_model.embed(audio_content)
    
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[PointStruct(
            id=video_id,
            vector=embedding.tolist(),
            payload={"video_id": video_id}
        )]
    )
    
    logger.info(f"Processed and stored embeddings for {video_id}")

# Load environment variables from .env file
load_dotenv()

# Set API keys and other configuration variables from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
SARVAM_API_KEY = "3e1dc970-b034-40b1-be06-159ff8fecb0c"
QDRANT_API_KEY = "5cCQ4gIgRIdhr1mZ6coBBOH6pi7RzKptijF-YeKwYmld4E7qrPjwig"

# Database configuration
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

# Update database connection parameters
db_params = {
    'dbname': DB_NAME,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'host': DB_HOST,
    'port': int(DB_PORT) if DB_PORT else None  # Convert to int if provided, else None
}

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def setup_logging():
    """
    Set up logging configuration for the application.
    
    Returns:
    logging.Logger: Configured logger object
    """
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = 'health_app.log'
    log_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    log_handler.setFormatter(log_formatter)
    logger = logging.getLogger('HealthAppLogger')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(log_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

def initialize_api_clients():
    """
    Initialize API clients for various services (OpenAI, Groq, Cohere, Pinecone, Qdrant, FastEmbed).
    
    Returns:
    tuple: Initialized API clients and models
    """
    try:
        logger.info("Initializing API clients")
        
        # OpenAI
        openai.api_key = OPENAI_API_KEY
        logger.debug("OpenAI API key set")

        # Groq
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
        logger.debug("Groq client initialized")

        # Cohere
        co = cohere.Client(COHERE_API_KEY)
        logger.debug("Cohere client initialized")

        # Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        pinecone_index.describe_index_stats()
        logger.info("Pinecone connection successful")

        # Qdrant
        qdrant_client = QdrantClient("https://tp53_meta_llama.eu-central-1-0.aws.cloud.qdrant.io:6333", api_key=QDRANT_API_KEY)
        logger.info("Qdrant connection successful")

        # FastEmbed
        embedding_model = TextEmbedding()
        logger.info("FastEmbed model initialized")

        index_stats = pinecone_index.describe_index_stats()
        logger.debug(f"Index dimension: {index_stats['dimension']}")

        logger.info("All API clients initialized successfully")
        return openai, groq_client, co, pinecone_index, qdrant_client, embedding_model
    except Exception as e:
        logger.error(f"Error initializing API clients: {str(e)}", exc_info=True)
        sys.exit(1)

openai, groq_client, co, pinecone_index, qdrant_client, embedding_model = initialize_api_clients()

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    
    Returns:
    psycopg2.extensions.connection: Database connection object
    """
    logger.debug("Attempting to establish database connection")
    try:
        conn = psycopg2.connect(**db_params)
        logger.info("Database connection established successfully")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}", exc_info=True)
        raise

def create_tables():
    """
    Create necessary tables in the database if they don't exist.
    """
    logger.info("Creating tables if they don't exist")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                logger.debug("Executing CREATE TABLE query for video_chunks")
                cur.execute("""
                CREATE TABLE IF NOT EXISTS video_chunks (
                    id SERIAL PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    start_time FLOAT NOT NULL,
                    end_time FLOAT NOT NULL,
                    text TEXT NOT NULL,
                    embedding_id TEXT NOT NULL,
                    language TEXT NOT NULL
                )
                """)            
        conn.commit()
        logger.info("Tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}", exc_info=True)
        raise

# Global queue for SSE messages
sse_queue = queue.Queue()

@app.route('/')
def serve_index():
    """Serve the index.html file"""
    return send_from_directory('frontend', 'index.html')

@app.route('/style.css')
def serve_css():
    """Serve the style.css file"""
    return send_from_directory('frontend', 'style.css')

@app.route('/script.js')
def serve_js():
    """Serve the script.js file"""
    return send_from_directory('frontend', 'script.js')

# Define supported languages
SUPPORTED_LANGUAGES = {
    'hi': 'Hindi',
    'bn': 'Bengali',
    'te': 'Telugu',
    'ta': 'Tamil',
    'mr': 'Marathi',
    'ur': 'Urdu',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'en': 'English'  # Added English for completeness
}

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe streaming audio using Whisper API or Sarvam API and push results to SSE queue.
    
    Returns:
    flask.Response: JSON response with transcription status
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    use_sarvam = request.form.get('use_sarvam', 'false').lower() == 'true'
    language = request.form.get('language', 'en')
    
    try:
        if use_sarvam:
            transcript_chunk, detected_language = transcribe_with_sarvam(audio_file, language)
        else:
            transcript_chunk, detected_language = transcribe_with_whisper(audio_file, language)
        
        sse_queue.put(json.dumps({'text': transcript_chunk, 'language': detected_language}))
        return jsonify({'success': True}), 200
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        return jsonify({'error': 'Transcription error'}), 500

def transcribe_with_sarvam(audio_file, language):
    """Transcribe audio using Sarvam API."""
    logger.info("Using Sarvam API for transcription")
    url = "https://api.sarvam.ai/v1/speech/transcribe"
    headers = {"Authorization": f"Bearer {SARVAM_API_KEY}"}
    files = {"file": (audio_file.filename, audio_file.stream, audio_file.content_type)}
    data = {"language": language}
    response = requests.post(url, headers=headers, files=files, data=data)
    response.raise_for_status()
    result = response.json()
    return result['text'], result['detected_language']

def transcribe_with_whisper(audio_file, language):
    """Transcribe audio using OpenAI Whisper API."""
    logger.info("Using OpenAI Whisper API for transcription")
    response = openai.Audio.transcribe("whisper-1", audio_file, language=language)
    return response['text'], language

@app.route('/transcribe_stream')
def transcribe_stream():
    """
    SSE endpoint for streaming transcription results.
    
    Returns:
    flask.Response: Server-Sent Events (SSE) response
    """
    def generate():
        while True:
            try:
                message = sse_queue.get(timeout=30)  # 30 second timeout
                yield f"data: {message}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'text': ''})}\n\n"  # Send empty message to keep connection alive

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    """
    Convert text to speech using either Sarvam API or OpenAI TTS API.
    
    Returns:
    flask.Response: Audio file response or error message
    """
    logger.info("Received text-to-speech request")
    try:
        data = request.json
        text = data['text']
        language = data.get('language', 'en')
        use_sarvam = data.get('use_sarvam', False)
        
        if use_sarvam:
            temp_audio_path = text_to_speech_sarvam(text, language)
        else:
            temp_audio_path = text_to_speech_openai(text, language)
        
        logger.info("Text-to-speech conversion completed")
        return send_file(temp_audio_path, mimetype='audio/mp3')
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def text_to_speech_sarvam(text, language):
    """Convert text to speech using Sarvam API."""
    logger.debug("Using Sarvam API for text-to-speech")
    tts_url = "https://api.sarvam.ai/v1/speech/tts"
    tts_headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json"
    }
    tts_payload = {
        "text": text,
        "language": language
    }
    tts_response = requests.post(tts_url, headers=tts_headers, json=tts_payload)
    tts_response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
        temp_audio.write(tts_response.content)
        return temp_audio.name

def text_to_speech_openai(text, language):
    """Convert text to speech using OpenAI TTS API."""
    logger.debug("Using OpenAI TTS API for text-to-speech")
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        language=language
    )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
        for chunk in response.iter_bytes(chunk_size=4096):
            temp_audio.write(chunk)
        return temp_audio.name

def get_processed_video_ids():
    """
    Retrieve a list of processed video IDs from the database.
    
    Returns:
    list: List of processed video IDs
    """
    logger.info("Getting a list of processed video IDs from the database")
    processed_video_ids = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT video_id FROM video_chunks")
                rows = cur.fetchall()
                processed_video_ids = [row[0] for row in rows]
        logger.info(f"Found {len(processed_video_ids)} processed video IDs")
    except Exception as e:
        logger.error(f"Error getting processed video IDs: {str(e)}", exc_info=True)
    return processed_video_ids

def process_audio_folder(folder_path):
    """
    Process all audio files in a given folder.
    
    Args:
    folder_path (str): Path to the folder containing audio files
    
    Returns:
    str: Status message
    """
    logger.info(f"Processing audio files in folder: {folder_path}")
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.mp3'):
                video_id = file_name.split('.')[0]
                logger.info(f"Processing file: {file_name}")
                
                if check_video_processed(video_id):
                    logger.info(f"Skipping {video_id}, as transcript, chunks & chunk embeddings exist in both postgres & vector db")
                    logger.debug(f"Skipping {video_id}, as transcript, chunks & chunk embeddings exist in both postgres & vector db")
                    continue
                
                file_path = os.path.join(folder_path, file_name)
                process_local_audio(file_path)
        
        logger.info("All audio files in the folder have been processed")
        return "All audio files processed successfully"
    except Exception as e:
        logger.error(f"Error processing audio folder: {str(e)}", exc_info=True)
        raise

def check_video_processed(video_id):
    """
    Check if a video has been processed and stored in the database.
    
    Args:
    video_id (str): ID of the video to check
    
    Returns:
    bool: True if video has been processed, False otherwise
    """
    logger.debug(f"Checking if video {video_id} has been processed")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM video_chunks WHERE video_id = %s", (video_id,))
                count = cur.fetchone()[0]
                
                if count > 0:
                    logger.debug(f"Video {video_id} found in database")
                    return True
                else:
                    logger.debug(f"Video {video_id} not found in database")
                    return False
    except Exception as e:
        logger.error(f"Error checking video processing status: {str(e)}", exc_info=True)
        raise

def process_local_audio(file_path):
    """
    Process a local audio file: transcribe, chunk, and store in database and vector store.
    
    Args:
    file_path (str): Path to the local audio file
    
    Returns:
    str: Status message
    """
    logger.info(f"Starting to process local audio file: {file_path}")
    try:
        video_id = os.path.basename(file_path).split('.')[0]
        logger.info(f"Video ID: {video_id}")

        # Transcribe using Whisper API
        logger.info("Starting transcription with Whisper API")
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

        logger.info("Transcription completed")

        # Process and store the transcript
        logger.info("Chunking transcript")
        chunks = chunk_transcript(transcript['text'])
        logger.info(f"Transcript chunked into {len(chunks)} parts")

        logger.info("Storing chunks and embeddings")
        store_chunks_and_embeddings(video_id, chunks)

        logger.info(f"Audio file {file_path} processed successfully")
        return f"Audio file {file_path} processed successfully"
    except Exception as e:
        logger.error(f"Error processing audio file {file_path}: {str(e)}", exc_info=True)
        raise

def chunk_transcript(transcript, chunk_duration=60):
    """
    Chunk a transcript into smaller segments.
    
    Args:
    transcript (str): Full transcript text
    chunk_duration (int): Desired duration of each chunk in seconds
    
    Returns:
    list: List of chunk dictionaries
    """
    logger.debug(f"Chunking transcript with duration {chunk_duration} seconds")
    words = transcript.split()
    logger.debug(f"Transcript contains {len(words)} words")
    chunks = []
    current_chunk = []
    current_duration = 0
    
    for word in words:
        current_chunk.append(word)
        current_duration += len(word) / 5  # Rough estimate: 5 characters per second
        
        if current_duration >= chunk_duration:
            chunk_text = " ".join(current_chunk)
            chunk = {
                "text": chunk_text,
                "start_time": len(chunks) * chunk_duration,
                "end_time": (len(chunks) + 1) * chunk_duration,
                "language": "auto"  # Whisper auto-detects language
            }
            chunks.append(chunk)
            logger.debug(f"Created chunk {len(chunks)}: {len(chunk_text)} characters")
            current_chunk = []
            current_duration = 0
    
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk = {
            "text": chunk_text,
            "start_time": len(chunks) * chunk_duration,
            "end_time": (len(chunks) + 1) * chunk_duration,
            "language": "auto"
        }
        chunks.append(chunk)
        logger.debug(f"Created final chunk {len(chunks)}: {len(chunk_text)} characters")
    
    logger.info(f"Chunking complete. Created {len(chunks)} chunks")
    return chunks

def store_chunks_and_embeddings(video_id, chunks):
    """
    Store transcript chunks and their embeddings in the database and vector store.
    
    Args:
    video_id (str): ID of the video
    chunks (list): List of chunk dictionaries
    """
    logger.info(f"Storing chunks and embeddings for video {video_id}")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for i, chunk in enumerate(chunks):
                    logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                    
                    logger.debug("Generating embedding using FastEmbed")
                    embedding = embedding_model.embed(chunk["text"])[0].tolist()
                    logger.debug("Embedding generated")
                    
                    embedding_id = f"{video_id}_{chunk['start_time']}"
                    logger.debug(f"Upserting embedding {embedding_id} to Qdrant")
                    try:
                        qdrant_client.upsert(
                            collection_name="video_chunks",
                            points=[
                                {
                                    "id": embedding_id,
                                    "vector": embedding,
                                    "payload": {"text": chunk["text"], "video_id": video_id}
                                }
                            ]
                        )
                        logger.debug(f"Embedding upserted to Qdrant with metadata: {{'text': (chunk text), 'video_id': {video_id}}}")
                    except Exception as e:
                        logger.error(f"Error during Qdrant upsert: {str(e)}")
                    
                    logger.debug("Inserting chunk into PostgreSQL")
                    cur.execute("""
                    INSERT INTO video_chunks (video_id, start_time, end_time, text, embedding_id, language)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """, (video_id, chunk["start_time"], chunk["end_time"], chunk["text"], embedding_id, chunk["language"]))
                    logger.debug("Chunk inserted into PostgreSQL")
                
        conn.commit()
        logger.info(f"All chunks and embeddings for video {video_id} stored successfully")
    except Exception as e:
        logger.error(f"Error storing chunks and embeddings: {str(e)}", exc_info=True)
        raise

def generate_youtube_url(video_id, start_time, end_time):
    """
    Generate a YouTube URL for a specific video segment.
    
    Args:
    video_id (str): YouTube video ID
    start_time (float): Start time of the segment in seconds
    end_time (float): End time of the segment in seconds
    
    Returns:
    str: YouTube URL for the specified video segment
    """
    base_url = f"https://www.youtube.com/embed/{video_id}"
    start_seconds = int(start_time)
    end_seconds = int(end_time)
    return f"{base_url}?start={start_seconds}&end={end_seconds}"

def generate_frame_urls(video_id, start_time, end_time):
    """
    Generate URLs for video frames at specific intervals.
    
    Args:
    video_id (str): YouTube video ID
    start_time (float): Start time of the segment in seconds
    end_time (float): End time of the segment in seconds
    
    Returns:
    dict: Dictionary containing URLs for first and last frames
    """
    logger.debug(f"Generating frame URLs for video {video_id} from {start_time} to {end_time}")
    base_url = f"https://img.youtube.com/vi/{video_id}/"
    
    # Calculate total duration and frame intervals
    duration = end_time - start_time
    interval = duration / 6  # Divide the segment into 6 parts to get 5 intervals
    
    first_frames = []
    last_frames = []
    
    # Generate URLs for first 5 frames
    for i in range(5):
        time = start_time + (i * interval)
        frame_number = math.floor(time / 60) * 60  # YouTube thumbnails are available every 60 seconds
        if frame_number == 0:
            url = f"{base_url}0.jpg"
        else:
            url = f"{base_url}1.jpg"  # Use the first thumbnail for non-zero frames
        first_frames.append(url)
    
    # Generate URLs for last 5 frames
    for i in range(5):
        time = end_time - ((4 - i) * interval)
        frame_number = math.floor(time / 60) * 60
        url = f"{base_url}2.jpg"  # Use the second thumbnail for end frames
        last_frames.append(url)
    
    return {
        'first_frames': first_frames,
        'last_frames': last_frames
    }

def generate_response_llama(prompt, query, model="llama-70b-chat"):
    """
    Generate a response using the Llama 3.1 API.
    
    Args:
    prompt (str): System prompt
    query (str): User query
    model (str): Model name to use
    
    Returns:
    str: Generated response
    """
    try:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=1024,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Llama 3.1 API: {str(e)}", exc_info=True)
        return "I apologize, but I encountered an error while processing your request."

def generate_response_gemini(prompt, query, frame_urls):
    """
    Generate a response using the Google Gemini API with visual analysis.
    
    Args:
    prompt (str): System prompt
    query (str): User query
    frame_urls (dict): Dictionary containing URLs for first and last frames
    
    Returns:
    str: Generated response
    """
    try:
        # Download and process images
        images = []
        for url in frame_urls['first_frames'] + frame_urls['last_frames']:
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))
            images.append(img)

        # Prepare the input for Gemini
        gemini_input = [
            prompt,
            f"Query: {query}",
            "Analyze the following frames from the video:",
            *images
        ]

        # Generate content using Gemini
        response = gemini_model.generate_content(gemini_input)
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}", exc_info=True)
        return "I apologize, but I encountered an error while processing your request."

def generate_embedding(text):
    """
    Generate an embedding for the given text using FastEmbed.
    
    Args:
    text (str): Input text
    
    Returns:
    list: Embedding vector
    """
    try:
        logger.debug("Generating embedding using FastEmbed")
        embedding = embedding_model.embed(text)[0].tolist()
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
        raise

def calculate_cosine_similarity(vector1, vector2):
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
    vector1 (list): First vector
    vector2 (list): Second vector
    
    Returns:
    float: Cosine similarity score
    """
    if not vector1 or not vector2:
        logger.warning("One or both vectors are empty in calculate_cosine_similarity")
        return 0
    return 1 - cosine(vector1, vector2)

@app.route('/process_query', methods=['POST'])
def process_query():
    logger.info("Received query processing request")
    try:
        data = request.json
        query = data['query']
        language = data.get('language', 'en-US')
        similarity_threshold = float(data.get('similarityThreshold', 0.8))  # Ensure it's parsed as float
        use_pinecone = data.get('use_pinecone', False)  # New parameter to optionally use Pinecone

        logger.debug(f"Query: {query}")
        logger.debug(f"Language: {language}")
        logger.debug(f"Similarity Threshold: {similarity_threshold}")
        logger.debug(f"Using Pinecone: {use_pinecone}")

        logger.debug("Generating embedding for query")
        query_embedding = generate_embedding(query)
        logger.debug(f"Query embedding generated, length: {len(query_embedding)}")

        logger.debug("Retrieving relevant video")
        
        if use_pinecone:
            # Use Pinecone for vector search
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True
            )
            
            filtered_results = [
                match for match in results['matches']
                if 'values' in match and match['values'] and
                calculate_cosine_similarity(query_embedding, match['values']) >= similarity_threshold
            ]
        else:
            # Use Qdrant for vector search
            search_result = qdrant_client.search(
                collection_name="video_chunks",
                query_vector=query_embedding,
                limit=10
            )
            
            filtered_results = [
                match for match in search_result
                if calculate_cosine_similarity(query_embedding, match.vector) >= similarity_threshold
            ]
        
        if filtered_results:
            if use_pinecone:
                video_id = filtered_results[0]['metadata']['video_id']
                chunk_details = [
                    {
                        'text': match['metadata']['text'],
                        'start_time': match['metadata']['start_time'],
                        'end_time': match['metadata']['end_time']
                    }
                    for match in filtered_results
                ]
            else:
                video_id = filtered_results[0].payload['video_id']
                chunk_details = [
                    {
                        'text': match.payload['text'],
                        'start_time': match.payload.get('start_time'),
                        'end_time': match.payload.get('end_time')
                    }
                    for match in filtered_results
                ]
        else:
            video_id = None
            chunk_details = []

        if not video_id:
            logger.info("No suitable video found for the query")
            answer = get_language_specific_no_match_response(language)
            disclaimer = get_language_specific_disclaimer(language)
            return jsonify({
                'response': answer + disclaimer,
                'video_id': None,
                'temporal_segments': [],
                'relevant_start': None,
                'relevant_end': None,
                'frame_urls': None
            })
        else:
            logger.debug(f"Retrieved video ID: {video_id}")


# Modify the process_query function
#@app.route('/process_query', methods=['POST'])
def process_query_old():
    logger.info("Received query processing request")
    try:
        data = request.json
        query = data['query']
        language = data.get('language', 'en-US')
        similarity_threshold = float(data.get('similarityThreshold', 0.8))  # Ensure it's parsed as float

        logger.debug(f"Query: {query}")
        logger.debug(f"Language: {language}")
        logger.debug(f"Similarity Threshold: {similarity_threshold}")

        logger.debug("Generating embedding for query")
        query_embedding = generate_embedding(query)
        logger.debug(f"Query embedding generated, length: {len(query_embedding)}")

        logger.debug("Retrieving relevant video")
        # Retrieve relevant video chunks directly from Pinecone
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        filtered_results = [
            match for match in results['matches']
            if 'values' in match and match['values'] and
            calculate_cosine_similarity(query_embedding, match['values']) >= similarity_threshold
        ]
        
        if filtered_results:
            video_id = filtered_results[0]['metadata']['video_id']
            chunk_details = [
                {
                    'text': match['metadata']['text'],
                    'start_time': match['metadata']['start_time'],
                    'end_time': match['metadata']['end_time']
                }
                for match in filtered_results
            ]
        else:
            video_id = None
            chunk_details = []
        if not video_id:
            logger.info("No suitable video found for the query")
            answer = get_language_specific_no_match_response(language)
            disclaimer = get_language_specific_disclaimer(language)
            return jsonify({
                'response': answer + disclaimer,
                'video_id': None,
                'temporal_segments': [],
                'relevant_start': None,
                'relevant_end': None,
                'frame_urls': None
            })
        else:
            logger.debug(f"Retrieved video ID: {video_id}")

            logger.debug("Searching for similar chunks in Pinecone")
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=10,  # Increased from 5 to 10
                include_metadata=True
            )

            logger.debug(f"Pinecone returned {len(results['matches'])} matches")

            filtered_results = [
                match for match in results['matches']
                if 'values' in match and match['values'] and
                calculate_cosine_similarity(query_embedding, match['values']) >= similarity_threshold
            ]

            logger.debug(f"Filtered results: {filtered_results}")

            if not filtered_results:
                logger.info("No matches found above the similarity threshold")
                answer = get_language_specific_no_match_response(language)
                disclaimer = get_language_specific_disclaimer(language)
                return jsonify({
                    'response': answer + disclaimer,
                    'video_id': None,
                    'temporal_segments': [],
                    'relevant_start': None,
                    'relevant_end': None,
                    'frame_urls': None
                })

            context = "\n".join([match['metadata']['text'] for match in filtered_results])

            chunk_details = []
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    for match in results['matches']:
                        embedding_id = match['id']
                        logger.debug(f"Retrieving chunk details for embedding_id: {embedding_id}")
                        cur.execute("""
                        SELECT video_id, start_time, end_time, language FROM video_chunks
                        WHERE embedding_id = %s
                        """, (embedding_id,))
                        chunk_detail = cur.fetchone()
                        if chunk_detail:
                            chunk_details.append({
                                'video_id': chunk_detail[0],
                                'start_time': chunk_detail[1],
                                'end_time': chunk_detail[2],
                                'language': chunk_detail[3],
                                'url': generate_youtube_url(chunk_detail[0], chunk_detail[1], chunk_detail[2])
                            })
                            logger.debug(f"Chunk detail added: {chunk_details[-1]}")

            relevant_start = min(chunk['start_time'] for chunk in chunk_details)
            relevant_end = max(chunk['end_time'] for chunk in chunk_details)

            frame_urls = generate_frame_urls(video_id, relevant_start, relevant_end)
            logger.debug(f"Frame URLs generated: {frame_urls}")

            logger.info("Generating response using OpenAI/Claude API")
            prompt = get_language_specific_prompt(language, query, context)

            #to test groq
            use_api = 'groq' #llama inference endpoint

            if use_api == 'openai':
                answer = generate_response_openai(prompt, query)
            elif use_api == 'groq':
                answer = generate_response_groq(prompt, query)
            elif use_api == 'gemini':
                answer = generate_response_gemini(prompt, query)
            else:
                raise ValueError("Invalid API choice")

            logger.info(f"{use_api.capitalize()} API response generated")
            logger.info("Query processing completed successfully")

            disclaimer = get_language_specific_disclaimer(language)

            logger.info("Composed response(with disclaimer) generated from OpenAI API")

            logger.info("Query processing completed successfully")
            return jsonify({
                'response': answer + disclaimer,
                'video_id': video_id,
                'temporal_segments': chunk_details,
                'relevant_start': relevant_start,
                'relevant_end': relevant_end,
                'frame_urls': frame_urls
            })
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
# Add this new debug endpoint
@app.route('/debug/index_stats', methods=['GET'])
def index_stats():
    stats = pinecone_index.describe_index_stats()
    return jsonify(stats)

def get_language_specific_prompt(language, query, context):
    if language.startswith('hi'):
        return f"""
        आप एक सहानुभूतिपूर्ण चिकित्सा ज्ञान एजेंट हैं। इस RAG (पुनर्प्राप्ति संवर्धित पीढ़ी) उपयोग के मामले के लिए नीचे दिए गए पाठ (एक वेक्टर डेटाबेस से पुनर्प्राप्त) के साथ चिकित्सा प्रश्न का एक सटीक, सहानुभूतिपूर्ण उत्तर हिंदी में उत्पन्न करें।

        प्रश्न: {query}
        संदर्भ: {context}

        उत्तर:
        """
    elif language.startswith('bn'):
        return f"""
        আপনি একজন সহানুভূতিশীল চিকিৎসা জ্ঞান এজেন্ট। এই RAG (রিট্রিভাল অগমেন্টেড জেনারেশন) ব্যবহারের ক্ষেত্রে নীচের টেক্সট (একটি ভেক্টর ডাটাবেস থেকে পুনরুদ্ধার করা) সহ চিকিৎসা প্রশ্নের একটি সঠিক, সহানুভূতিশীল প্রতিক্রিয়া বাংলায় তৈরি করুন।

        প্রশ্ন: {query}
        প্রসঙ্গ: {context}

        উত্তর:
        """
    else:  # Default to English
        return f"""
        You are an empathetic medical knowledge agent. Generate an accurate, empathetic response in English to the medical query, with the text below (retrieved from a vector db) for this RAG (retrieval augmented generation) usecase.

        Query: {query}
        Context: {context}

        Response:
        """

def get_language_specific_no_match_response(language):
    if language.startswith('hi'):
        return "मुझे खेद है, लेकिन मेरे कॉर्पस में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है। क्या आप कृपया अपना प्रश्न दोबारा पूछ सकते हैं या किसी अलग विषय के बारे में पूछ सकते हैं?"
    elif language.startswith('bn'):
        return "দুঃখিত, কিন্তু আমার কর্পাসে এই প্রশ্নের উত্তর দেওয়ার জন্য পর্যাপ্ত তথ্য নেই। আপনি কি অনুগ্রহ করে আপনার প্রশ্নটি পুনরায় জিজ্ঞাসা করতে পারেন বা একটি ভিন্ন বিষয় সম্পর্কে জিজ্ঞাসা করতে পারেন?"
    else:  # Default to English
        return "I'm sorry, but I don't have sufficient information in my corpus to answer this question. Could you please rephrase or ask about a different topic?"

def old_get_language_specific_no_match_response(language):
    if language.startswith('hi'):
        return "आपके स्वास्थ्य प्रश्न का कोई मिलान हमारे वीडियो संग्रह में नहीं मिला। कृपया एक मान्यता प्राप्त स्वास्थ्य सेवा प्रदाता से परामर्श लें या हमारे वीडियो संग्रह में अनुक्रमित और जोड़े जाने के लिए एक वीडियो जमा करें।"
    elif language.startswith('bn'):
        return "আপনার স্বাস্থ্য প্রশ্নের সাথে আমাদের ভিডিও সংগ্���হে কোনো মিল পাওয়া যায়নি। অনুগ্রহ করে একজন স্বীকৃত স্বাস্থ্যসেবা প্রদানকারীর সাথে পরামর্শ করুন অথবা আমাদের ভিডিও সংগ্রহে সূচীভুক্ত ও যোগ করার জন্য একটি ভিডিও জমা দিন।"
    else:
        return "No match to your health query was found in our video corpus. Please consider an accredited healthcare provider or submit a video to be indexed & added in our video corpus."

def get_language_specific_video_segments(language, chunk_details):
    if language.startswith('hi'):
        return "\n\nप्रासंगिक वीडियो खंड:\n" + "\n".join([f"- {detail['url']} (भाषा: {detail['language']})" for detail in chunk_details])
    elif language.startswith('bn'):
        return "\n\nপ্রাসঙ্গিক ভিডিও সেগমেন্ট:\n" + "\n".join([f"- {detail['url']} (ভাষা: {detail['language']})" for detail in chunk_details])
    else:
        return "\n\nRelevant video segments:\n" + "\n".join([f"- {detail['url']} (Language: {detail['language']})" for detail in chunk_details])

def get_language_specific_disclaimer(language):
    if language.startswith('hi'):
        return "\n\nयह ऐप और इस ऐप का उपयोग करके प्रदान की गई सलाह केवल शोध उद्देश्यों के लिए है और यह स्वास्थ्य सेवा सलाह नहीं है। कृपया वास्तविक स्वास्थ्य सेवा सलाह के लिए एक मान्यता प्राप्त स्वास्थ्य सेवा प्रदाता से परामर्श करें।"
    elif language.startswith('bn'):
        return "\n\nএই অ্যাপ এবং এই অ্যাপ ব্যবহার করে প্রদত্ত পরামর্শ শুধুমাত্র গবেষণার উদ্দেশ্যে এবং এটি স্বাস্থ্যসেবা পরামর্শ নয়। অনুগ্রহ করে প্রকৃত স্বাস্থ্যসেবা পরামর্শের জন্য একজন স্বীকৃত স্বাস্থ্যসেবা প্রদানকারীর সাথে পরামর্শ করুন।"
    else:
        return "\n\nThis app and advice provided using this app is purely for research purposes and is NOT healthcare advice. Please consult an accredited healthcare provider for actual healthcare advice."


if __name__ == '__main__':
    logger.info("Application started")

    try:
        # Create database tables
        create_tables()
        logger.info("Database tables created/verified")

        # Get processed video IDs from the database
        processed_video_ids = get_processed_video_ids()
        logger.info(f"Retrieved {len(processed_video_ids)} processed video IDs")

        # Process local audio files
        current_directory = os.path.dirname(os.path.abspath(__file__))
        logger.debug(f"Current directory: {current_directory}")

        mp3_files = [f for f in os.listdir(current_directory) if f.endswith('.mp3')]
        logger.info(f"Found {len(mp3_files)} MP3 files in the current local directory")

        for mp3_file in mp3_files:
            video_id = mp3_file.split('.')[0]
            if video_id not in processed_video_ids:
                logger.info(f"Processing new MP3 file: {mp3_file}")
                file_path = os.path.join(current_directory, mp3_file)

                if not check_video_processed(video_id):
                    result = process_local_audio(file_path)
                    logger.info(f"Audio processing result: {result}")
                else:
                    logger.info(f"Skipping {video_id}, as it's already processed")

        # Initialize thread for SSE
        #sse_thread = threading.Thread(target=lambda: app.run(threaded=True), daemon=True)
        #sse_thread.start()

        logger.info("Starting Flask application")
        # Run the Flask app with SSE support
        app.run(host='0.0.0.0', port=8081, debug=True)

    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}", exc_info=True)
        sys.exit(1)
