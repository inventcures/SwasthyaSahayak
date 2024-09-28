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
import groq
import cohere
# This function has been moved just above the process_query function definition
from pinecone import Pinecone 
import psycopg2
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import yt_dlp
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from gtts import gTTS
import tempfile
from io import BytesIO
import io
import queue
import threading

# Configuration
OPENAI_API_KEY = 'sk-proj-0IZxYOjPpGdzjXQJV2y5T3BlbkFJRJ4ZTL0xFv2IayPs58bo'

#GOOGLE_API_KEY = "your-google-api-key-here"
ANTHROPIC_API_KEY = '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'
GROQ_API_KEY = 'gsk_q7hXXowq0mnwN0VXZnABWGdyb3FYdA5CuPEZkX65XgMCgMUM1X6egsk_q7hXXowq0mnwN0VXZnABWGdyb3FYdA5CuPEZkX65XgMCgMUM1X6e'
COHERE_API_KEY = '4UYkUp7TsQzRRV9m8IPMXG5WnJAke7qvd7bYdvai'

PINECONE_API_KEY = '09e30548-0e73-4c07-a646-50e05ae66899'
PINECONE_ENVIRONMENT = 'us-east-1'
PINECONE_INDEX_NAME = 'vid-rag'

DB_NAME = 'medvidqa-india-db'
DB_USER = 'tp53'
DB_PASSWORD = 'tracheostomy'
DB_HOST = '34.131.130.102'
DB_PORT = '5432'


# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
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

# Initialize API clients
try:
    logger.info("Initializing API clients")
    
    # OpenAI
    try:
        openai.api_key = OPENAI_API_KEY
        logger.debug("OpenAI API key set")
    except Exception as e:
        logger.error(f"Error setting OpenAI API key: {str(e)}", exc_info=True)

    # Anthropic
    try:
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.debug("Anthropic client initialized")
    except Exception as e:
        logger.error(f"Error initializing Anthropic client: {str(e)}", exc_info=True)

    # Groq
    try:
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
        logger.debug("Groq client initialized")
    except Exception as e:
        logger.error(f"Error initializing Groq client: {str(e)}", exc_info=True)

    # Cohere
    try:
        co = cohere.Client(COHERE_API_KEY)
        logger.debug("Cohere client initialized")
    except Exception as e:
        logger.error(f"Error initializing Cohere client: {str(e)}", exc_info=True)

    # Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        pinecone_index.describe_index_stats()
        logger.info("Pinecone connection successful")
    except Exception as e:
        logger.error(f"Error initializing Pinecone client: {str(e)}", exc_info=True)

    logger.info("All API clients initialized successfully")
except Exception as e:
    logger.error(f"Error initializing API clients: {str(e)}", exc_info=True)
    sys.exit(1)

# Database connection parameters
db_params = {
    'dbname': DB_NAME,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'host': DB_HOST,
    'port': DB_PORT
}

def get_db_connection():
    logger.debug("Attempting to establish database connection")
    try:
        conn = psycopg2.connect(**db_params)
        logger.info("Database connection established successfully")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}", exc_info=True)
        raise

def create_tables():
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
    return send_from_directory('frontend', 'index.html')

@app.route('/style.css')
def serve_css():
    return send_from_directory('frontend', 'style.css')

@app.route('/script.js')
def serve_js():
    return send_from_directory('frontend', 'script.js')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribes streaming audio using Whisper API and pushes results to SSE queue."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    
    try:
        response = openai.Audio.transcribe("whisper-1", audio_file)
        transcript_chunk = response['text']
        sse_queue.put(json.dumps({'text': transcript_chunk}))
        return jsonify({'success': True}), 200
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        return jsonify({'error': 'Transcription error'}), 500

@app.route('/transcribe_stream')
def transcribe_stream():
    """SSE endpoint for streaming transcription results."""
    def generate():
        while True:
            try:
                message = sse_queue.get(timeout=30)  # 30 second timeout
                yield f"data: {message}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'text': ''})}\n\n"  # Send empty message to keep connection alive

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def get_processed_video_ids():
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
    logger.info(f"Processing audio files in folder: {folder_path}")
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.mp3'):
                video_id = file_name.split('.')[0]
                logger.info(f"Processing file: {file_name}")
                
                if check_video_processed(video_id):
                    logger.info(f"Skipping {video_id}, as transcript, chunks & chunk embeddings exist in both postgres & pinecone")
                    logger.debug(f"Skipping {video_id}, as transcript, chunks & chunk embeddings exist in both postgres & pinecone")
                    continue
                
                file_path = os.path.join(folder_path, file_name)
                process_local_audio(file_path)
        
        logger.info("All audio files in the folder have been processed")
        return "All audio files processed successfully"
    except Exception as e:
        logger.error(f"Error processing audio folder: {str(e)}", exc_info=True)
        raise

def check_video_processed(video_id):
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
    logger.debug(f"Chunking transcript with duration {chunk_duration} seconds")
    words = transcript.split()
    logger.debug(f"Transcript contains {len(words)} words")
    chunks = []
    current_chunk = []
    current_duration = 0
    
    for i, word in enumerate(words):
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
    logger.info(f"Storing chunks and embeddings for video {video_id}")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for i, chunk in enumerate(chunks):
                    logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                    
                    logger.debug("Generating embedding")
                    embedding = openai.Embedding.create(
                        input=chunk["text"],
                        model="text-embedding-ada-002"
                    )['data'][0]['embedding']
                    logger.debug("Embedding generated")
                    
                    embedding_id = f"{video_id}_{chunk['start_time']}"
                    logger.debug(f"Upserting embedding {embedding_id} to Pinecone")
                    try:
                        pinecone_index.upsert([(embedding_id, embedding, {"text": chunk["text"], "video_id": video_id})])
                        logger.debug(f"Embedding upserted to Pinecone with metadata: {{'text': (chunk text), 'video_id': {video_id}}}")
                    except Exception as e:
                        logger.error(f"Error during Pinecone upsert: {str(e)}")
                    
                    logger.debug("Inserting chunk into PostgreSQL")
                    cur.execute("""
                    INSERT INTO video_chunks (video_id, start_time, end_time, text, embedding_id, language)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """, (video_id, chunk["start_time"], chunk["end_time"], chunk["text"], embedding_id, chunk["language"]))
                    logger.debug("Chunk inserted into PostgreSQL")
                
                conn.commit()
        logger.info(f"Stored {len(chunks)} chunks for video {video_id}")
    except Exception as e:
        logger.error(f"Error storing chunks and embeddings: {str(e)}", exc_info=True)
        raise

def generate_youtube_url(video_id, start_time, end_time):
    url = f"https://www.youtube.com/watch?v={video_id}&start={int(start_time)}&end={int(end_time)}"
    logger.debug(f"Generated YouTube URL: {url}")
    return url

@app.route('/process_folder', methods=['POST'])
def process_folder():
    logger.info("Received request to process folder")
    try:
        data = request.json
        folder_path = data.get('folder_path')
        if not folder_path:
            return jsonify({'error': 'Folder path not provided'}), 400
        
        result = process_audio_folder(folder_path)
        logger.info("Folder processing completed successfully")
        return jsonify({'message': result}), 200
    except Exception as e:
        logger.error(f"Error in process_folder: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    logger.info("Received text-to-speech request")
    try:
        data = request.json
        text = data['text']
        language = data.get('language', 'en')
        
        # Map frontend language codes to gTTS language codes
        language_map = {
            'en-US': 'en',
            'hi-IN': 'hi',
            'bn-IN': 'bn'
        }
        tts_language = language_map.get(language, 'en')
        
        logger.debug(f"Converting text to speech. Language: {tts_language}")
        tts = gTTS(text=text, lang=tts_language, slow=False)
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            tts.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        logger.info("Text-to-speech conversion completed")
        return send_file(temp_audio_path, mimetype='audio/mp3')
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Add new function for video retrieval
# Modify the retrieve_relevant_video function
def retrieve_relevant_video(query_embedding, similarity_threshold):
    logger.debug("Retrieving relevant video from Pinecone")
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=10,  # Increased from 1 to 10
        include_metadata=True
    )
    logger.debug(f"Pinecone query results: {results}")
    
    if results['matches']:
        for match in results['matches']:
            similarity_score = cosine_similarity(query_embedding, match['values'])
            logger.debug(f"Match ID: {match['id']}, Cosine similarity: {similarity_score}")
            
            if similarity_score >= similarity_threshold:
                if 'video_id' in match.get('metadata', {}):
                    return match['metadata']['video_id']
                else:
                    match_id = match.get('id', '')
                    video_id = match_id.split('_')[0] if '_' in match_id else None
                    if video_id:
                        logger.debug(f"Extracted video_id from match ID: {video_id}")
                        return video_id
        
       
    logger.warning("No matches found above the similarity threshold")
    return None

def generate_frame_urls(video_id, start_time, end_time):
    """
    Generate URLs for video frames using YouTube's thumbnail API.
    
    :param video_id: YouTube video ID
    :param start_time: Start time of the relevant segment in seconds
    :param end_time: End time of the relevant segment in seconds
    :return: Dictionary containing URLs for first 5 and last 5 frames
    """
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


def generate_response_groq(prompt, query, model="llama2-70b-4096"):
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
        logger.error(f"Error calling Groq API: {str(e)}", exc_info=True)
        return "I apologize, but I encountered an error while processing your request."

def generate_response_openai(prompt, query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=1500
        )
        return response.choices[0].message['content']
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}", exc_info=True)
        return "I apologize, but I encountered an error while processing your request."

def generate_response_gemini(prompt, query):
    try:
        model = GenerativeModel('gemini-pro')
        response = model.generate_content(f"{prompt}\n\nQuery: {query}")
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}", exc_info=True)
        return "I apologize, but I encountered an error while processing your request."


""" def generate_embedding(text, use_openai=False):
    try:
        if use_openai:
            logger.debug("Generating embedding using OpenAI API")
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        else:
            logger.debug("Generating embedding using Cohere API")
            response = co.embed(
                texts=[text],
                model='embed-english-v3.0',
                input_type='search_query'
            )
            # Cohere's 'embed-english-v3.0' model generates embeddings of exactly 1536 dimensions
            return response.embeddings[0]
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
        raise """


 # Fallback implementation:
def generate_embedding(text, use_openai=False):
    try:
        if use_openai:
            logger.debug("Generating embedding using OpenAI API")
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        else:
            logger.debug("Generating embedding using Cohere API")
            response = co.embed(
                texts=[text],
                model='embed-english-v3.0',
                input_type='search_query'
            )
            # Ensure the embedding is 1536 dimensions
            embedding = response.embeddings[0][:1536]
            return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
        raise


def calculate_cosine_similarity(query_embedding, chunk_embedding):
    return 1 - cosine(query_embedding, chunk_embedding)

# Modify the process_query function
@app.route('/process_query', methods=['POST'])
def process_query():
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
        video_id = retrieve_relevant_video(query_embedding, similarity_threshold)
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
                if cosine_similarity(query_embedding, match['values']) >= similarity_threshold
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
            use_api = 'groq'

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