# SwasthyaSahayak: Finding Semantic Answers to Your Health Queries

SwasthyaSahayak is an innovative AI-powered health information system designed to provide accurate and accessible health information in India using WhatsApp. This repository contains the backend for the video RAG (Retrieval Augmented Generation) component of the system.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

SwasthyaSahayak addresses the critical challenge of unreliable health information in Low and Middle-Income Countries (LMICs), particularly India. By combining a curated, multi-modal dataset, advanced natural language processing, and seamless WhatsApp integration, this system provides accessible, accurate, and contextually relevant health information.

The video RAG backend is a crucial component that enables the system to extract relevant information from educational health videos and provide targeted responses to user queries.

## Features

- **Video Content Integration**: Identifies and retrieves relevant segments from YouTube videos, enhancing user engagement and providing access to a wider range of educational content.
- **Speech-to-Text Conversion**: Utilizes state-of-the-art models like Whisper for accurate transcription of voice queries.
- **Retrieval Augmented Generation (RAG)**: Combines information retrieval with language generation to ensure accurate and relevant responses.
- **Multilingual Support**: Provides responses in multiple Indian languages.
- **WhatsApp Integration**: Seamlessly integrates with WhatsApp for broad accessibility.

## Architecture

The SwasthyaSahayak video RAG backend consists of the following key components:

1. **User Interface**: A web-based interface for user interactions.
2. **Speech-to-Text Engine**: Uses OpenAI's Whisper model for voice query transcription.
3. **Query Processing**: Generates embeddings using OpenAI's text-embedding-ada-002 model.
4. **Vector Database**: Utilizes Pinecone for efficient storage and retrieval of embeddings.
5. **Retrieval Augmented Generation (RAG)**: Combines information retrieval with language generation.
6. **Language Model**: Employs powerful models like GPT-4 or Claude for response generation.
7. **Video Content Integration**: A specialized module for identifying and retrieving relevant video segments.
8. **Database**: PostgreSQL for storing metadata about video segments and transcriptions.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/swasthyasahayak-video-rag.git

# Change to the project directory
cd swasthyasahayak-video-rag

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

1. Start the server:

```bash
python app.py
```

2. Send a POST request to the `/query` endpoint with your health-related question:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "What are the post-operative care instructions for knee surgery?"}' http://localhost:5000/query
```

3. The system will process your query, retrieve relevant information from the video database, and return a response.

## Contributing

We welcome contributions to SwasthyaSahayak! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
