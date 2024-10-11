# Google_L400_Labs_Code_Solutions
Build and Deploy a Generative AI solution using a RAG framework
> **⚠️ Warning:** Make sure to replace all instances of `PROJECT_ID = "qwiklabs-gcp-00-e4970b8c386a"` with your own Google Cloud project ID. Failing to do so will result in errors or unintended behavior when accessing Google Cloud resources.


# Cymbal Ingest to Vector Database

This guide provides step-by-step instructions on how to ingest documents into a vector database, create embeddings, and set up a Flask API for interacting with the data. This process involves integrating various Google Cloud services, such as Firestore, Vertex AI, and Cloud Run.

## Required Libraries and Modules

```python
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import pickle
from IPython.display import display, Markdown
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
```

### Explanation
This code imports the necessary modules and libraries. It includes:
- **vertexai**: For utilizing Google Cloud's Vertex AI functionalities.
- **pickle**: For loading and saving serialized data.
- **IPython.display**: For displaying Markdown content in notebooks.
- **Firestore**: To create and manage a vector database using Firestore.
- **LangChain**: For integrating with the Vertex AI Embeddings model.

## Setting Up the Project and Embedding Model

```python
PROJECT_ID = "qwiklabs-gcp-00-e4970b8c386a"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")
```

### Explanation
- Initializes Vertex AI using your project ID and location.
- Sets up the embedding model `text-embedding-004` for generating embeddings.

## Loading and Cleaning the Document

```python
!gcloud storage cp gs://partner-genai-bucket/genai069/nyc_food_safety_manual.pdf .

loader = PyMuPDFLoader("./nyc_food_safety_manual.pdf")
data = loader.load()

def clean_page(page):
  return page.page_content.replace("-\n","")\
                          .replace("\n"," ")\
                          .replace("\x02","")\
                          .replace("\x03","")\
                          .replace("fo d P R O T E C T I O N  T R A I N I N G  M A N U A L","")\
                          .replace("N E W  Y O R K  C I T Y  D E P A R T M E N T  O F  H E A L T H  &  M E N T A L  H Y G I E N E","")
```

### Explanation
- Downloads the PDF document from a Google Cloud Storage bucket.
- Uses PyMuPDF to load the document and defines a `clean_page` function to remove unwanted characters and text from each page.

## Chunking and Creating Embeddings

```python
cleaned_pages = []
for pages in data:
  cleaned_pages.append(clean_page(pages))

text_splitter = SemanticChunker(embedding_model)
docs = text_splitter.create_documents(cleaned_pages[0:4])
chunked_content = [doc.page_content for doc in docs]
chunked_embeddings = embedding_model.embed_documents(chunked_content)
```

### Explanation
- Cleans the document pages and uses the `SemanticChunker` to split the text into meaningful chunks.
- Generates embeddings for each chunk using the Vertex AI Embedding model.

## Storing in Firestore

```python
db = firestore.Client(project=PROJECT_ID)
collection = db.collection('food-safety')

for i, (content, embedding) in enumerate(zip(chunked_content, chunked_embeddings)):
    doc_ref = collection.document(f"doc_{i}")
    doc_ref.set({
        "content": content,
        "embedding": Vector(embedding)
    })
```

### Explanation
- Creates a Firestore client and sets up a collection named `food-safety`.
- Iterates through the chunked content and embeddings, adding them as documents in Firestore.

## Vector Search Function

```python
def search_vector_database(query: str):
  query_embedding = embedding_model.embed_query(query)
  vector_query = collection.find_nearest(
    vector_field="embedding",
    query_vector=Vector(query_embedding),
    distance_measure=DistanceMeasure.EUCLIDEAN,
    limit=5,
  )
  docs = vector_query.stream()
  context = [result.to_dict()['content'] for result in docs]
  return context
```

### Explanation
- This function searches the vector database for the closest matching documents based on the query.
- Retrieves the top 5 documents using the Euclidean distance measure and returns the matching content.

## Flask API Setup (`main.py`)

```python
import os
import json
import logging
import google.cloud.logging
from flask import Flask, render_template, request
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from langchain_google_vertexai import VertexAIEmbeddings
```

### Explanation
- Imports the necessary libraries for setting up a Flask API.
- Sets up Google Cloud logging and Firestore connection.

### Search and Response Function

```python
def ask_gemini(question):
    prompt_template = "Using the context provided below, answer the following question:\nContext: {context}\nQuestion: {question}\nAnswer:"
    context = search_vector_database(question)
    formatted_prompt = prompt_template.format(context=context, question=question)
    generation_config = GenerationConfig(
        temperature=0.7,
        max_output_tokens=256,
        response_mime_type="application/json",
    )
    contents = [{"role": "user", "parts": [{"text": formatted_prompt}]}]
    response = gen_model.generate_content(
        contents=contents,
        generation_config=generation_config
    )
    response_text = response.text if response else "{}"
    try:
        response_json = json.loads(response_text)
        answer = response_json.get("answer", "No answer found.")
    except json.JSONDecodeError:
        answer = "Error: Unable to parse response."
    return answer
```

### Explanation
- The `ask_gemini` function utilizes the Gemini model to answer user queries using the context retrieved from the vector database.
- It formats a prompt with the context and question, sends it to the generative model, and returns the model’s response.

## Docker and Cloud Run Deployment

### Dockerfile

```bash
docker build -t cymbal-docker-image -f Dockerfile .
docker tag cymbal-docker-image us-central1-docker.pkg.dev/qwiklabs-gcp-01-a365df012da7/cymbal-artifact-repo/cymbal-docker-image
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/qwiklabs-gcp-01-a365df012da7/cymbal-artifact-repo/cymbal-docker-image
```

### Explanation
- Builds the Docker image for the application and tags it with the artifact repository URL.
- Configures Docker to use gcloud credentials and pushes the image to Google Cloud's Artifact Registry.

### Cloud Run Deployment

```bash
gcloud run deploy cymbal-freshbot     --image=us-central1-docker.pkg.dev/qwiklabs-gcp-01-a365df012da7/cymbal-artifact-repo/cymbal-docker-image     --platform=managed     --region=us-central1     --allow-unauthenticated
```

### Explanation
- Deploys the Docker image to Cloud Run, allowing unauthenticated access to the service.
