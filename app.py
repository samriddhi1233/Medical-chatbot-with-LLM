from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

# FREE GEMINI API IMPORT
from google import genai

app = Flask(__name__)

load_dotenv()

# Load keys from .env (keep your variable names)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # <-- stores Gemini API Key

# Set Pinecone env
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Gemini client using the key
gemini_client = genai.Client(api_key=OPENAI_API_KEY)

# Load embeddings
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

# Load Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# GEMINI CHAT FUNCTION 
def gemini_chat(prompt_text):
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",  # <-- valid model from your list
        contents=prompt_text
    )
    return response.text



# combine docs + prompt for RAG
def gemini_stuff_documents_chain(docs, user_input):
    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = system_prompt.format(context=context, input=user_input)
    return gemini_chat(final_prompt)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    # retrieve from Pinecone
    retrieved_docs = retriever.get_relevant_documents(msg)

    # generate RAG response
    answer = gemini_stuff_documents_chain(retrieved_docs, msg)

    print("Response:", answer)
    return str(answer)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
