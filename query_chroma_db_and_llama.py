import chromadb
from gpt4all import Embed4All
from gpt4all import GPT4All
from langchain.chains import RetrievalQA
from nomic import embed

embedder = Embed4All()

client = chromadb.PersistentClient(path="data/db/")
#collection = client.get_collection("LR_Disco_2_nomic")
collection = client.get_collection("LR_Disco_2_embed4all")

def query_collection(query_text):
    embedded_query = embedder.embed(query_text)
    results = collection.query(query_embeddings=[embedded_query], n_results=5)
    #embedded_query = embed.text([query_text], inference_mode="local")['embeddings']
    #results = collection.query(query_embeddings=embedded_query, n_results=5)
    
    results_text = ""
    for res in results['documents'][0]:
        results_text += res
    return results_text

input_prompt = "Tell me briefly about land rover discovery 2 model"
context = query_collection(input_prompt)

system_prompt = f"You are a helpful assistant. Use the given context to answer the question. Use around six sentences and keep the answer concise. Context: {context}"
user_prompt = f'User: {input_prompt}'

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device='kompute:NVIDIA GeForce RTX 3060')

with model.chat_session():
    formatted_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    print(model.generate(formatted_prompt, max_tokens=1024))