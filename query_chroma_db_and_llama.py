import chromadb
from gpt4all import Embed4All
from gpt4all import GPT4All
from langchain.chains import RetrievalQA
import argparse

def query_collection(query_text):
    embedded_query = embedder.embed(query_text)
    results = collection.query(query_embeddings=[embedded_query], n_results=5)
    #embedded_query = embed.text([query_text], inference_mode="local")['embeddings']
    #results = collection.query(query_embeddings=embedded_query, n_results=5)
    results_text = ""
    for res in results['documents'][0]:
        results_text += res
    if args.debug:
        print("DEBUG: Chroma DB retrieval results:")
        print(results['documents'][0])
        print(results['distances'][0])
    return results_text

def process_query(input_prompt, context):
    system_prompt = f"You are a helpful assistant. Use the given context to answer the question. Use around six sentences and keep the answer concise. Context: {context}"
    user_prompt = f'User: {input_prompt}'   
    with model.chat_session():
        formatted_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        return model.generate(formatted_prompt, max_tokens=1024)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt", help = "User prompt about Land Rover Discovery II and its maintenance")
parser.add_argument("-g", "--gpu", action='store_true', help = "Try GPU if available")
parser.add_argument("-d", "--debug", action='store_true', help='Enable the debug output')
parser.add_argument("-i", "--interactive", action='store_true', help='Enable interactive chat mode')
args = parser.parse_args()

if args.prompt:
    input_prompt = args.prompt
else:
    input_prompt = "Tell me briefly about land rover discovery 2 model"

m_device = "gpu"
if args.gpu:
    if len(GPT4All.list_gpus()[0]) > 0:
        m_device = GPT4All.list_gpus()[0]

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device=m_device)
client = chromadb.PersistentClient(path="data/db/")

embedder = Embed4All()
collection = client.get_collection("LR_Disco_2_embed4all")
#from nomic import embed
#collection = client.get_collection("LR_Disco_2_nomic")

if args.interactive:
    while True:
        input_prompt = input("Ask about Land Rover Discovery 2 (or exit to quit): ")
        
        if input_prompt.lower() == 'exit':
            print("Exiting the interactive prompt. Goodbye!")
            break
        else:
            context = query_collection(input_prompt)
            res = process_query(input_prompt, context)
            print(res)
else:
    context = query_collection(input_prompt)
    res = process_query(input_prompt, context)
    print(res)
