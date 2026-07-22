import argparse
import os
import chromadb
from gpt4all import Embed4All, GPT4All
from langchain.chains import RetrievalQA

args = argparse.Namespace(prompt=None, gpu=False, debug=False, interactive=False)
model = None
client = None
embedder = None
collection = None


def query_collection(query_text, debug=False):
    embedded_query = embedder.embed(query_text)
    results = collection.query(query_embeddings=[embedded_query], n_results=5)
    results_text = "\n\n".join(results['documents'][0])
    if debug:
        print("DEBUG: Chroma DB retrieval results:")
        print(results['documents'][0])
        print(results['distances'][0])
    return results_text


def process_query(input_prompt, context):
    system_prompt = (
        f"You are a helpful assistant. Use the given context to answer the question concisely. "
        f"Context: {context}"
    )
    with model.chat_session():
        formatted_prompt = f"{system_prompt}\n\nUser: {input_prompt}\n\nAssistant:"
        return model.generate(formatted_prompt, max_tokens=1024)


def load_resources(model_path=None, db_path=None, gpu=False):
    global model, client, embedder, collection

    model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    if db_path is None:
        db_path = os.environ.get("CHROMA_DB_PATH", "data/db/")

    device = "cpu"
    if gpu:
        try:
            gpu_list = GPT4All.list_gpus() or []
            if gpu_list:
                device = gpu_list[0]
        except Exception as exc:
            print("WARNING: GPU detection failed, falling back to CPU.")
            print(f"Details: {exc}")
            device = "cpu"

    model = GPT4All(model_name, device=device)
    client = chromadb.PersistentClient(path=db_path)
    embedder = Embed4All()
    collection = client.get_collection("LR_Disco_2_embed4all")


def main(argv=None):
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", help="User prompt about Land Rover Discovery II and its maintenance")
    parser.add_argument("-g", "--gpu", action='store_true', help="Try GPU if available")
    parser.add_argument("-d", "--debug", action='store_true', help='Enable debug output for the Chroma retrieval')
    parser.add_argument("-i", "--interactive", action='store_true', help='Enable interactive chat mode')
    args = parser.parse_args(argv)

    if args.prompt:
        input_prompt = args.prompt
    else:
        input_prompt = "Tell me briefly about land rover discovery 2 model"

    load_resources(gpu=args.gpu)

    if args.interactive:
        while True:
            input_prompt = input("Ask about Land Rover Discovery 2 (or exit to quit): ")
            if input_prompt.lower() == 'exit':
                print("Exiting the interactive prompt. Goodbye!")
                break
            context = query_collection(input_prompt, debug=args.debug)
            res = process_query(input_prompt, context)
            print(res)
    else:
        context = query_collection(input_prompt, debug=args.debug)
        res = process_query(input_prompt, context)
        print(res)


if __name__ == "__main__":
    main()
