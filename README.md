This is a short trial of the Retrieval-Augmented Generation (RAG) method with a quantized version of the LLAMA 3 8B Large Language Model. The retrieval is performed using the Chroma vector database, which stores parts of the document and their respective embeddings from the Land Rover Discovery II maintenance manual (not included in this repository). I.e. this solution is supposed to have some sort of the knowledge of the Land Rover Discovery II models and their maintenance. Queries can be about the Discovery II and its maintenance can be done with [this simple python script](query_chroma_db_and_llama.py)

Based on the following readily available components:
* [GPT4All](https://www.nomic.ai/gpt4all)
* [Langchain wrappered Unstructured PDF processing](https://unstructured.io/)
* [ChromaDB](https://www.trychroma.com/)
* Embeddings from Nomic and/or GPT4ALL

# Moving parts:
  * [read_pdf_to_chroma_langchain.py](read_pdf_to_chroma_langchain.py) Preprocesses the input PDF file located in the data/pdf folder, divides it to chunks of text and does embedding, finally stores it to the Chroma DB
  * [query_chroma_db_and_llama.py](query_chroma_db_and_llama.py) loads the LLAMA 3 model, formats the user prompts with the retrieved augmentation part from the Chroma DB and finally invokes the model to generate output

The default prompt is "Tell me briefly about land rover discovery 2 model"
```
C:\Users\tai\Documents\kikkailuja\LRDISCO2_RAG_LLAMA3>python query_chroma_db_and_llama.py
The Land Rover Discovery Series II is a SUV that was produced from 1999 to 2004. It's known for its ruggedness, reliability, and off-road capabilities. The vehicle features a 4.0L V8 engine or the Td5 diesel engine option, with a manual or automatic transmission. The Discovery has a reputation for being capable in challenging terrain and is popular among outdoor enthusiasts.
```
Some technical details with custom prompt
```
C:\Users\tai\Documents\kikkailuja\LRDISCO2_RAG_LLAMA3>python query_chroma_db_and_llama.py -p "Tell me what is the firing order of the td5 diesel engine"
According to the provided context, the firing order for the 2.5 litre in-line direct injection diesel, turbocharged and intercooled Td5 engine is:

1 - 2 - 4 - 5 - 3
```
Or just some opinnions :D
```
C:\Users\tai\Documents\kikkailuja\LRDISCO2_RAG_LLAMA3>python query_chroma_db_and_llama.py -p "Why is land rover discovery 2 better than its modern counter parts?"
Based on the workshop manual, I can provide some insights on why the Land Rover Discovery Series II (1999MY onwards) might be considered better than its modern counterparts in certain aspects. Here are a few points:

* The Series II Discovery was known for its ruggedness and off-road capabilities, which were unmatched by many of its contemporaries.
* Its design allowed it to tackle challenging terrain with ease, thanks to features like high ground clearance, four-wheel drive, and a robust suspension system.
* In terms of reliability, the Land Rover brand has historically been associated with durability and low maintenance costs. The Discovery Series II was no exception, with many owners reporting low repair rates and long lifespan for their vehicles.

Of course, this is just based on general information from the workshop manual, and opinions may vary depending on individual experiences and preferences.
```
# Wishlist:
* Fancy web ui
* Improved PDF content capturing including tables (as HTML?) and maybe even images
