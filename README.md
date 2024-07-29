This is a short trial of the Retrieval-Augmented Generation (RAG) method with a quantized version of the LLAMA 3 8B Large Language Model. The retrieval is performed using the Chroma vector database, which stores parts of the document and their respective embeddings from the Land Rover Discovery II maintenance manual (not included in this repository). I.e. this solution is supposed to have some sort of the knowledge of the Land Rover Discovery II models and their maintenance. Queries can be about the Discovery II and its maintenance can be done with [this simple python script](query_chroma_db_and_llama.py)

# Moving parts:
  * [read_pdf_to_chroma_langchain.py](read_pdf_to_chroma_langchain.py) Processes the input PDF file located in the data/pdf folder, divides it to chunks of text and does embedding, finally stores it to the Chroma DB
  * [query_chroma_db_and_llama.py](query_chroma_db_and_llama.py) loads the LLAMA 3 model, formats the user prompts with the retrieved augmentation part from the Chroma DB and finally invokes the model to generate output

The default prompt is "Tell me briefly about land rover discovery 2 model"
```
C:\Users\tai\Documents\kikkailuja\LRDISCO2_RAG_LLAMA3>python query_chroma_db_and_llama.py
The Land Rover Discovery Series II is a SUV that was produced from 1999 to 2004. It's known for its ruggedness, reliability, and off-road capabilities. The vehicle features a 4.0L V8 engine or the Td5 diesel engine option, with a manual or automatic transmission. The Discovery has a reputation for being capable in challenging terrain and is popular among outdoor enthusiasts.
```


