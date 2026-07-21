FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/TommiA/LRDISCO2_RAG_LLAMA3.git .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make the script executable
RUN chmod +x query_chroma_db_and_llama.py

# Expose port if needed (e.g., for web UI)
# EXPOSE 8000

# Define the entrypoint to run the application and forward runtime CLI switches
ENTRYPOINT ["python", "query_chroma_db_and_llama.py"]