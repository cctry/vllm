# Use the official Python image from the Docker Hub
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Create the Conda environment
RUN conda create -n vllm python=3.9 -y

# Install the required Conda packages
RUN /bin/bash -c "source activate vllm && conda install -c conda-forge -c rapidsai ucx-proc=*=gpu 'ucx>=1.14' ucx-py -y"

# Install additional required Python packages including aiohttp, uvicorn, and uvloop
RUN /bin/bash -c "source activate vllm && conda install -c conda-forge aiohttp uvicorn uvloop fastapi -y"

# Copy the decode_worker.py file into the container
COPY ../splitwise/decode_worker.py /app/decode_worker.py
COPY ../splitwise/test_stub.py /app/test_stub.py
COPY ../splitwise/utils.py /app/utils.py
COPY ../vllm /app/vllm

# Expose the port that the HTTP server will run on
EXPOSE 8001

# Set the entry point to run the HTTP server
ENTRYPOINT ["/bin/bash", "-c", "source activate vllm && python /app/decode_worker.py"]
