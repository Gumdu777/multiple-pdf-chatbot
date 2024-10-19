FROM python:3.12-slim

# Install system dependencies for Pillow and FAISS
RUN apt-get update && apt-get install -y \
    zlib1g-dev libjpeg-dev libpng-dev \
    swig libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your application code
COPY . .

# Set entry point for Streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
