FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install HDDM and its dependencies
RUN pip install --no-cache-dir \
    hddm>=0.9.8 \
    arviz>=0.15.0 \
    pymc3>=3.11.5 \
    theano-pymc>=1.1.2 \
    jupyterlab

# Create a non-root user and switch to it
RUN useradd -m -r user && \
    chown -R user:user /app
USER user

# Set the default command to start a shell
CMD ["/bin/bash"]
