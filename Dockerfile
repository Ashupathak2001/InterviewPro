# ✅ Use Python 3.10 slim image
FROM python:3.10-slim

# ✅ Avoid creating .pyc files and ensure real-time logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ✅ Set working directory inside the container
WORKDIR /app

# ✅ Install system dependencies needed for OpenCV, PyAudio, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    portaudio19-dev \  
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ✅ Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ✅ Copy entire project files into the container
COPY . .

# ✅ Expose Streamlit port
EXPOSE 8501

# ✅ Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ✅ Run Streamlit with optimized settings
CMD ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]