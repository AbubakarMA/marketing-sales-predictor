# Start from a lightweight Python image
FROM python:3.12-slim

# Define the working directory inside the image
WORKDIR /app

# Copy the dependency file
COPY requirements.txt .

# TODO 1: Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# TODO 2: Copy app.py into /app
COPY app.py .

# TODO 3: Copy the models directory into /app/models
COPY models/ ./models/

# TODO 4: Expose Streamlit's default port
EXPOSE 8501

# TODO 5: Start the Streamlit application
# Bind to 0.0.0.0 and port 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]