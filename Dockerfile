# Use an official Python runtime
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the training script to create model files
# RUN python train_model.py

# Expose port (important for Railway)
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
