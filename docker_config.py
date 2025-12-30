# Use Python base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the required files into the Docker container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose ports (if necessary for communication)
EXPOSE 8080

# Command to run the application
CMD ["python", "main.py"]
