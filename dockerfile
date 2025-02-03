FROM python:3.9-slim

# Create and change to the app directory.
WORKDIR /app

# Copy the requirements file and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code.
COPY . .

# Expose the port that Spaces expects (7860).
EXPOSE 7860

# Run your Flask app.
CMD ["python", "app.py"]
