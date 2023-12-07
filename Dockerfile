FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY train_deploy.py .

# Copy the credentials.json
COPY tensile-nebula-406509-8fd0cc70c363.json .

# Copy metadata.json
COPY meta_data.json .

# Run the train_deploy script
CMD ["python", "train_deploy.py"]
