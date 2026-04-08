# 1. UPGRADED to 3.10 to support openenv-core 0.2.0+
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first
COPY requirements.txt .

# 4. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY . .

# 6. Install the project in "editable" mode 
RUN pip install -e .

# 7. Expose the port Hugging Face uses
EXPOSE 7860

# 8. Launch using the module runner
CMD ["python", "-m", "server.app"]