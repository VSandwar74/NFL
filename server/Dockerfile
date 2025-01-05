# Use an official Python image from the Python Docker Hub repository
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy only the requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install -r requirements.txt

RUN --mount=type=secret,id=_env,dst=/etc/secrets/.env \
    pip install -r requirements.txt
    
# Copy the rest of the application
COPY . .

# Build argument for secret
ARG SUPABASE_URL
ARG SERVICE_URL

# Set environment variable during runtime
ENV SUPABASE_URL=$SUPABASE_URL
ENV SERVICE_URL=$SERVICE_URL

# Expose the port that your application will use
EXPOSE 8000

# Command to run FastAPI with uvicorn in the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
