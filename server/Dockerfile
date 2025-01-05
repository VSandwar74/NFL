# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Create a virtual environment inside the container
RUN python -m venv /venv

# Set the environment variable to use the venv's Python and pip
ENV PATH="/venv/bin:$PATH"

# Install the required packages in the virtual environment
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application files into the container
COPY . .

# Expose the port that uvicorn will run on
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
