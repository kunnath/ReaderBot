# Use the official Python image as a base image
FROM python:3.9-slim
#FROM continuumio/miniconda3:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HUGGINGFACEHUB_API_TOKEN='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'

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .


# Install the required dependencies

# Update pip to the latest version
RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt -v  # Add verbose flag for detailed output


# Copy the current directory contents into the container at /app
COPY . /app


# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app6.py"]