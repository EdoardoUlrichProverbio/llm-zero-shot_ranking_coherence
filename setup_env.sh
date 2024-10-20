#!/bin/bash

# Check if the requirements.txt file exists
if [ ! -f "requirements.txt" ]; then
  echo "requirements.txt not found!"
  exit 1
fi

# Check if Python is installed
if ! command -v python &> /dev/null; then
  echo "Python is not installed. Please install Python first."
  exit 1
fi

# Create the virtual environment in the 'env' directory
echo "Creating virtual environment in the 'env' directory..."
python -m venv env

# Check if the virtual environment was created successfully
if [ ! -d "env" ]; then
  echo "Failed to create the virtual environment."
  exit 1
fi

# Activate the virtual environment (Windows Git Bash version)
echo "Activating virtual environment..."
if [ -f "env/Scripts/activate" ]; then
  source env/Scripts/activate
else
  echo "Activation script not found. Something went wrong."
  exit 1
fi

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Deactivate the virtual environment
echo "Deactivating virtual environment..."
deactivate

echo "All dependencies installed and environment setup. To activate later, run:"
echo "source env/Scripts/activate"

