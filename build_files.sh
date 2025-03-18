#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run migrations (optional, uncomment if needed)
# echo "Applying migrations..."
# python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Any additional build steps can be added here
echo "Build process completed."

