import os
import logging
from src.logger import logging

print("Test script running")
logging.info("Test script logging")

# Test file writing
try:
    with open('artifacts/test_file.txt', 'w') as f:
        f.write("Test content")
    print("Successfully wrote test file")
except Exception as e:
    print(f"Error writing file: {str(e)}")
