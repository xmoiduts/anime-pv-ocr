import sys
import os

# Ensure the root and src directory are in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gemini_ocr import main

if __name__ == "__main__":
    main()

