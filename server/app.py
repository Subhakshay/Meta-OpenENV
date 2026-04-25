import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import app

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
