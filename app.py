import os
import subprocess
import sys

def main():
    print("Starting Django server on port 7860...")
    
    # Run database migrations and collect static files
    subprocess.run(["python", "manage.py", "migrate"], check=True)
    subprocess.run(["python", "manage.py", "collectstatic", "--noinput"], check=True)
    
    # Start Gunicorn on port 7860 (the port Hugging Face Gradio spaces expose)
    process = subprocess.Popen(
        [
            "gunicorn", 
            "backend.wsgi:application", 
            "--bind", "0.0.0.0:7860", 
            "--workers", "1", 
            "--threads", "4",
            "--timeout", "120"
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    process.wait()

if __name__ == "__main__":
    main()
