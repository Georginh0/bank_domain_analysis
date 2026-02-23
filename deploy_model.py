# deploy_model.py
import subprocess
import os

def deploy_to_docker():
    # Create Dockerfile
    dockerfile = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY models/ models/
COPY app/ app/

EXPOSE 5000

CMD ["python", "app/app.py"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    
    # Build and run Docker
    subprocess.run(['docker', 'build', '-t', 'banking-risk-api', '.'])
    subprocess.run(['docker', 'run', '-p', '5000:5000', 'banking-risk-api'])

def deploy_to_heroku():
    # Create Procfile
    with open('Procfile', 'w') as f:
        f.write('web: python app/app.py')
    
    # Deploy
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', 'Deploy to Heroku'])
    subprocess.run(['git', 'push', 'heroku', 'main'])

if __name__ == "__main__":
    deploy_to_docker()
    print("✓ Model deployed to Docker")