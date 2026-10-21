# base image
FROM python:3.12-slim

# work dir
WORKDIR /app

# copy
COPY . .

# run
RUN pip install --no-cache-dir -r requirement.txt

# port
EXPOSE 8501

# command
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]