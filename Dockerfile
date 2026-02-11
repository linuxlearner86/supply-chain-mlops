From python:3.10-slim 
WORKDIR /app
copy . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn","src.inference.predict:app","--host","0.0.0.0","--port","8000"]