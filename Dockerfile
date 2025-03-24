FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Create entrypoint script
RUN echo '#!/bin/bash\n\
streamlit run streamlit_ui.py --server.address=0.0.0.0 --server.port=${PORT:-8501}' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Use shell form of ENTRYPOINT to ensure environment variables are expanded
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
