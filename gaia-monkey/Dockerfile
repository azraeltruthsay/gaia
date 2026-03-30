FROM node:20-slim AS promptfoo
RUN npm install -g promptfoo

FROM python:3.12-slim
COPY --from=promptfoo /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=promptfoo /usr/local/bin/node /usr/local/bin/node
COPY --from=promptfoo /usr/local/bin/promptfoo /usr/local/bin/promptfoo

# Create non-root user for container execution (host UID mapping)
RUN groupadd -g 1000 gaia && useradd -u 1000 -g gaia -m -s /bin/bash gaia

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Set ownership for non-root execution
RUN chown -R gaia:gaia /app

# Run as non-root user
USER gaia

CMD ["python", "-m", "uvicorn", "gaia_monkey.main:app", "--host", "0.0.0.0", "--port", "6420"]
