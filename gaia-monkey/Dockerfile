FROM node:20-slim AS promptfoo
RUN npm install -g promptfoo

FROM python:3.12-slim
COPY --from=promptfoo /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=promptfoo /usr/local/bin/node /usr/local/bin/node
COPY --from=promptfoo /usr/local/bin/promptfoo /usr/local/bin/promptfoo
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "uvicorn", "gaia_monkey.main:app", "--host", "0.0.0.0", "--port", "6420"]
