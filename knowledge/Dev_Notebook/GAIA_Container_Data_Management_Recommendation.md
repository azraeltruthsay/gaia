
---

# 🧾 GAIA Container Data Management Recommendation

## 🔎 Problem

Your container builds and boots GAIA successfully, but runtime errors show missing **knowledge files**, **persona configs**, and **models**:

```
/gaia-assistant/knowledge/system_reference/cheat_sheet.json
/gaia-assistant/knowledge/personas/
/gaia-assistant/models/gaia-prime.gguf
```

The current `Dockerfile` only copies code (`app/`, `gaia_rescue.py`, etc.), leaving these data dirs out of the image.

---

## ✅ Goals

* Ensure **knowledge**, **models**, and **shared state** are available inside GAIA at runtime.
* Avoid unnecessary rebuilds during **development** (mount from host).
* Ensure **deployment builds** are self-contained (bake data into image).

---

## 🛠️ Dockerfile (sealed builds)

### 🔧 Patch

```dockerfile
# Set working directory
WORKDIR /gaia-assistant

# Ensure GAIA root is always on Python import path
ENV PYTHONPATH=/gaia-assistant

# Copy core files
COPY ./app ./app
COPY ./requirements.txt .
COPY ./runserver.py .
COPY ./gaia_rescue.py .

# Copy GAIA knowledge, models, and shared data
COPY ./knowledge ./knowledge
COPY ./models ./models
COPY ./shared ./shared

# Declare data volumes (mountable at runtime if desired)
VOLUME ["/gaia-assistant/knowledge", "/gaia-assistant/models", "/gaia-assistant/shared"]
```

---

## 🐳 docker-compose.yml (dev workflow)

### 🔧 Patch

```yaml
services:
  gaia:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./app:/gaia-assistant/app
      - ./requirements.txt:/gaia-assistant/requirements.txt
      # GAIA knowledge, models, and shared sessions (mounted for dev)
      - ./knowledge:/gaia-assistant/knowledge
      - ./models:/gaia-assistant/models
      - ./shared:/gaia-assistant/shared
```

---

## 🚀 Outcome

* **Development mode** (docker-compose)

  * `knowledge/`, `models/`, and `shared/` are mounted from the host.
  * You can edit personas, drop new models, and tweak sessions live without rebuilding.

* **Deployment mode** (direct `docker build`)

  * The same dirs are baked into the image via `COPY`.
  * The container runs standalone with all necessary data included.

* **Volumes declared** → Docker knows these dirs are persistent data, making it easy to override or bind at runtime.

---

## 🧭 Next Steps

1. Apply the Dockerfile patch.
2. Update `docker-compose.yml`.
3. Rebuild:

   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
   ```
4. Verify GAIA loads personas, cheat sheets, and model files without `FileNotFoundError`.

---
