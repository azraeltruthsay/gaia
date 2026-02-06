import openai
import logging
import time
from gaia_core.config import Config, get_config


class GPTAPIModel:
    def __init__(self, model_alias: str = "oracle_openai", config: Config = None):
        self.config = config or get_config()
        self.model_alias = model_alias
        self.client = openai.OpenAI(api_key=self.config.get_api_key("openai"))
        self.logger = logging.getLogger(__name__)

    def create_chat_completion(self, messages, max_tokens, temperature, top_p, stream=False):
        self.logger.debug(f"Oracle Request: {messages}")
        start_time = time.time()

        # Resolve the configured model name for this alias; previously we hard‑coded
        # "oracle" which returned None and caused the OpenAI API to reject the call.
        model_name = (
            self.config.get_model_name(self.model_alias)
            or self.config.MODEL_CONFIGS.get(self.model_alias, {}).get("model")
            or self.model_alias
        )

        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream
        )

        duration = time.time() - start_time
        self.logger.debug(f"Oracle request duration: {duration:.2f} seconds")

        if stream:
            return self._stream_response(response)

        content = getattr(response.choices[0].message, "content", None)
        if not content:
            raise RuntimeError("GPT Oracle response missing content")

        self.logger.debug(f"Oracle Response: {content[:200]}...")
        self._log_token_usage(response)

        return {
            "choices": [{
                "message": {
                    "content": content
                }
            }]
        }

    def _stream_response(self, response_stream):
        content = ""
        for chunk in response_stream:
            delta = getattr(chunk.choices[0].delta, "content", "")
            if delta:
                print(delta, end="", flush=True)  # Optional: stream live output to console
                content += delta

        self.logger.debug(f"Oracle Streamed Response: {content[:200]}...")
        return {
            "choices": [{
                "message": {
                    "content": content
                }
            }]
        }

    def _log_token_usage(self, response):
        try:
            usage = response.usage
            self.logger.info(
                f"Oracle Token Usage - Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}, "
                f"Total: {usage.total_tokens}"
            )
        except Exception as e:
            self.logger.warning(f"Could not extract token usage from Oracle response: {e}")
