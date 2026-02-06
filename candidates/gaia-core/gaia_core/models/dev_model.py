import json
import sys

class DevModel:
    """
    A mock model that prints the prompt to the console and waits for user input.
    """

    def __init__(self, name="dev_model"):
        self.name = name

    def create_chat_completion(self, messages, **kwargs):
        """
        Prints the prompt to the console and waits for user input.
        """
        print("---PROMPT SENT TO DEV MODEL---", file=sys.stderr)
        print(json.dumps(messages, indent=2), file=sys.stderr)
        print("------------------------------------", file=sys.stderr)
        print("Enter the response for the model:", file=sys.stderr)
        response_text = sys.stdin.readline()
        return {
            "choices": [
                {
                    "message": {
                        "content": response_text.strip(),
                    }
                }
            ]
        }
