"""
Sarvam model generation module for the Telugu Multi-Turn Dialogue Research Pipeline.

Provides the SarvamModel class, which wraps the Sarvam causal language model
via HuggingFace transformers. Handles prompt encoding, autoregressive text
generation, and decoding of output tokens into Telugu text.

Supports CUDA, Apple Silicon MPS, and CPU compute backends.
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config


def _resolve_device() -> torch.device:
    """
    Selects the best available compute device.

    Returns:
        torch.device: CUDA if available, then MPS (Apple Silicon), then CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SarvamModel:
    """
    Handles text generation using the Sarvam causal language model.

    Attributes:
        config: Pipeline configuration object.
        logger: Module-level logger.
        device: PyTorch compute device (CUDA, MPS, or CPU).
        tokenizer: Sarvam tokenizer.
        model: Sarvam causal language model.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the SarvamModel, loading the tokenizer and model.

        Args:
            config: Configuration object containing sarvam_model_name and
                    generation hyperparameters.

        Raises:
            OSError: If the model cannot be found or downloaded.
            RuntimeError: If device allocation fails.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = _resolve_device()

        self.logger.info(
            "Initializing SarvamModel using model: %s on %s",
            self.config.sarvam_model_name, self.device
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.sarvam_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.sarvam_model_name,
                torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16,
            )
            self.model.to(self.device)
            self.model.eval()
        except OSError as e:
            self.logger.error("Failed to load the model or tokenizer: %s", str(e))
            raise OSError(
                f"Could not load Sarvam model {self.config.sarvam_model_name}. "
                f"Check internet connectivity or the model identifier."
            ) from e
        except RuntimeError as e:
            self.logger.error("Hardware error while moving model to device: %s", str(e))
            raise RuntimeError(
                "Failed to allocate the Sarvam model to the compute device."
            ) from e

    def generate_responses(self, prompts: list[str]) -> list[str]:
        """
        Generates text completions for a batch of provided prompts.

        Args:
            prompts: A list of formatted input strings with context and instructions.

        Returns:
            A list of generated response texts corresponding to the input prompts.

        Raises:
            TypeError: If the prompts argument is not a list.
            ValueError: If the prompts list is empty.
            RuntimeError: If generation fails.
        """
        if not isinstance(prompts, list):
            self.logger.error("Input prompts must be a list.")
            raise TypeError("Expected prompts to be a list of strings.")

        if not prompts:
            self.logger.error("Received an empty list of prompts for generation.")
            raise ValueError("The input prompts list cannot be empty.")

        self.logger.debug("Generating responses for batch of %d prompts.", len(prompts))

        try:
            # Set pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            with torch.no_grad():
                encoded_input = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                )

                input_ids = encoded_input["input_ids"].to(self.device)
                attention_mask = encoded_input["attention_mask"].to(self.device)
                input_length = input_ids.shape[1]

                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_length,
                    num_beams=self.config.num_beams,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                responses: list[str] = []
                for i in range(len(prompts)):
                    generated_tokens = output_ids[i][input_length:]
                    response_text = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    ).strip()
                    responses.append(response_text)

        except RuntimeError as e:
            self.logger.error("Runtime error during sequence generation: %s", str(e))
            raise RuntimeError(
                "Failed to generate batch response, potentially out of memory."
            ) from e

        self.logger.debug("Successfully generated batch response of length %d", len(responses))
        return responses