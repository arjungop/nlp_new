"""
Prompt building module for the Telugu Multi-Turn Dialogue Research Pipeline.

This module provides the PromptBuilder class, which constructs highly specialized,
linguistically and culturally informed prompts for Telugu movie dialogue response
generation. It supports Raw prompting and structured Chain-of-Thought (COT) prompting,
with awareness of Telugu cinematic conventions, emotional register, speaker dynamics,
and multi-turn dialogue coherence requirements.

Inputs:
    - Multi-turn dialogue context as a formatted string
    - Optional metadata such as speaker labels or inferred genre

Outputs:
    - Fully formatted prompt strings ready for inference with T5-class or
      decoder-only language models
"""

import logging
import textwrap
from typing import Optional
from config import Config


class PromptBuilder:
    """
    Constructs linguistically specialized and culturally informed prompts
    for Telugu movie dialogue response generation.

    Telugu cinema dialogues carry distinct emotional intensity, dramatic register,
    honorific structures, and culturally specific interpersonal dynamics. Generic
    prompting strategies fail to capture these properties. This class encodes
    domain knowledge about Telugu cinematic dialogue directly into the prompt
    structure to elicit responses that are contextually appropriate, emotionally
    coherent, and linguistically authentic.

    Supported prompt strategies:
        - Raw: Direct instruction prompting with domain-specific framing.
        - COT: Structured Chain-of-Thought prompting with explicit reasoning
                stages covering emotional analysis, speaker intent, relational
                subtext, and cinematic register before response generation.

    Attributes:
        config: Pipeline configuration object.
        logger: Module-level logger for diagnostic tracing.
    """

    # Canonical reasoning stages embedded in COT prompts.
    # Each stage targets a specific analytical dimension of Telugu dialogue.
    _COT_STAGES: tuple[str, ...] = (
        "Step 1 - Emotional Register: Identify the dominant emotion driving this"
        " dialogue moment (e.g., grief, defiance, longing, comic relief, moral"
        " conflict). Note any shift in emotional tone across turns.",

        "Step 2 - Speaker Relationship and Power Dynamic: Determine the social"
        " relationship between speakers (e.g., father-son, rivals, lovers,"
        " mentor-student). Identify whether the dynamic is hierarchical or"
        " equal, and whether the power balance has shifted in this exchange.",

        "Step 3 - Cinematic Genre and Dramatic Register: Assess whether this"
        " dialogue belongs to a dramatic, comedic, romantic, action, or socially"
        " conscious cinematic context. The register of the response must match"
        " the genre conventions of Telugu cinema for that type of scene.",

        "Step 4 - Honorific and Linguistic Register: Telugu dialogue uses"
        " distinct honorific levels (formal, semi-formal, colloquial, intimate)"
        " that are grammatically encoded. Identify the appropriate register"
        " based on speaker relationships and genre.",

        "Step 5 - Unresolved Tension and Narrative Intent: Identify what the"
        " next response must accomplish narratively. Does it escalate conflict,"
        " provide resolution, reveal character, introduce irony, or advance a"
        " subplot? The response must serve the dialogue's narrative function.",

        "Step 6 - Final Response Generation: Using all observations above,"
        " generate the next dialogue line in authentic Telugu. The response must"
        " be a single, complete spoken line consistent with cinematic delivery."
        " Do not include any English words unless code-switching is clearly"
        " established in the context.",
    )

    _RAW_INSTRUCTION: str = (
        "You are an expert in Telugu cinema and a native Telugu speaker."
        " You are given a sequence of spoken dialogue lines from a Telugu movie."
        " Your task is to generate the single most appropriate and natural"
        " next spoken line in Telugu that continues this dialogue."
        "\n\n"
        "Requirements for your response:\n"
        "- Write in authentic Telugu script.\n"
        "- Match the emotional intensity and tone of the preceding dialogue.\n"
        "- Respect the honorific register appropriate to the speakers' relationship.\n"
        "- Produce exactly one spoken line. Do not produce explanations or alternatives.\n"
        "- Do not transliterate Telugu into English script.\n"
        "- Do not introduce new characters or plot elements not implied by the context."
    )

    _COT_INSTRUCTION: str = (
        "You are an expert in Telugu linguistics, cinema, and dramatic writing."
        " You are given a sequence of spoken dialogue lines from a Telugu movie."
        " Before generating the next line, you must reason through the dialogue"
        " systematically using the steps below. Your final output must be a single"
        " spoken line in authentic Telugu."
        "\n\n"
        "Requirements for your final response:\n"
        "- Write in authentic Telugu script.\n"
        "- Match the emotional intensity and dramatic register of the scene.\n"
        "- Respect the honorific and linguistic register appropriate to the relationship.\n"
        "- Produce exactly one spoken line after completing all reasoning steps.\n"
        "- Do not transliterate Telugu into English script.\n"
        "- Do not introduce new characters or plot elements not implied by the context."
    )

    def __init__(self, config: Config) -> None:
        """
        Initializes the PromptBuilder with the pipeline configuration.

        Args:
            config: A Config dataclass instance containing all pipeline parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def build_raw_prompt(
        self,
        context: str,
        speaker_label: Optional[str] = None,
    ) -> str:
        """
        Constructs a domain-specialized raw instruction prompt for Telugu
        dialogue response generation without explicit reasoning steps.

        The raw prompt is not naive — it encodes Telugu cinematic domain
        knowledge and linguistic requirements directly as constraints,
        yielding significantly better baseline responses than a generic
        instruction while remaining free of step-by-step reasoning scaffolding.

        Args:
            context: The formatted multi-turn dialogue context string.
                     Each turn should be on a new line. Speaker labels
                     (e.g., "రాజు: ...") are supported and improve output quality.
            speaker_label: Optional label identifying whose turn comes next
                           (e.g., "రాజు"). If provided, it is appended to the
                           prompt to prime the model for speaker-consistent output.

        Returns:
            str: A complete, model-ready prompt string for raw inference.

        Raises:
            TypeError: If context is not a string.
            ValueError: If context is empty or contains only whitespace.
        """
        self._validate_context(context, prompt_type="raw")

        next_turn_prefix = (
            f"{speaker_label.strip()}:" if speaker_label and speaker_label.strip()
            else "Next Line:"
        )

        prompt = (
            f"{self._RAW_INSTRUCTION}"
            f"\n\n"
            f"Dialogue Context:\n"
            f"{context.strip()}"
            f"\n\n"
            f"{next_turn_prefix}"
        )

        self.logger.debug(
            "Raw prompt constructed. Context length: %d characters. "
            "Speaker label: %s.",
            len(context),
            speaker_label or "not provided",
        )
        return prompt

    def build_cot_prompt(
        self,
        context: str,
        speaker_label: Optional[str] = None,
    ) -> str:
        """
        Constructs a structured Chain-of-Thought prompt for Telugu dialogue
        response generation with explicit multi-stage reasoning scaffolding.

        The COT prompt guides the model through six analytical dimensions before
        response generation: emotional register, speaker relationship and power
        dynamics, cinematic genre and dramatic register, Telugu honorific and
        linguistic register, unresolved narrative tension, and final generation.

        This structure is specifically designed for Telugu cinematic dialogue,
        where emotional subtext, honorific grammar, and genre conventions are
        deeply intertwined and cannot be recovered from surface form alone.

        Args:
            context: The formatted multi-turn dialogue context string.
                     Each turn should be on a new line. Speaker labels
                     (e.g., "రాజు: ...") are supported and improve output quality.
            speaker_label: Optional label identifying whose turn comes next
                           (e.g., "రాజు"). If provided, it is appended after
                           the reasoning steps to prime the final generation.

        Returns:
            str: A complete, model-ready prompt string for COT inference.

        Raises:
            TypeError: If context is not a string.
            ValueError: If context is empty or contains only whitespace.
        """
        self._validate_context(context, prompt_type="COT")

        formatted_stages = "\n\n".join(self._COT_STAGES)

        next_turn_prefix = (
            f"{speaker_label.strip()}:" if speaker_label and speaker_label.strip()
            else "Final Telugu Response:"
        )

        prompt = (
            f"{self._COT_INSTRUCTION}"
            f"\n\n"
            f"Dialogue Context:\n"
            f"{context.strip()}"
            f"\n\n"
            f"Reasoning:\n"
            f"{formatted_stages}"
            f"\n\n"
            f"{next_turn_prefix}"
        )

        self.logger.debug(
            "COT prompt constructed. Context length: %d characters. "
            "Reasoning stages: %d. Speaker label: %s.",
            len(context),
            len(self._COT_STAGES),
            speaker_label or "not provided",
        )
        return prompt

    def format_dialogue_context(
        self,
        turns: list[str],
        speakers: Optional[list[str]] = None,
    ) -> str:
        """
        Formats a list of raw dialogue turns into a structured multi-turn
        context string suitable for insertion into any prompt template.

        If speaker labels are provided, each turn is prefixed with the
        corresponding speaker label in Telugu-compatible colon-separated format.
        If no speaker labels are provided, turns are numbered sequentially.

        Args:
            turns: An ordered list of dialogue strings, each representing
                   one spoken turn in the conversation.
            speakers: An optional list of speaker label strings corresponding
                      positionally to each turn in the turns list.

        Returns:
            str: A formatted multi-line dialogue string ready for prompt insertion.

        Raises:
            TypeError: If turns is not a list or contains non-string elements.
            ValueError: If turns is empty.
            ValueError: If speakers is provided but its length does not match turns.
        """
        if not isinstance(turns, list):
            raise TypeError(
                f"Expected turns to be a list. Received: {type(turns).__name__}."
            )
        if not turns:
            raise ValueError("Turns list cannot be empty.")
        if not all(isinstance(t, str) for t in turns):
            raise TypeError("All elements in turns must be strings.")
        if speakers is not None:
            if not isinstance(speakers, list):
                raise TypeError(
                    f"Expected speakers to be a list. Received: {type(speakers).__name__}."
                )
            if len(speakers) != len(turns):
                raise ValueError(
                    f"Length mismatch: {len(turns)} turns provided but "
                    f"{len(speakers)} speaker labels given."
                )

        lines: list[str] = []
        for index, turn in enumerate(turns):
            if speakers:
                label = speakers[index].strip() if speakers[index].strip() else f"Speaker {index + 1}"
                lines.append(f"{label}: {turn.strip()}")
            else:
                lines.append(f"Turn {index + 1}: {turn.strip()}")

        formatted = "\n".join(lines)
        self.logger.debug(
            "Formatted %d dialogue turns into context string (%d characters).",
            len(turns),
            len(formatted),
        )
        return formatted

    def _validate_context(self, context: str, prompt_type: str) -> None:
        """
        Validates that the dialogue context is a non-empty string.

        Args:
            context: The context string to validate.
            prompt_type: A label identifying the calling prompt type,
                         used only for diagnostic logging.

        Raises:
            TypeError: If context is not a string.
            ValueError: If context is empty or contains only whitespace.
        """
        if not isinstance(context, str):
            self.logger.error(
                "%s prompt validation failed. Context must be a string."
                " Received type: %s.",
                prompt_type,
                type(context).__name__,
            )
            raise TypeError(
                f"Context must be a string. Received: {type(context).__name__}."
            )
        if not context.strip():
            self.logger.error(
                "%s prompt validation failed. Context is empty or whitespace-only.",
                prompt_type,
            )
            raise ValueError("Context cannot be empty or consist solely of whitespace.")