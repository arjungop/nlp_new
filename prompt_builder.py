"""
Prompt building module for the Telugu Multi-Turn Dialogue Research Pipeline.

This module provides the PromptBuilder class, which constructs highly specialized,
linguistically and culturally informed prompts for Telugu movie dialogue response
generation.

DESIGN PHILOSOPHY
─────────────────
Telugu cinematic dialogue is one of the most linguistically complex domains for
generation: honorific grammar is morphologically encoded (verb conjugations change
entirely by register), nine classical rasas govern emotional logic, SOV word order
is non-negotiable, and five distinct genre conventions each demand different
rhetorical shapes. Generic prompting collapses all of this into noise.

This module treats prompts as linguistic instruments, not instruction templates.
Every constraint is grounded in either Telugu grammar, Indian aesthetic theory
(Natyashastra), or Tollywood cinematic convention. Nothing is decorative.

SUPPORTED STRATEGIES
────────────────────
  Raw   : Dense, constraint-rich instruction prompt. Token-efficient (~350–400
           lines of context headroom). Encodes domain knowledge as hard rules
           rather than reasoning scaffolding. Best for fast inference on Sarvam.

  CoT   : Six-stage Chain-of-Thought scaffold. Each stage builds on the previous:
           rasa → relationship → genre → register → narrative function → generation.
           Designed for Gemma 3 (32K context). The reasoning trace itself improves
           the final line by forcing the model to commit to an interpretation before
           generating — preventing register drift and rasa inconsistency.

  Few-Shot : Raw instruction augmented with k gold examples drawn from the same
             genre and register profile as the target context. Dramatically reduces
             hallucinated code-switching and honorific errors.

  Adaptive : Selects Raw or CoT automatically based on model name and context
             token budget from the pipeline Config.

TOKEN BUDGET (approximate, based on SentencePiece tokenization of Telugu Unicode)
────────────────────────────────────────────────────────────────────────────────
  Model                 Max Tokens    Max Context Lines (~80 chars/line)
  ─────────────────     ──────────    ──────────────────────────────────
  Sarvam (Raw only)      ~7,500        ~350–400 lines
  Gemma 3 (CoT+Raw)     ~32,000       ~1,500–1,600 lines

INPUTS
──────
  - Multi-turn dialogue context as a formatted string or structured turn list
  - Optional metadata: speaker labels, target dialect, genre hint, next speaker

OUTPUTS
───────
  - Fully formatted prompt strings ready for inference with T5-class,
    seq2seq, or decoder-only language models
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from config import Config


# ──────────────────────────────────────────────────────────────────────────────
# Domain Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class PromptStrategy(Enum):
    """Supported prompt construction strategies."""
    RAW        = auto()   # Direct instruction with hard linguistic constraints
    COT        = auto()   # Six-stage Chain-of-Thought with rasa/register analysis
    FEW_SHOT   = auto()   # Raw instruction + k in-context gold examples
    ADAPTIVE   = auto()   # Auto-selects RAW or COT based on config + token budget


class TeluguDialect(Enum):
    """
    Primary regional dialect variants of Telugu.

    Dialect selection affects vocabulary, phonological markers, and in some
    cases morphological patterns. The model must not mix dialects within a
    single generated line unless code-switching is established by the context.
    """
    COASTAL_ANDHRA = "coastal_andhra"   # Literary standard; news/film baseline
    RAYALASEEMA    = "rayalaseema"       # Harder consonants; "ఎట్లా" vs "ఎలా"
    TELANGANA      = "telangana"         # Urdu/Hindi loanwords; "అవునా", "సరే బే"
    UNSPECIFIED    = "unspecified"       # Infer from context; do not force


class CinematicGenre(Enum):
    """
    Primary Telugu cinematic genre modes.

    Each genre encodes a distinct set of rhetorical conventions, emotional
    registers, and structural expectations that the generated line must satisfy.
    """
    MASS_COMMERCIAL = "mass_commercial"  # Power dialogues, hero glorification
    FAMILY_DRAMA    = "family_drama"     # Kinship terms, sacrifice, generational
    ROMANCE         = "romance"          # Poetic, metaphor-heavy, indirection
    COMEDY          = "comedy"           # Timing-sensitive, callbacks, absurdist
    SOCIAL_ART      = "social_art"       # Naturalistic, understated, ideological
    UNSPECIFIED     = "unspecified"      # Infer genre from context


# ──────────────────────────────────────────────────────────────────────────────
# Example Container for Few-Shot Prompting
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DialogueExample:
    """
    A single gold example for few-shot prompt construction.

    Examples must be sourced from authentic Telugu film transcripts and
    manually verified for honorific correctness, dialect consistency, and
    genre appropriateness. Synthetic examples degrade few-shot quality.

    Attributes:
        context:        Formatted multi-turn dialogue string (the input).
        response:       The gold next line in Telugu script (the target).
        genre:          The cinematic genre of this example.
        dialect:        The Telugu dialect used.
        rasa:           The dominant emotional register (e.g., "కరుణ", "వీరం").
        speaker_label:  Optional label of the speaker who delivers the response.
    """
    context:       str
    response:      str
    genre:         CinematicGenre       = CinematicGenre.UNSPECIFIED
    dialect:       TeluguDialect        = TeluguDialect.UNSPECIFIED
    rasa:          str                  = ""
    speaker_label: Optional[str]        = None


# ──────────────────────────────────────────────────────────────────────────────
# Core Prompt Templates
# ──────────────────────────────────────────────────────────────────────────────

# NOTE ON TOKEN ECONOMY
# ─────────────────────
# These strings are class-level constants — they are defined once and reused
# across all instances. Every sentence earns its place. No redundancy.
# The Raw prompt targets ~480 tokens. The CoT instruction preamble targets ~200
# tokens, with the six stages adding ~650 tokens. Total CoT overhead: ~850 tokens.

_RAW_INSTRUCTION = """\
You are a master of Telugu cinema dialogue — a native Telugu speaker with deep \
fluency in Tollywood screenwriting conventions, classical Indian aesthetic theory, \
and all registers of the Telugu language.

You are given a sequence of spoken dialogue lines from a Telugu film. Your sole \
task is to generate the single most authentic and cinematically appropriate next \
spoken line in Telugu.

══════════════════════════════════════════════
 LINGUISTIC RULES  (derive everything from the given context — not from defaults)
══════════════════════════════════════════════

HONORIFIC REGISTER — Telugu encodes respect morphologically in verb conjugations.
  • Formal / Respectful (మీరు-form): Used for elders, strangers, authority figures,
    or any social superior. Verb endings: -తున్నారు, -తారు, -డారు, -చ్చారు.
    Example: "మీరు ఏమి చేస్తున్నారు?" / "వెళ్తారా?"
  • Intimate / Informal (నువ్వు-form): Used for close friends, lovers, younger
    siblings, or social subordinates. Verb endings: -తున్నావు, -తావు, -డావు, -చ్చావు.
    Example: "నువ్వు ఏమి చేస్తున్నావు?" / "వెళ్తావా?"
  → RULE: Identify the honorific level FROM the preceding turns. Maintain it
    exactly. Change it only if the dialogue explicitly marks a power shift —
    and if so, let the grammar encode that shift deliberately.

WORD ORDER — Telugu is strictly SOV (Subject → Object → Verb). The verb is last.
  Correct:   "నేను నిన్ను ఎప్పటికీ వదిలిపోను."  (I you never leave-not)
  Incorrect: "నేను వదిలిపోను నిన్ను ఎప్పటికీ."

AGGLUTINATIVE MORPHOLOGY — Suffixes stack on roots. Do not isolate what belongs
  together. Post-positional markers (-కి, -తో, -లో, -వల్ల, -గురించి) are suffixes,
  not standalone words.

DIALECT — Match the dialect established in the context precisely:
  • Coastal Andhra (standard): Literary baseline; most Tollywood productions.
  • Rayalaseema: Harder consonants; "ఎట్లా" not "ఎలా"; "పోద్దాం" not "వెళ్దాం".
  • Telangana: Urdu loanwords (ఖతం, జల్దీ, మాఫ్); markers like "సరే బే", "అవునా".
  → Never mix dialects within a single line unless the character is
    established as a dialect-code-switcher in the context.

CODE-SWITCHING — Include English loanwords ONLY if the context already contains
  them. Rural, village, period, or elder-character dialogue must be pure Telugu.

══════════════════════════════════════════════
 CINEMATIC RULES  (derive genre from the emotional and lexical texture of the dialogue)
══════════════════════════════════════════════

MASS / COMMERCIAL — Impact lands in the final verb. Power dialogues are declarative,
  percussive, and rhythmically complete. The hero's register is authoritative;
  the villain's is threatening or contemptuous. One-liners end on the hardest word.

FAMILY DRAMA — Kinship terms (అమ్మా, నాన్నా, అక్కా, తమ్ముడూ) are emotional anchors,
  not just addresses. Incomplete sentences are valid — emotion finishes what
  grammar doesn't. Restraint breaking is the climax.

ROMANCE — Prefer indirection over declaration. Metaphor and natural imagery
  (వెన్నెల, గాలి, నది, పువ్వు) carry emotional weight. Poetic rhythm matters —
  the line should feel speakable AND musical.

COMEDY — The punchline is structural, not lexical. It lands on an unexpected verb
  or a word that recontextualizes the entire prior sentence. Timing is encoded
  in sentence length and rhythm. Do not explain the joke.

SOCIAL / ART — Naturalistic, spare, understated. Sub-text over text. No melodrama.
  Ideological weight lives in ordinary words. Silence (expressed as trailing
  syntax or an unanswered question) is a valid structural choice.

EMOTIONAL TRAJECTORY — The emotional arc across turns is a vector with direction
  and magnitude. If escalating, your line must be at measurably higher intensity.
  If de-escalating, softer. Breaking the trajectory requires deliberate narrative
  purpose — and if you break it, break it with maximum impact.

══════════════════════════════════════════════
 OUTPUT RULES — NON-NEGOTIABLE
══════════════════════════════════════════════

  ✓  Exactly ONE spoken line. No alternatives, no commentary, no explanation.
  ✓  Telugu Unicode script only. No transliteration. No Roman characters
     representing Telugu phonemes.
  ✓  No stage directions, no quotation marks, no speaker labels in output.
  ✓  No new characters, locations, or plot elements not present or strongly
     implied in the given context.
  ✓  The line must be cinematically deliverable — a real actor could speak it.\
"""


_COT_INSTRUCTION_PREAMBLE = """\
You are an expert in Telugu linguistics, classical Indian dramaturgy, and \
Tollywood cinematic writing. You are given a sequence of spoken dialogue lines \
from a Telugu film.

Before generating the next line, you MUST reason through six analytical stages \
in sequence. Each stage builds on the findings of the previous one. Your final \
output is a single spoken line in authentic Telugu script — nothing else.

══════════════════════════════════════════════
 FINAL OUTPUT REQUIREMENTS — READ BEFORE REASONING
══════════════════════════════════════════════

  ✓  One complete spoken line in Telugu Unicode script.
  ✓  No transliteration, no Roman characters for Telugu phonemes.
  ✓  No stage directions, quotation marks, or speaker labels.
  ✓  No new characters or plot elements not implied by the context.
  ✓  Cinematically deliverable — a real actor could speak it.\
"""


# Six CoT stages — each stage explicitly references the discoveries of prior
# stages, creating a cumulative analytical chain rather than independent checks.
_COT_STAGES: tuple[str, ...] = (

    # ── Stage 1: Rasa ─────────────────────────────────────────────────────────
    """\
STEP 1 — DOMINANT RASA AND EMOTIONAL TRAJECTORY

From the nine rasas of the Natyashastra, identify which governs this dialogue:

  శృంగారం  (Shringara)  — romantic love, longing, beauty, erotic tenderness
  హాస్యం   (Hasya)      — comedy, wit, absurdity, playful mockery
  కరుణ    (Karuna)     — pathos, grief, compassion, mourning
  రౌద్రం  (Raudra)     — fury, outrage, righteous anger, violent passion
  వీరం    (Veera)      — heroism, courage, moral defiance, glory
  భయానకం  (Bhayanaka)  — fear, dread, existential threat, tension
  బీభత్సం (Bibhatsa)   — disgust, revulsion, moral horror, aversion
  అద్భుతం (Adbhuta)    — wonder, revelation, awe, the uncanny
  శాంతం   (Shanta)     — peace, resolution, acceptance, spiritual stillness

State the PRIMARY rasa in this dialogue. If a secondary rasa is in active
tension with the primary (creating dramatic irony, tonal complexity, or
subtext), name it and describe the tension.

Then state the EMOTIONAL TRAJECTORY across turns: is the intensity
  (a) escalating — building toward confrontation, breakdown, or climax?
  (b) de-escalating — moving toward resolution, forgiveness, or exhaustion?
  (c) pivoting — a tonal break, revelation, or ironic reversal?

The next line must continue this trajectory unless you identify a specific
narrative justification for breaking it — in which case, the break itself
becomes the most important structural event in the line.\
""",

    # ── Stage 2: Relationship + Power ─────────────────────────────────────────
    """\
STEP 2 — SPEAKER RELATIONSHIP AND POWER AXIS

Using the rasa and trajectory from Step 1, map the speaker relationship on
two intersecting axes:

  AXIS 1 — HIERARCHICAL POSITION:
    Superior → Subordinate  (parent/child, guru/shishya, employer/worker,
                              elder/younger, zamindar/tenant, officer/constable)
    Equal ↔ Equal            (peers, lovers of matched status, rivals)
    Subordinate → Superior  (child to parent, student to guru, younger to elder)

  AXIS 2 — EMOTIONAL ALIGNMENT:
    Allied      (trust, love, shared goal, solidarity)
    Opposed     (conflict, betrayal, rivalry, ideological clash)
    Ambivalent  (unresolved, oscillating — the most cinematically productive axis)

From these axes determine:
  1. What honorific form is GRAMMATICALLY MANDATORY for the next speaker?
     (మీరు-form for upward address or formal equal; నువ్వు-form for downward
      address or intimate equal)
  2. Has the power balance SHIFTED during this exchange? (e.g., a subordinate
     asserting dominance, a superior breaking down into vulnerability, a rival
     acknowledging respect for the first time) If so, name the shift — the
     language register must encode it.\
""",

    # ── Stage 3: Genre + Scene Type ───────────────────────────────────────────
    """\
STEP 3 — CINEMATIC GENRE AND SCENE TYPE

Using the rasa (Step 1) and relationship (Step 2), identify the cinematic genre
from the lexical texture, rhetorical cadence, and emotional logic of the dialogue:

  MASS / COMMERCIAL  — Punchy delivery. Power dialogues. Hero glorification.
    Villain menace. Declarative sentences that end on impact verbs. One-liners
    that compress an entire moral stance into a single breath.

  FAMILY DRAMA  — Kinship address terms as emotional load-bearers. Generational
    conflict or reconciliation. Sacrifice themes. Sentences that break off because
    emotion overcomes syntax. The unsaid weighs as much as the said.

  ROMANCE  — Metaphor density. Natural imagery. Deliberate indirection — the
    speaker circles the feeling without naming it. Rhythm approaches song.
    Confession through negation ("నేను నిన్ను మర్చిపోలేను").

  COMEDY  — The setup is in the prior turns; the punchline is yours to deliver.
    Structural: the humor lives in the unexpected final word or verb. The sentence
    builds toward something plausible and then swerves. Do not explain the joke.

  SOCIAL / ART  — Everyday language carrying enormous ideological weight.
    Understatement. Incomplete thoughts. Questions that expose systemic injustice
    in ordinary syntax. No melodrama — the weight comes from restraint.

Now name the SPECIFIC SCENE TYPE within that genre (e.g., "villain-hero climactic
confrontation," "mother-son reconciliation after betrayal," "lover's first
indirect confession," "comedic case of mistaken identity payoff," "whistleblower
confronting a corrupt official"). The scene type determines the exact rhetorical
SHAPE of the next line — its length, rhythm, and what it must do emotionally.\
""",

    # ── Stage 4: Linguistic Register ──────────────────────────────────────────
    """\
STEP 4 — HONORIFIC SYSTEM, DIALECT, AND LINGUISTIC REGISTER

From the relationship (Step 2) and scene type (Step 3), specify every linguistic
parameter of the next line:

  HONORIFIC FORM — Commit to one:
    మీరు-form  (Formal): Verb endings -తున్నారు / -తారు / -డారు / -చ్చారు / -ండి
      Required: elder address, authority, strangers, formal equals, upward mobility.
    నువ్వు-form (Intimate): Verb endings -తున్నావు / -తావు / -డావు / -చ్చావు / -వు
      Required: close friends, lovers, younger people, social subordinates.

    ⚠ If Step 2 identified a power SHIFT, the honorific form itself may be the
    mechanism of that shift — a subordinate suddenly using నువ్వు-form to a superior
    is a seismic dramatic event. Use this only if the context warrants it.

  DIALECT MARKERS — Identify the dialect present in the context and carry it:
    Coastal Andhra  : Literary standard; "ఎలా", "వెళ్దాం", "అవును".
    Rayalaseema     : "ఎట్లా", "పోద్దాం", "ఆఁ", harder retroflex consonants.
    Telangana       : Urdu loanwords (ఖతం, జల్దీ, మాఫ్ చేయ్), "అవునా", "సరే బే",
                      "ఏం జేస్తున్నావ్" not "ఏం చేస్తున్నావు".

  CODE-SWITCHING CALIBRATION:
    If the context contains English loanwords → calibrate density to match context.
    If context is pure Telugu → produce pure Telugu. No exceptions.

  CHARACTER VOICE CONSISTENCY — Identify any specific lexical or syntactic
    markers the next speaker has used in prior turns (preferred intensifiers,
    habitual filler particles like -గా, -యా, -రా, -లే, recurring metaphors,
    sentence-final particles). Reproduce them to maintain character voice.\
""",

    # ── Stage 5: Narrative Function ───────────────────────────────────────────
    """\
STEP 5 — UNRESOLVED TENSION AND SINGLE NARRATIVE FUNCTION

Every line in a Telugu film serves a precise narrative function. The next line
must fulfill EXACTLY ONE of the following — doing two at once dilutes both:

  ESCALATION      — Raises conflict intensity. Introduces information that
                    deepens the wound or commits to an irrevocable action.
                    The line must be measurably more intense than the last.

  DE-ESCALATION   — Breaks the tension. Offers forgiveness, creates a turning
    / RESOLUTION    point, or signals narrative closure. The gentling must feel
                    earned, not sudden.

  REVELATION      — Discloses a hidden truth. Exposes subtext as explicit text.
                    Reframes what the audience understood about prior lines.

  DEFIANCE        — The speaker refuses, rejects, or claims power. This is a
    / ASSERTION     declaration. The line marks a character becoming themselves.

  VULNERABILITY   — The speaker yields emotionally. Admits something, breaks
    / SURRENDER     down, asks for something, or lets a wall fall. The power
                    that comes from visible weakness.

  IRONIC PIVOT    — The surface tone contradicts the emotional subtext. Calm
                    in fury. Humor at the moment of greatest grief. The pivot
                    recontextualizes everything before it.

  COMEDIC PAYOFF  — Delivers the punchline. Completes the structural setup
                    established in prior turns. Subverts the expected verb
                    or word to land the laugh.

State the FUNCTION you have selected and articulate, in one sentence, the
SINGLE MOST IMPORTANT THING the next line must accomplish. Step 6 will execute
this and only this — not a combination, not a hedge.\
""",

    # ── Stage 6: Final Generation ──────────────────────────────────────────────
    """\
STEP 6 — FINAL TELUGU LINE GENERATION

Synthesize the complete analytical record:

  From Step 1: Carry the dominant rasa at the exact intensity level
    established by the emotional trajectory. If the trajectory required a
    pivot, execute the pivot in the first or last word — not buried in the middle.

  From Step 2: Reflect the power dynamic in the grammar. If the balance shifted,
    let the honorific form encode that shift — it is more powerful than any
    explicit statement.

  From Step 3: Deliver in the rhetorical shape of the scene type. A power
    dialogue is percussive and short. A family drama line may be syntactically
    broken. A romantic line curves and avoids. A comedy line swerves at the end.

  From Step 4: Use the exact honorific form, dialect markers, and character
    voice indicators identified. This is not aesthetic — it is linguistic
    fidelity. Errors here break immersion and character consistency.

  From Step 5: Execute the single narrative function. One line cannot do
    everything. Restraint is mastery.

Now write the line.

The line must be:
  — One complete spoken sentence in Telugu Unicode script.
  — Consistent with this character's established voice and speech rhythm.
  — Free of transliteration, stage directions, and meta-commentary.
  — Cinematically deliverable. A real actor could speak it on screen.\
""",

)


_FEW_SHOT_EXAMPLE_HEADER = """\
══════════════════════════════════════════════
 EXAMPLES  (gold references — study register, rhythm, and honorific form)
══════════════════════════════════════════════\
"""

_FEW_SHOT_EXAMPLE_TEMPLATE = """\
── Example {index} [{genre}  |  {dialect}  |  రస: {rasa}] ──
Context:
{context}

{speaker_prefix}{response}
"""

_FEW_SHOT_TASK_HEADER = """\
══════════════════════════════════════════════
 YOUR TASK
══════════════════════════════════════════════\
"""


# ──────────────────────────────────────────────────────────────────────────────
# PromptBuilder
# ──────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Constructs linguistically specialized and culturally informed prompts
    for Telugu movie dialogue response generation.

    Telugu cinema dialogues carry distinct emotional intensity encoded by
    classical rasa theory, grammatically mandatory honorific registers,
    SOV morphology, regional dialect variation, and genre-specific rhetorical
    conventions. Generic prompting strategies collapse all of this into noise.
    This class encodes that domain knowledge directly into prompt structure.

    SUPPORTED STRATEGIES
    ────────────────────
    Raw      : Dense instruction prompt with hard linguistic constraints.
               Token-efficient. Encodes rules without reasoning scaffolding.
               Suitable for Sarvam (~7,500 token budget).

    CoT      : Six-stage Chain-of-Thought. Rasa → Relationship → Genre →
               Register → Narrative Function → Final Line. Each stage feeds
               the next. Suitable for Gemma 3 (~32,000 token budget).

    Few-Shot : Raw instruction augmented with k verified gold examples scoped
               to the target genre and dialect. Reduces honorific errors and
               hallucinated code-switching more than any prompt engineering
               technique alone.

    Adaptive : Selects Raw or CoT automatically based on Config.model_name
               and token budget. Falls back to Raw when context is near budget.

    Attributes
    ──────────
    config : Pipeline configuration object (Config dataclass).
    logger : Module-level logger for diagnostic tracing.
    """

    # ── Models known to support CoT (large context, instruction-tuned) ──────
    _COT_CAPABLE_MODEL_SUBSTRINGS: tuple[str, ...] = (
        "gemma",
        "gemma-3",
        "gemini",
        "llama-3",
        "mistral",
        "claude",
    )

    # ── Approximate token overhead of each prompt component ─────────────────
    # These estimates are conservative (Telugu Unicode ~1.2–1.8 tokens/char).
    _TOKEN_ESTIMATE_RAW_OVERHEAD:  int = 520   # Instruction + separators
    _TOKEN_ESTIMATE_COT_OVERHEAD:  int = 920   # Preamble + 6 stages
    _TOKEN_ESTIMATE_PER_EXAMPLE:   int = 180   # Per few-shot gold example
    _TOKEN_ESTIMATE_PER_CHAR:      float = 0.6 # Conservative chars→tokens ratio

    def __init__(self, config: Config) -> None:
        """
        Initializes the PromptBuilder with the pipeline configuration.

        Args:
            config: A Config dataclass instance containing all pipeline parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Prompt Construction
    # ──────────────────────────────────────────────────────────────────────────

    def build_raw_prompt(
        self,
        context: str,
        speaker_label: Optional[str] = None,
        genre_hint:    Optional[CinematicGenre] = None,
        dialect_hint:  Optional[TeluguDialect]  = None,
    ) -> str:
        """
        Constructs a token-efficient, domain-specialized raw instruction prompt
        for Telugu dialogue response generation.

        The raw prompt is not naive — it encodes the Telugu honorific system
        with specific verb-ending examples, SOV word order, dialect markers,
        code-switching rules, and all five major cinematic genre conventions.
        It yields dramatically stronger baselines than generic instruction
        prompts while remaining lean enough for Sarvam's token budget.

        Args:
            context:      Formatted multi-turn dialogue string. One turn per line.
                          Speaker labels (e.g., "రాజు: ...") are supported and
                          substantially improve output quality.
            speaker_label: Optional label of the next speaker (e.g., "రాజు").
                           If provided, it is appended to prime speaker-consistent
                           output. If omitted, a generic "Next Line:" cue is used.
            genre_hint:   Optional CinematicGenre to prepend as a genre framing
                          hint when genre is known externally (e.g., from metadata).
                          If UNSPECIFIED or None, the prompt instructs the model
                          to derive genre from context.
            dialect_hint: Optional TeluguDialect to prepend as a dialect anchor.
                          If UNSPECIFIED or None, the prompt instructs derivation
                          from context.

        Returns:
            str: Complete, model-ready prompt string for raw inference.

        Raises:
            TypeError:  If context is not a string.
            ValueError: If context is empty or whitespace-only.
        """
        self._validate_context(context, prompt_type="Raw")

        sections: list[str] = [_RAW_INSTRUCTION]

        # ── Optional genre/dialect priming block ────────────────────────────
        hints = self._build_hint_block(genre_hint, dialect_hint)
        if hints:
            sections.append(hints)

        # ── Dialogue context ─────────────────────────────────────────────────
        sections.append(f"══ DIALOGUE CONTEXT ══\n{context.strip()}")

        # ── Next-turn cue ────────────────────────────────────────────────────
        next_cue = self._build_next_turn_cue(speaker_label, cot_mode=False)
        sections.append(next_cue)

        prompt = "\n\n".join(sections)

        self.logger.debug(
            "Raw prompt built | context_chars=%d | speaker=%s | "
            "genre=%s | dialect=%s | est_tokens~%d",
            len(context),
            speaker_label or "—",
            genre_hint.value   if genre_hint   else "infer",
            dialect_hint.value if dialect_hint else "infer",
            self.estimate_prompt_tokens(prompt),
        )
        return prompt

    def build_cot_prompt(
        self,
        context: str,
        speaker_label: Optional[str] = None,
        genre_hint:    Optional[CinematicGenre] = None,
        dialect_hint:  Optional[TeluguDialect]  = None,
    ) -> str:
        """
        Constructs a structured six-stage Chain-of-Thought prompt for Telugu
        dialogue response generation.

        The six stages are analytically interdependent:
          1. Rasa + emotional trajectory (the emotional vector)
          2. Speaker relationship + power axis (the social grammar)
          3. Cinematic genre + scene type (the rhetorical shape)
          4. Honorific form + dialect + character voice (the linguistic specification)
          5. Narrative function (the single thing this line must accomplish)
          6. Final line synthesis (execute all of the above)

        Each stage explicitly references the findings of prior stages, creating
        a cumulative analytical chain. This prevents the model from treating each
        stage as independent and then ignoring earlier outputs in generation.

        Designed for Gemma 3's 32K context window. Not recommended for Sarvam
        due to token overhead (~920 tokens before any context is added).

        Args:
            context:      Formatted multi-turn dialogue string. One turn per line.
                          Speaker labels substantially improve CoT quality by
                          giving the relationship analysis (Step 2) concrete anchors.
            speaker_label: Optional label of the next speaker. If provided, it is
                           appended after the final reasoning stage to prime
                           speaker-consistent generation.
            genre_hint:   Optional CinematicGenre for external genre anchoring.
            dialect_hint: Optional TeluguDialect for dialect anchoring.

        Returns:
            str: Complete, model-ready prompt string for CoT inference.

        Raises:
            TypeError:  If context is not a string.
            ValueError: If context is empty or whitespace-only.
        """
        self._validate_context(context, prompt_type="CoT")

        formatted_stages = "\n\n─────────────────────────────────────────────\n\n".join(
            _COT_STAGES
        )

        sections: list[str] = [_COT_INSTRUCTION_PREAMBLE]

        hints = self._build_hint_block(genre_hint, dialect_hint)
        if hints:
            sections.append(hints)

        sections.append(f"══ DIALOGUE CONTEXT ══\n{context.strip()}")
        sections.append(f"══ REASONING ══\n{formatted_stages}")

        next_cue = self._build_next_turn_cue(speaker_label, cot_mode=True)
        sections.append(next_cue)

        prompt = "\n\n".join(sections)

        self.logger.debug(
            "CoT prompt built | context_chars=%d | stages=%d | speaker=%s | "
            "genre=%s | dialect=%s | est_tokens~%d",
            len(context),
            len(_COT_STAGES),
            speaker_label or "—",
            genre_hint.value   if genre_hint   else "infer",
            dialect_hint.value if dialect_hint else "infer",
            self.estimate_prompt_tokens(prompt),
        )
        return prompt

    def build_few_shot_prompt(
        self,
        context:       str,
        examples:      list[DialogueExample],
        speaker_label: Optional[str] = None,
        genre_hint:    Optional[CinematicGenre] = None,
        dialect_hint:  Optional[TeluguDialect]  = None,
        max_examples:  int = 3,
    ) -> str:
        """
        Constructs a few-shot instruction prompt augmented with verified gold
        Telugu dialogue examples.

        Few-shot prompting is the single most effective intervention against the
        two most common failure modes in Telugu dialogue generation:
          (a) Honorific register drift — the model oscillates between మీరు and
              నువ్వు forms within or across turns, violating morphological consistency.
          (b) Hallucinated code-switching — the model inserts English loanwords
              into pure Telugu rural or period dialogue where none belong.

        Gold examples must be sourced from verified Telugu film transcripts and
        scoped to match the target genre and dialect as closely as possible.
        Mismatched examples (e.g., Telangana dialect examples for Rayalaseema
        contexts) actively harm generation quality.

        Args:
            context:      Formatted multi-turn dialogue string. The task input.
            examples:     List of DialogueExample instances. Each provides a
                          context/response pair with genre, dialect, and rasa
                          metadata. Truncated to max_examples.
            speaker_label: Optional label of the next speaker.
            genre_hint:   Optional CinematicGenre for task anchoring.
            dialect_hint: Optional TeluguDialect for task anchoring.
            max_examples: Maximum number of examples to include (default: 3).
                          More than 3 examples rarely improve quality and consume
                          significant token budget (~180 tokens each).

        Returns:
            str: Complete, model-ready few-shot prompt string.

        Raises:
            TypeError:  If context is not a string or examples not a list.
            ValueError: If context is empty, or examples list is empty.
        """
        self._validate_context(context, prompt_type="Few-Shot")
        if not isinstance(examples, list) or not examples:
            raise ValueError(
                "At least one DialogueExample must be provided for few-shot prompting."
            )

        selected = examples[:max_examples]

        # ── Build example block ───────────────────────────────────────────────
        example_blocks: list[str] = [_FEW_SHOT_EXAMPLE_HEADER]
        for idx, ex in enumerate(selected, start=1):
            speaker_prefix = (
                f"{ex.speaker_label.strip()}: "
                if ex.speaker_label and ex.speaker_label.strip()
                else ""
            )
            block = _FEW_SHOT_EXAMPLE_TEMPLATE.format(
                index         = idx,
                genre         = ex.genre.value   if ex.genre   else "—",
                dialect       = ex.dialect.value if ex.dialect else "—",
                rasa          = ex.rasa           if ex.rasa   else "—",
                context       = ex.context.strip(),
                speaker_prefix= speaker_prefix,
                response      = ex.response.strip(),
            )
            example_blocks.append(block.strip())

        examples_section = "\n\n".join(example_blocks)

        # ── Build task section ────────────────────────────────────────────────
        hints = self._build_hint_block(genre_hint, dialect_hint)

        task_parts: list[str] = [_FEW_SHOT_TASK_HEADER]
        if hints:
            task_parts.append(hints)
        task_parts.append(f"Context:\n{context.strip()}")
        task_parts.append(self._build_next_turn_cue(speaker_label, cot_mode=False))
        task_section = "\n\n".join(task_parts)

        # ── Assemble full prompt ──────────────────────────────────────────────
        prompt = "\n\n".join([
            _RAW_INSTRUCTION,
            examples_section,
            task_section,
        ])

        self.logger.debug(
            "Few-shot prompt built | examples=%d | context_chars=%d | "
            "speaker=%s | est_tokens~%d",
            len(selected),
            len(context),
            speaker_label or "—",
            self.estimate_prompt_tokens(prompt),
        )
        return prompt

    def build_adaptive_prompt(
        self,
        context:       str,
        speaker_label: Optional[str] = None,
        genre_hint:    Optional[CinematicGenre] = None,
        dialect_hint:  Optional[TeluguDialect]  = None,
        examples:      Optional[list[DialogueExample]] = None,
    ) -> tuple[str, PromptStrategy]:
        """
        Selects and constructs the optimal prompt strategy based on the
        pipeline Config and estimated token budget.

        SELECTION LOGIC:
          1. If examples are provided and token budget permits → Few-Shot.
          2. If Config.model_name matches a CoT-capable model substring AND
             the estimated (CoT overhead + context) fits within the model's
             context window → CoT.
          3. Otherwise → Raw.

        This ensures Sarvam never receives a CoT prompt (which would overflow
        its 7,500-token budget), while Gemma 3 automatically gets the richer
        CoT scaffold when context is small enough to fit.

        Args:
            context:       Formatted multi-turn dialogue string.
            speaker_label: Optional next-speaker label.
            genre_hint:    Optional genre anchor.
            dialect_hint:  Optional dialect anchor.
            examples:      Optional list of gold DialogueExample instances.
                           If provided, few-shot is preferred when budget allows.

        Returns:
            tuple[str, PromptStrategy]: The constructed prompt string and the
            strategy that was selected (for logging/experiment tracking).

        Raises:
            TypeError:  If context is not a string.
            ValueError: If context is empty or whitespace-only.
        """
        self._validate_context(context, prompt_type="Adaptive")

        context_tokens = int(len(context) * self._TOKEN_ESTIMATE_PER_CHAR)
        model_name     = getattr(self.config, "model_name", "").lower()
        max_tokens     = getattr(self.config, "max_tokens", 7_500)

        # ── Few-shot path ─────────────────────────────────────────────────────
        if examples:
            few_shot_tokens = (
                self._TOKEN_ESTIMATE_RAW_OVERHEAD
                + context_tokens
                + len(examples[:3]) * self._TOKEN_ESTIMATE_PER_EXAMPLE
            )
            if few_shot_tokens <= max_tokens:
                prompt = self.build_few_shot_prompt(
                    context, examples, speaker_label, genre_hint, dialect_hint
                )
                self.logger.info(
                    "Adaptive → FEW_SHOT | est_tokens~%d / %d",
                    few_shot_tokens, max_tokens,
                )
                return prompt, PromptStrategy.FEW_SHOT

        # ── CoT path ──────────────────────────────────────────────────────────
        cot_tokens = self._TOKEN_ESTIMATE_COT_OVERHEAD + context_tokens
        is_cot_model = any(sub in model_name for sub in self._COT_CAPABLE_MODEL_SUBSTRINGS)

        if is_cot_model and cot_tokens <= max_tokens:
            prompt = self.build_cot_prompt(
                context, speaker_label, genre_hint, dialect_hint
            )
            self.logger.info(
                "Adaptive → COT | model=%s | est_tokens~%d / %d",
                model_name, cot_tokens, max_tokens,
            )
            return prompt, PromptStrategy.COT

        # ── Raw fallback ──────────────────────────────────────────────────────
        prompt = self.build_raw_prompt(
            context, speaker_label, genre_hint, dialect_hint
        )
        self.logger.info(
            "Adaptive → RAW | model=%s | est_tokens~%d / %d",
            model_name,
            self._TOKEN_ESTIMATE_RAW_OVERHEAD + context_tokens,
            max_tokens,
        )
        return prompt, PromptStrategy.RAW

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Context Formatting
    # ──────────────────────────────────────────────────────────────────────────

    def format_dialogue_context(
        self,
        turns:    list[str],
        speakers: Optional[list[str]] = None,
        dialect:  Optional[TeluguDialect] = None,
        genre:    Optional[CinematicGenre] = None,
    ) -> str:
        """
        Formats a list of raw dialogue turns into a structured multi-turn
        context string suitable for insertion into any prompt template.

        Speaker-labeled output (e.g., "రాజు: ...") is strongly preferred over
        numbered turns when speaker identities are known. Speaker labels allow
        the CoT reasoning stages to identify power relationships, match character
        voice, and enforce honorific consistency with concrete anchors rather
        than inferring speaker identity from pronoun patterns alone.

        An optional header block is prepended when dialect or genre metadata is
        provided, giving the model explicit anchors before the dialogue begins.

        Args:
            turns:    Ordered list of dialogue strings, each one spoken turn.
            speakers: Optional list of speaker label strings, positionally
                      aligned with turns. Unlabeled turns (empty string labels)
                      fall back to "Speaker {n}".
            dialect:  Optional TeluguDialect to embed as a metadata header.
                      Provides an explicit dialect anchor before the dialogue.
            genre:    Optional CinematicGenre to embed as a metadata header.
                      Signals genre to the prompt before any text is read.

        Returns:
            str: Formatted multi-line dialogue string ready for prompt insertion.

        Raises:
            TypeError:  If turns is not a list, contains non-strings, or
                        speakers is provided but is not a list.
            ValueError: If turns is empty, or speakers length ≠ turns length.
        """
        if not isinstance(turns, list):
            raise TypeError(
                f"turns must be a list. Received: {type(turns).__name__}."
            )
        if not turns:
            raise ValueError("turns cannot be empty.")
        if not all(isinstance(t, str) for t in turns):
            raise TypeError("All elements of turns must be strings.")
        if speakers is not None:
            if not isinstance(speakers, list):
                raise TypeError(
                    f"speakers must be a list. Received: {type(speakers).__name__}."
                )
            if len(speakers) != len(turns):
                raise ValueError(
                    f"Length mismatch: {len(turns)} turns but {len(speakers)} "
                    f"speaker labels. They must be the same length."
                )

        # ── Optional metadata header ──────────────────────────────────────────
        header_parts: list[str] = []
        if genre and genre != CinematicGenre.UNSPECIFIED:
            header_parts.append(f"[Genre: {genre.value}]")
        if dialect and dialect != TeluguDialect.UNSPECIFIED:
            header_parts.append(f"[Dialect: {dialect.value}]")
        header = "  ".join(header_parts)

        # ── Dialogue lines ────────────────────────────────────────────────────
        lines: list[str] = []
        for i, turn in enumerate(turns):
            if speakers:
                raw_label = speakers[i].strip()
                label = raw_label if raw_label else f"Speaker {i + 1}"
                lines.append(f"{label}: {turn.strip()}")
            else:
                lines.append(f"Turn {i + 1}: {turn.strip()}")

        body = "\n".join(lines)
        formatted = f"{header}\n{body}".strip() if header else body

        self.logger.debug(
            "Formatted %d dialogue turns | chars=%d | genre=%s | dialect=%s",
            len(turns),
            len(formatted),
            genre.value   if genre   else "—",
            dialect.value if dialect else "—",
        )
        return formatted

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def estimate_prompt_tokens(self, prompt: str) -> int:
        """
        Produces a conservative estimate of prompt token count for Telugu Unicode.

        Telugu Unicode characters tokenize at a higher rate than ASCII/Latin
        text due to multi-byte encoding and subword tokenization behavior.
        This estimate uses a character-to-token ratio calibrated against
        SentencePiece tokenization of mixed Telugu/English prompts.

        This estimate is intentionally conservative (may overcount by ~10–15%)
        to provide a safe budget check. Use actual tokenizer counts for
        production budget enforcement.

        Args:
            prompt: The complete prompt string to estimate.

        Returns:
            int: Estimated token count (conservative upper bound).
        """
        # Telugu Unicode chars tokenize denser than ASCII chars.
        telugu_chars = sum(1 for c in prompt if "\u0C00" <= c <= "\u0C7F")
        ascii_chars  = len(prompt) - telugu_chars
        # Telugu Unicode: ~2.1 tokens/char; ASCII: ~0.27 tokens/char
        estimate = int(telugu_chars * 2.1 + ascii_chars * 0.27)
        return max(estimate, 1)

    def check_token_budget(
        self,
        prompt:     str,
        strategy:   PromptStrategy = PromptStrategy.RAW,
    ) -> dict[str, int | bool | str]:
        """
        Checks whether an assembled prompt fits within the token budget
        defined in the pipeline Config and appropriate for the given strategy.

        Args:
            prompt:   The fully assembled prompt string to check.
            strategy: The PromptStrategy used to construct the prompt.
                      Determines which model token limit to apply.

        Returns:
            dict with keys:
              "estimated_tokens" (int)  : Conservative token count.
              "max_tokens"       (int)  : Budget limit from Config.
              "within_budget"    (bool) : True if estimated ≤ max.
              "headroom"         (int)  : max_tokens - estimated_tokens.
              "strategy"         (str)  : Strategy name.
        """
        estimated  = self.estimate_prompt_tokens(prompt)
        max_tokens = getattr(self.config, "max_tokens", 7_500)
        within     = estimated <= max_tokens

        if not within:
            self.logger.warning(
                "Prompt EXCEEDS token budget | strategy=%s | "
                "estimated=%d | max=%d | overflow=%d",
                strategy.name, estimated, max_tokens, estimated - max_tokens,
            )

        return {
            "estimated_tokens": estimated,
            "max_tokens":       max_tokens,
            "within_budget":    within,
            "headroom":         max_tokens - estimated,
            "strategy":         strategy.name,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_next_turn_cue(
        self,
        speaker_label: Optional[str],
        cot_mode:      bool,
    ) -> str:
        """
        Constructs the terminal cue that primes model generation.

        In raw mode, a speaker label cues character-consistent output.
        In CoT mode, the cue follows after all reasoning steps, so the
        model has already committed to register and rasa before generating.

        Args:
            speaker_label: Optional next speaker identifier.
            cot_mode:      If True, uses CoT-appropriate phrasing.

        Returns:
            str: The next-turn cue string.
        """
        if speaker_label and speaker_label.strip():
            clean = speaker_label.strip()
            return f"{clean}:"
        return "Final Telugu Response:" if cot_mode else "Next Line:"

    def _build_hint_block(
        self,
        genre_hint:   Optional[CinematicGenre],
        dialect_hint: Optional[TeluguDialect],
    ) -> str:
        """
        Constructs an optional genre/dialect priming block for prompt injection.

        Priming blocks are only emitted when the caller has reliable external
        metadata about the genre or dialect. When metadata is absent or
        UNSPECIFIED, the prompt instructs the model to derive these from context —
        which is preferable to injecting an incorrect anchor.

        Args:
            genre_hint:   Optional CinematicGenre value.
            dialect_hint: Optional TeluguDialect value.

        Returns:
            str: Formatted hint block string, or empty string if no hints given.
        """
        parts: list[str] = []

        if genre_hint and genre_hint != CinematicGenre.UNSPECIFIED:
            parts.append(
                f"GENRE ANCHOR: This dialogue is from a "
                f"{genre_hint.value.replace('_', ' ').upper()} film. "
                f"Apply the corresponding cinematic register conventions."
            )
        if dialect_hint and dialect_hint != TeluguDialect.UNSPECIFIED:
            parts.append(
                f"DIALECT ANCHOR: The speakers use "
                f"{dialect_hint.value.replace('_', ' ').title()} Telugu. "
                f"Match all phonological and lexical markers of this dialect."
            )

        if not parts:
            return ""

        return (
            "══════════════════════════════════════════════\n"
            " CONTEXT METADATA\n"
            "══════════════════════════════════════════════\n"
            + "\n".join(parts)
        )

    def _validate_context(self, context: str, prompt_type: str) -> None:
        """
        Validates that the dialogue context is a non-empty string.

        Args:
            context:     The context string to validate.
            prompt_type: Calling prompt type label for diagnostic logging.

        Raises:
            TypeError:  If context is not a string.
            ValueError: If context is empty or whitespace-only.
        """
        if not isinstance(context, str):
            self.logger.error(
                "%s prompt validation failed | expected str, got %s",
                prompt_type, type(context).__name__,
            )
            raise TypeError(
                f"Context must be a string. Received: {type(context).__name__}."
            )
        if not context.strip():
            self.logger.error(
                "%s prompt validation failed | context is empty or whitespace-only",
                prompt_type,
            )
            raise ValueError(
                "Context cannot be empty or consist solely of whitespace."
            )
