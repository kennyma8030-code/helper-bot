"""Core agent loop — LLM-directed retrieval over the message store (specs.md Part 1).

Shape: triage -> budgeted while loop (plan+judge per pass) -> synthesize.
One model, role-scoped prompts, native function calling with manual dispatch.
State lives in an append-only ledger; raw rows live for exactly one pass.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai

import db
from llm import client
from prompts import PLANNER_PROMPT, SYNTH_PROMPT, TRIAGE_PROMPT

load_dotenv()

log = logging.getLogger(__name__)

MODEL = "gemini-3.5-flash"

# Hard budget — enforced in code, not by the prompt (specs.md decision 2).
MAX_PASSES = 16
MAX_RETRIEVALS = 40
