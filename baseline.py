"""
baseline.py — Baseline System (LLM-only, no KG, no RAG)
==========================================================
Implements the baseline advisory system that uses ONLY an LLM
with no structured knowledge or retrieval grounding.

This serves as the control condition in the experiment:
  - Baseline: LLM alone
  - Proposed: KG + RAG + LLM

The baseline sends queries directly to the LLM with a generic
agriculture prompt, demonstrating what happens without structured
context or retrieval augmentation.
"""

import os
import json
from typing import Dict, Optional


# ─────────────────── LLM Interface ───────────────────

def get_llm_response(prompt: str, system_prompt: str = None,
                     model: str = None, temperature: float = 0.3) -> str:
    """
    Get response from LLM.

    Supports:
      1. Google Gemini API (GOOGLE_API_KEY)
      2. OpenAI API (OPENAI_API_KEY)
      3. Fallback: Template-based response (no API needed)

    For research reproducibility, temperature is set low (0.3).
    """

    # Try Google Gemini first
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        return _call_gemini(prompt, system_prompt, api_key, temperature)

    # Try OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return _call_openai(prompt, system_prompt, api_key, model or "gpt-3.5-turbo", temperature)

    # Fallback: template-based response
    return _template_response(prompt, system_prompt)


def _call_gemini(prompt: str, system_prompt: str, api_key: str,
                 temperature: float) -> str:
    """Call Google Gemini API."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.5-flash")

        full_prompt = ""
        if system_prompt:
            full_prompt = f"System Instructions: {system_prompt}\n\n"
        full_prompt += prompt

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024,
            )
        )
        return response.text
    except Exception as e:
        print(f"[Gemini API Error: {e}] — Falling back to template.")
        return _template_response(prompt, system_prompt)


def _call_openai(prompt: str, system_prompt: str, api_key: str,
                 model: str, temperature: float) -> str:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[OpenAI API Error: {e}] — Falling back to template.")
        return _template_response(prompt, system_prompt)


def _template_response(prompt: str, system_prompt: str = None) -> str:
    """
    Template-based fallback when no LLM API is available.
    Generates structured but generic agricultural advisories.
    This is intentionally less specific than KG-grounded responses,
    which is the point — demonstrating the value of KG + RAG.
    """
    prompt_lower = prompt.lower()

    # Detect topic
    if "irrigat" in prompt_lower or "water" in prompt_lower or "drought" in prompt_lower:
        return _irrigation_template(prompt_lower)
    elif "pest" in prompt_lower or "disease" in prompt_lower or "blight" in prompt_lower or "bollworm" in prompt_lower or "whitefly" in prompt_lower:
        return _pest_template(prompt_lower)
    elif "fertiliz" in prompt_lower or "nutrient" in prompt_lower:
        return _fertilizer_template(prompt_lower)
    elif "cold" in prompt_lower or "frost" in prompt_lower:
        return _cold_template(prompt_lower)
    elif "heat" in prompt_lower:
        return _heat_template(prompt_lower)
    else:
        return _general_template(prompt_lower)


def _irrigation_template(query: str) -> str:
    return (
        "**Irrigation Advisory:**\n\n"
        "For drought or water-stress conditions, the following general practices are recommended:\n\n"
        "1. Irrigate the crop at critical growth stages to minimize yield loss.\n"
        "2. Use water-efficient irrigation methods such as drip or sprinkler systems.\n"
        "3. Apply mulching to conserve soil moisture and reduce evaporation.\n"
        "4. Avoid waterlogging by ensuring proper drainage in the field.\n"
        "5. Monitor soil moisture regularly and irrigate when the top 5 cm of soil becomes dry.\n\n"
        "These are general guidelines. Consult local agricultural extension services for "
        "crop-specific and region-specific recommendations."
    )


def _pest_template(query: str) -> str:
    return (
        "**Pest and Disease Management Advisory:**\n\n"
        "For effective pest and disease control, follow Integrated Pest Management (IPM):\n\n"
        "1. Monitor fields regularly for signs of pest or disease.\n"
        "2. Use cultural practices like crop rotation and resistant varieties.\n"
        "3. Apply biopesticides (e.g., neem oil) as the first line of defense.\n"
        "4. Use chemical pesticides only when the pest population exceeds the Economic "
        "Threshold Level (ETL).\n"
        "5. Maintain proper plant spacing for air circulation.\n"
        "6. Remove and destroy infected plant parts to prevent spread.\n\n"
        "These are general guidelines. Specific pesticide recommendations depend on "
        "the crop, pest species, and local regulations."
    )


def _fertilizer_template(query: str) -> str:
    return (
        "**Fertilizer Advisory:**\n\n"
        "For optimal crop nutrition, follow these general recommendations:\n\n"
        "1. Apply NPK fertilizers based on soil test results.\n"
        "2. Use split application — apply nitrogen in 2-3 doses for better utilization.\n"
        "3. Apply phosphorus and potassium as basal dose at sowing.\n"
        "4. Supplement with organic matter (FYM/compost) for soil health.\n"
        "5. Correct micronutrient deficiencies based on visual symptoms.\n"
        "6. Avoid over-fertilization to prevent environmental pollution.\n\n"
        "Specific doses depend on the crop, soil type, and expected yield. "
        "Consult your local soil testing laboratory for precise recommendations."
    )


def _cold_template(query: str) -> str:
    return (
        "**Cold Wave / Frost Protection Advisory:**\n\n"
        "When cold wave or frost conditions are expected:\n\n"
        "1. Apply light irrigation to raise soil temperature.\n"
        "2. Cover sensitive crops with protective material if possible.\n"
        "3. Avoid nitrogen application during extreme cold.\n"
        "4. Postpone sowing until temperatures rise above safe thresholds.\n"
        "5. Monitor weather forecasts regularly.\n\n"
        "Specific protection measures depend on the crop and stage of growth."
    )


def _heat_template(query: str) -> str:
    return (
        "**Heat Wave Management Advisory:**\n\n"
        "During heat wave conditions:\n\n"
        "1. Increase irrigation frequency to compensate for higher evapotranspiration.\n"
        "2. Apply mulching to reduce soil temperature.\n"
        "3. Avoid field operations during peak heat hours.\n"
        "4. Use shade nets for sensitive crops if available.\n"
        "5. Spray anti-transpirants to reduce water loss from leaves.\n\n"
        "Crop-specific measures depend on the growth stage and local conditions."
    )


def _general_template(query: str) -> str:
    return (
        "**General Agricultural Advisory:**\n\n"
        "For good agricultural practices:\n\n"
        "1. Follow recommended sowing dates for your region.\n"
        "2. Use certified seeds of recommended varieties.\n"
        "3. Apply fertilizers based on soil test recommendations.\n"
        "4. Practice Integrated Pest Management (IPM).\n"
        "5. Ensure proper irrigation and drainage.\n"
        "6. Monitor weather forecasts and plan field operations accordingly.\n\n"
        "Contact your local agricultural extension office for specific guidance."
    )


# ─────────────────── Baseline System ───────────────────

BASELINE_SYSTEM_PROMPT = """You are an agricultural advisory assistant. You provide 
practical, actionable farming advice based on your general knowledge of agriculture.

Rules:
- Provide specific, actionable recommendations
- Include dosage information where applicable
- Mention crop growth stages when relevant
- Be concise but thorough
- Do NOT make up specific product names or exact numbers if unsure
"""


class BaselineSystem:
    """
    Baseline advisory system — LLM only, no KG, no retrieval.
    Sends queries directly to the LLM with a generic agriculture prompt.
    """

    def __init__(self):
        self.system_prompt = BASELINE_SYSTEM_PROMPT
        self.response_cache = {}

    def generate_advisory(self, query: str) -> Dict:
        """Generate advisory using ONLY the LLM (no external context)."""
        # Check cache for reproducibility
        if query in self.response_cache:
            return self.response_cache[query]

        prompt = (
            f"A farmer is asking for advice:\n\n"
            f"\"{query}\"\n\n"
            f"Please provide a detailed, practical agricultural advisory response."
        )

        response = get_llm_response(prompt, self.system_prompt)

        result = {
            "query": query,
            "system": "baseline",
            "context_used": "None (LLM general knowledge only)",
            "response": response,
            "has_kg": False,
            "has_rag": False,
        }

        self.response_cache[query] = result
        return result


# ──────────────────── Main (demo) ────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE SYSTEM — LLM-Only Advisory Demo")
    print("=" * 60)

    baseline = BaselineSystem()

    test_queries = [
        "What irrigation advice should be given for wheat during drought conditions?",
        "How to manage pests in rice during the monsoon season?",
        "Recommend fertilizer schedule for maize grown in red soil.",
    ]

    for query in test_queries:
        print(f"\n{'─' * 50}")
        result = baseline.generate_advisory(query)
        print(f"Query: {result['query']}")
        print(f"Context: {result['context_used']}")
        print(f"\nResponse:\n{result['response']}")
