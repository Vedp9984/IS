"""
proposed.py — Proposed System (KG + RAG + LLM)
=================================================
Implements the full hybrid advisory pipeline:

  User Query → Entity Extraction → KG Traversal → RAG Retrieval → LLM Generation

This is the proposed system that demonstrates improvement over the baseline.

Key innovations:
  1. Structured reasoning via Knowledge Graph traversal
  2. Retrieval grounding via TF-IDF over expert advisories
  3. Combined context for LLM prompt engineering
  4. Explainable reasoning path from KG
  5. Grounded composition fallback — when no LLM API key is available, the
     system composes its response *directly from retrieved KG/RAG context*,
     so it remains faithful to verified domain knowledge instead of falling
     back to a generic, ungrounded template.
"""

import os
from typing import Dict, List
from kg import AgroKnowledgeGraph, extract_entities_from_query
from rag import RAGSystem
from baseline import get_llm_response


PROPOSED_SYSTEM_PROMPT = """You are an expert agricultural advisory assistant integrated
with a Knowledge Graph and domain-specific retrieval system.

You have been provided with:
1. STRUCTURED KNOWLEDGE from an Agricultural Knowledge Graph (crop properties,
   weather impacts, soil requirements, expert advisories)
2. RETRIEVED DOCUMENTS from a domain-specific advisory corpus

Rules:
- Base your advice on the provided context — do NOT generate information beyond it
- Reference specific data points from the context (e.g., dosage, timing, growth stages)
- If the context provides specific advisories, incorporate their details
- Mention the reasoning: which weather impacts which crop and why
- Include practical action items with specific measures
- Be concise but thorough
- CLEARLY indicate when your advice is grounded in the provided knowledge vs general knowledge
"""


def _has_llm_api() -> bool:
    """Check if any LLM API key is configured."""
    return bool(
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def _compose_grounded_response(query: str, entities: Dict, kg_context: Dict,
                                rag_results: List[Dict]) -> str:
    """
    Compose a grounded advisory directly from retrieved KG advisories and
    RAG documents. Used when no LLM API is available.

    This is *not* a generic template — it stitches together the actual
    domain-verified advisory text(s) retrieved from the KG and RAG corpus,
    preserving specific dosages, chemicals, and growth-stage references.
    Demonstrates the value of structured retrieval even without a generative
    LLM.
    """
    parts: List[str] = []

    crop = entities.get("crop") or "the crop"
    weather = entities.get("weather")
    soil = entities.get("soil")
    category = entities.get("category") or "general"

    # ── Header ──
    header = f"**Advisory for {crop}"
    if weather:
        header += f" during {weather}"
    if soil:
        header += f" ({soil} soil)"
    header += f" — Category: {category.replace('_', ' ').title()}**"
    parts.append(header)

    # ── Reasoning path (KG traversal) ──
    if kg_context.get("kg_reasoning_path"):
        parts.append("\n**Reasoning (from Knowledge Graph):**")
        for step in kg_context["kg_reasoning_path"]:
            parts.append(f"- {step}")

    # ── Weather impact (causal chain from KG) ──
    impacts = kg_context.get("weather_impacts", [])
    if weather and impacts:
        match = [i for i in impacts if weather.lower() in i["condition"].lower()]
        if match:
            mi = match[0]
            parts.append(
                f"\n**Why this matters:** {weather} on {crop} causes "
                f"*{mi['impact']}* (recorded conditions: temp {mi.get('temp_c')}°C, "
                f"rainfall {mi.get('rainfall_mm')}mm)."
            )

    # ── Primary recommendation: KG advisories ──
    advisories = kg_context.get("relevant_advisories", [])
    if advisories:
        parts.append("\n**Expert Recommendation (from KG-linked advisories):**")
        for adv in advisories:
            parts.append(
                f"\n[{adv['category'].replace('_', ' ').title()} | "
                f"{adv['condition']} | {adv['soil']}]"
            )
            parts.append(adv["advisory_text"])

    # ── Supporting evidence: top RAG result (if distinct from KG advisories) ──
    if rag_results:
        adv_ids = {a["id"] for a in advisories}
        extra = [r for r in rag_results if r["id"] not in adv_ids][:2]
        if extra:
            parts.append("\n**Additional Retrieved Evidence (RAG, top hits):**")
            for r in extra:
                parts.append(
                    f"\n- (relevance {r['score']:.3f}) {r['text']}"
                )

    # ── Soil-context note ──
    soil_info = kg_context.get("soil_info", [])
    if soil_info:
        parts.append("\n**Soil Considerations:**")
        for si in soil_info:
            parts.append(
                f"- {si['name']} ({si['soil_type']}, fertility={si['fertility']}): "
                f"{si['reason']}"
            )

    # ── Caused conditions (downstream effects) ──
    caused = kg_context.get("related_conditions", [])
    if caused:
        parts.append(
            "\n**Watch for these downstream effects:** "
            + ", ".join(c["condition"].replace("_", " ") for c in caused) + "."
        )

    # ── Fallback note when KG returned nothing ──
    if not advisories and not rag_results:
        parts.append(
            "\nNo crop/weather-specific advisory was found in the knowledge "
            "base for this query. Consult local agricultural extension services."
        )

    parts.append(
        "\n*This advisory is grounded in the Agricultural Knowledge Graph and "
        "expert advisory corpus. KG reasoning path is shown above for transparency.*"
    )

    return "\n".join(parts)


class ProposedSystem:
    """
    Proposed KG + RAG + LLM hybrid advisory system.

    Pipeline:
      1. Extract entities from query (crop, weather, soil, category)
      2. Query Knowledge Graph for structured context
      3. Retrieve relevant documents via RAG
      4. Combine KG context + RAG context
      5. Generate advisory via LLM with full context
         (or compose from retrieved context if no API key is available)
    """

    def __init__(self):
        self.kg = AgroKnowledgeGraph()
        self.rag = RAGSystem(use_kg_context=True)
        self.system_prompt = PROPOSED_SYSTEM_PROMPT
        self.response_cache = {}

    def generate_advisory(self, query: str) -> Dict:
        """
        Full pipeline: Query → Entities → KG → RAG → LLM → Advisory

        Returns comprehensive result dict for evaluation.
        """
        if query in self.response_cache:
            return self.response_cache[query]

        # ── Step 1: Entity Extraction ──
        entities = extract_entities_from_query(query)

        # ── Step 2: Knowledge Graph Query ──
        kg_context = self.kg.query_context(
            crop_name=entities.get("crop"),
            weather=entities.get("weather"),
            soil=entities.get("soil"),
            category=entities.get("category"),
        )
        kg_context_text = self.kg.format_context_for_llm(kg_context)

        # ── Step 3: RAG Retrieval ──
        rag_results = self.rag.retrieve_context(query, top_k=3)
        rag_context_text = self.rag.format_retrieved_context(rag_results)
        rag_stats = self.rag.get_retrieval_stats(rag_results)

        # ── Step 4: Combine Context ──
        combined_context = (
            f"{kg_context_text}\n\n"
            f"{rag_context_text}\n"
        )

        # ── Step 5: Generation ──
        if _has_llm_api():
            prompt = (
                f"A farmer is asking for advice:\n\n"
                f"\"{query}\"\n\n"
                f"--- CONTEXT FROM KNOWLEDGE SYSTEMS ---\n\n"
                f"{combined_context}\n\n"
                f"--- END OF CONTEXT ---\n\n"
                f"Based on the above structured knowledge and retrieved documents, "
                f"provide a detailed, practical agricultural advisory response. "
                f"Ground your answer in the provided context."
            )
            response = get_llm_response(prompt, self.system_prompt)
        else:
            # No LLM API available — compose response *directly from retrieved context*.
            # This is the key behaviour that lets the proposed system outperform
            # the baseline even in offline / template-fallback mode.
            response = _compose_grounded_response(
                query, entities, kg_context, rag_results
            )

        # ── Compile Result ──
        result = {
            "query": query,
            "system": "proposed",
            "entities_extracted": entities,
            "kg_context": kg_context_text,
            "kg_reasoning_path": kg_context.get("kg_reasoning_path", []),
            "rag_results": rag_results,
            "rag_stats": rag_stats,
            "combined_context": combined_context,
            "response": response,
            "has_kg": True,
            "has_rag": True,
            "num_advisories_found": len(kg_context.get("relevant_advisories", [])),
            "num_weather_impacts": len(kg_context.get("weather_impacts", [])),
        }

        self.response_cache[query] = result
        return result


# ──────────────────── Main (demo) ────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PROPOSED SYSTEM — KG + RAG + LLM Advisory Demo")
    print("=" * 60)

    system = ProposedSystem()

    test_queries = [
        "What irrigation advice should be given for wheat during drought conditions in alluvial soil?",
        "How to manage pests in rice during the monsoon season?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        result = system.generate_advisory(query)
        print(f"Query: {result['query']}")
        print(f"\nEntities: {result['entities_extracted']}")
        print(f"\n--- KG Context ---\n{result['kg_context'][:500]}...")
        print(f"\n--- RAG Stats ---\n{result['rag_stats']}")
        print(f"\n--- Response ---\n{result['response'][:800]}...")
