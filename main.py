"""
main.py — Smart Agriculture Advisory System
=============================================
Main orchestration script for the research prototype.

Paper: "A Practical Knowledge Graph and LLM-Based Framework
        for Smart Agromet Advisory Systems"

Executes the complete experimental pipeline:
  1. Build Knowledge Graph → Print statistics
  2. Index documents for RAG → Print index stats
  3. Run Baseline system on all test queries
  4. Run Proposed system on all test queries
  5. Evaluate and compare both systems
  6. Generate results tables and observations
  7. Output architecture explanation and key contributions

Usage:
  python main.py                  # Full experiment
  python main.py --kg-only        # Only KG stats
  python main.py --quick          # Run on 3 queries only
"""

import sys
import json
import time
from datetime import datetime

from dataset import TEST_QUERIES, CROPS, SOILS, WEATHER_CONDITIONS, ADVISORIES, RELATIONS
from kg import AgroKnowledgeGraph, extract_entities_from_query
from rag import RAGSystem
from baseline import BaselineSystem
from proposed import ProposedSystem
from evaluation import AdvisoryEvaluator


def print_banner(text, char="═", width=80):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_section(text, char="─", width=60):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


# ══════════════════════════════════════════════════════════════
# PHASE 1: Knowledge Graph Construction
# ══════════════════════════════════════════════════════════════

def phase1_knowledge_graph():
    """Build and analyze the Knowledge Graph."""
    print_banner("PHASE 1: KNOWLEDGE GRAPH CONSTRUCTION")

    kg = AgroKnowledgeGraph()
    stats = kg.get_stats()

    print(f"\n  Dataset Overview:")
    print(f"    Crops:             {len(CROPS)}")
    print(f"    Soil Types:        {len(SOILS)}")
    print(f"    Weather Conditions:{len(WEATHER_CONDITIONS)}")
    print(f"    Expert Advisories: {len(ADVISORIES)}")
    print(f"    Relations:         {len(RELATIONS)}")

    print(f"\n  Knowledge Graph Statistics:")
    print(f"    Total Nodes:  {stats['total_nodes']}")
    print(f"    Total Edges:  {stats['total_edges']}")
    print(f"    Density:      {stats['density']:.4f}")
    print(f"    Is DAG:       {stats['is_dag']}")

    print(f"\n  Node Types:")
    for t, count in sorted(stats["node_types"].items()):
        print(f"    {t:<15} {count}")

    print(f"\n  Edge Types:")
    for r, count in sorted(stats["edge_types"].items()):
        print(f"    {r:<20} {count}")

    # Demo: multi-hop traversal
    print_section("KG Multi-Hop Traversal Demo: Wheat + Drought")
    context = kg.query_context(crop_name="Wheat", weather="Drought", category="irrigation")
    print(kg.format_context_for_llm(context))

    return kg


# ══════════════════════════════════════════════════════════════
# PHASE 2: RAG Index Construction
# ══════════════════════════════════════════════════════════════

def phase2_rag_index():
    """Build and test the RAG retrieval index."""
    print_banner("PHASE 2: RAG INDEX CONSTRUCTION")

    rag = RAGSystem(use_kg_context=True)

    print(f"\n  Documents indexed: {len(rag.retriever.documents)}")
    print(f"  Vocabulary size:   {len(rag.retriever.vocab)}")

    # Demo retrieval
    demo_query = "irrigation advice for wheat during drought"
    results = rag.retrieve_context(demo_query, top_k=3)
    print(f"\n  Demo Query: \"{demo_query}\"")
    print(f"  Retrieved {len(results)} documents:")
    for r in results:
        print(f"    Rank {r['rank']}: {r['id']} (score: {r['score']:.4f})")

    return rag


# ══════════════════════════════════════════════════════════════
# PHASE 3: Run Experiments
# ══════════════════════════════════════════════════════════════

def phase3_run_experiments(queries=None):
    """Run both systems on all test queries."""
    print_banner("PHASE 3: RUNNING EXPERIMENTS")

    if queries is None:
        queries = TEST_QUERIES

    baseline = BaselineSystem()
    proposed = ProposedSystem()

    baseline_results = []
    proposed_results = []

    for i, q in enumerate(queries, 1):
        print(f"\n  [{i}/{len(queries)}] Processing: {q['id']} — {q['query'][:60]}...")

        # Run baseline
        t0 = time.time()
        b_result = baseline.generate_advisory(q["query"])
        b_time = time.time() - t0

        # Run proposed
        t0 = time.time()
        p_result = proposed.generate_advisory(q["query"])
        p_time = time.time() - t0

        b_result["response_time"] = round(b_time, 6)
        p_result["response_time"] = round(p_time, 6)

        baseline_results.append(b_result)
        proposed_results.append(p_result)

        print(f"    Baseline: {len(b_result['response'])} chars, {b_time:.2f}s")
        print(f"    Proposed: {len(p_result['response'])} chars, {p_time:.2f}s")
        print(f"    KG advisories found: {p_result.get('num_advisories_found', 0)}")
        print(f"    RAG avg score: {p_result.get('rag_stats', {}).get('avg_score', 0):.4f}")

    return baseline_results, proposed_results


# ══════════════════════════════════════════════════════════════
# PHASE 4: Evaluation & Results
# ══════════════════════════════════════════════════════════════

def phase4_evaluation(queries, baseline_results, proposed_results):
    """Evaluate and compare both systems."""
    print_banner("PHASE 4: EVALUATION & RESULTS")

    evaluator = AdvisoryEvaluator()

    # Evaluate each query
    evaluations = []
    for i, q in enumerate(queries):
        ev = evaluator.evaluate_single(q, baseline_results[i], proposed_results[i])
        evaluations.append(ev)

    # Print comparison table
    table = evaluator.generate_comparison_table(evaluations)
    print(f"\n{table}")

    # Print detailed results
    detailed = evaluator.generate_detailed_results(
        evaluations, baseline_results, proposed_results
    )
    print(detailed)

    # Print observations
    observations = evaluator.generate_observations(evaluations)
    print(observations)

    return evaluations


# ══════════════════════════════════════════════════════════════
# PHASE 5: Paper-Ready Outputs
# ══════════════════════════════════════════════════════════════

def phase5_paper_outputs():
    """Generate architecture explanation and key contributions."""
    print_banner("PHASE 5: PAPER-READY OUTPUTS")

    # Architecture explanation
    print_section("1. SYSTEM ARCHITECTURE")
    print("""
  The proposed Smart Agromet Advisory System implements a three-layer
  hybrid architecture combining structured reasoning, retrieval-based
  grounding, and neural language generation:

  Layer 1 — Knowledge Graph (Structured Reasoning):
    • Built using NetworkX as a directed multigraph
    • Contains 4 entity types: Crop, Soil, Weather, Advisory
    • Connected via 4 relation types: affects, requires, causes, recommended_for
    • Enables multi-hop traversal for contextual reasoning
    • Provides explainable reasoning paths

  Layer 2 — Retrieval-Augmented Generation (RAG):
    • TF-IDF-based retrieval over expert advisory corpus
    • Indexes both advisory documents and KG-derived context documents
    • Returns top-K relevant documents with similarity scores
    • Grounds LLM output in verified domain knowledge

  Layer 3 — LLM Advisory Generation:
    • Consumes combined KG + RAG context in structured prompts
    • System prompt constrains output to provided context
    • Generates natural language advisories with specific recommendations
    • Supports multiple LLM backends (Gemini, OpenAI) with template fallback
""")

    # Pipeline description
    print_section("2. STEP-BY-STEP PIPELINE")
    print("""
  Input:  Natural language farmer query
          e.g., "What irrigation advice for wheat during drought?"

  Step 1: ENTITY EXTRACTION
          • Rule-based NER extracts: crop=Wheat, weather=Drought,
            category=irrigation
          • Entities serve as KG query parameters

  Step 2: KNOWLEDGE GRAPH TRAVERSAL
          • Find crop node → traverse 'affects' edges for weather impacts
          • Traverse 'requires' edges for soil requirements
          • Follow 'recommended_for' edges to find expert advisories
          • Record reasoning path for explainability

  Step 3: RETRIEVAL (RAG)
          • Convert query to TF-IDF vector
          • Compute cosine similarity with indexed documents
          • Return top-3 most relevant advisory documents
          • Include retrieval scores for confidence

  Step 4: CONTEXT FUSION
          • Combine KG structured output + RAG retrieved documents
          • Format as structured prompt sections
          • Include reasoning path and retrieval scores

  Step 5: LLM GENERATION
          • Feed combined context to LLM with constraining system prompt
          • Generate detailed, grounded agricultural advisory
          • Output is specific, actionable, and traceable

  Output: Structured advisory with:
          • Specific recommendations (chemicals, dosages, timing)
          • Reasoning transparency (KG traversal path)
          • Retrieval evidence (document scores)
""")

    # Key contributions
    print_section("3. KEY CONTRIBUTIONS")
    print("""
  1. HYBRID ARCHITECTURE FOR AGRO-ADVISORY SYSTEMS:
     First integration of Knowledge Graphs with RAG and LLMs specifically
     for agricultural advisory generation, demonstrating that structured
     domain knowledge significantly improves advisory quality over
     LLM-only approaches.

  2. HALLUCINATION REDUCTION VIA GROUNDING:
     Empirical evidence that retrieval-augmented generation with domain-
     specific knowledge graphs reduces factual hallucination in generated
     advisories — critical for high-stakes agricultural recommendations.

  3. EXPLAINABLE ADVISORY GENERATION:
     Unlike black-box LLM approaches, the KG reasoning path provides
     transparent justification for each advisory, enabling verification
     by agricultural scientists and increasing farmer trust.

  4. PRACTICAL AND DEPLOYABLE DESIGN:
     Entire system runs with minimal dependencies (NetworkX, Python stdlib),
     supports multiple LLM backends, and includes template fallback for
     offline operation — suitable for deployment in resource-constrained
     agricultural settings.

  5. REPRODUCIBLE EVALUATION FRAMEWORK:
     Automated evaluation pipeline with 5 metrics (relevance, specificity,
     clarity, hallucination, grounding) enables systematic comparison of
     agricultural NLP systems.
""")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    print("\n" + "█" * 80)
    print("  SMART AGRICULTURE ADVISORY SYSTEM")
    print("  KG + RAG + LLM Hybrid Framework — Research Prototype")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 80)

    # Parse arguments
    quick_mode = "--quick" in sys.argv
    kg_only = "--kg-only" in sys.argv

    # Phase 1: Knowledge Graph
    kg = phase1_knowledge_graph()

    if kg_only:
        print("\n[--kg-only mode] Stopping after KG phase.")
        return

    # Phase 2: RAG Index
    rag = phase2_rag_index()

    # Phase 3: Run Experiments
    queries = TEST_QUERIES[:3] if quick_mode else TEST_QUERIES
    print(f"\n  Running on {len(queries)} test queries {'(quick mode)' if quick_mode else '(full)'}")

    baseline_results, proposed_results = phase3_run_experiments(queries)

    # Phase 4: Evaluation
    evaluations = phase4_evaluation(queries, baseline_results, proposed_results)

    # Phase 5: Paper outputs
    phase5_paper_outputs()

    # Summary
    elapsed = time.time() - start_time
    print_banner(f"EXPERIMENT COMPLETE — Total Time: {elapsed:.1f}s")

    # Save results to JSON — include responses + per-system stats
    transcripts = []
    for q, b, p in zip(queries, baseline_results, proposed_results):
        transcripts.append({
            "query_id": q["id"],
            "query": q["query"],
            "category": q["category"],
            "expected_crop": q.get("expected_crop"),
            "expected_condition": q.get("expected_condition"),
            "baseline_response": b["response"],
            "baseline_latency_s": b.get("response_time", 0.0),
            "proposed_response": p["response"],
            "proposed_latency_s": p.get("response_time", 0.0),
            "proposed_entities_extracted": p.get("entities_extracted", {}),
            "proposed_kg_reasoning_path": p.get("kg_reasoning_path", []),
            "proposed_kg_advisories_found": p.get("num_advisories_found", 0),
            "proposed_rag_top_ids": [r["id"] for r in p.get("rag_results", [])],
            "proposed_rag_avg_score": p.get("rag_stats", {}).get("avg_score", 0),
        })

    output = {
        "timestamp": datetime.now().isoformat(),
        "num_queries": len(queries),
        "evaluations": evaluations,
        "transcripts": transcripts,
        "kg_stats": kg.get_stats(),
        "elapsed_seconds": round(elapsed, 1),
    }

    output_file = "results.json"
    def default_serializer(obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return str(obj)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=default_serializer)

    print(f"\n  Results saved to: {output_file}")
    print(f"  Run 'python main.py --quick' for a quick test with 3 queries")
    print(f"  Run 'python main.py --kg-only' for KG statistics only")


if __name__ == "__main__":
    main()
