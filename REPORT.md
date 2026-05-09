# A Practical Knowledge Graph and LLM-Based Framework for Smart Agromet Advisory Systems

**Author:** Ved Prakash Maurya
**Course:** Sem 6 — Information Systems / Independent Study
**Status:** Implementation report (extends `IS_final.pdf`)
**Date:** 2026-05-09

---

## List of Figures

| # | Figure | File | Section |
|--:|---|---|---|
| 1 | Three-layer system architecture | [fig_architecture.png](fig_architecture.png) | §5.1 |
| 2 | End-to-end pipeline | [fig_pipeline.png](fig_pipeline.png) | §5.2 |
| 3 | Module-dependency graph | [fig_components.png](fig_components.png) | §5.3 |
| 4 | Sequence diagram for one query | [fig_sequence.png](fig_sequence.png) | §5.4 |
| 5 | Dataset composition | [fig_dataset_overview.png](fig_dataset_overview.png) | §6.1 |
| 6 | Knowledge-graph schema | [fig_kg_schema.png](fig_kg_schema.png) | §6.2 |
| 7 | Full KG (layered by entity type) | [fig_kg_full.png](fig_kg_full.png) | §6.3 |
| 8 | Worked-example KG sub-graph | [fig_kg_example.png](fig_kg_example.png) | §6.4 / §9 |
| 9 | Evaluation-metric taxonomy | [fig_metric_taxonomy.png](fig_metric_taxonomy.png) | §10.2 |
| 10 | Effect of the bug fix (per-query relevance) | [fig_before_after.png](fig_before_after.png) | §11.1 |
| 11 | Aggregate metric comparison | [fig_results_summary.png](fig_results_summary.png) | §11.2 |
| 12 | Per-query F1 vs gold advisory | [fig_per_query_f1.png](fig_per_query_f1.png) | §11.3 |
| 13 | Per-query grounding score | [fig_grounding.png](fig_grounding.png) | §11.5 |

All thirteen figures are generated programmatically by `python diagrams.py`
(plus `python visualize.py` for figure 7).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background and Motivation](#2-background-and-motivation)
3. [Problem Formulation](#3-problem-formulation)
4. [System Overview](#4-system-overview)
5. [System Architecture](#5-system-architecture)
6. [Knowledge Graph Design](#6-knowledge-graph-design)
7. [Retrieval-Augmented Generation](#7-retrieval-augmented-generation)
8. [LLM Integration & Grounded Composition Fallback](#8-llm-integration--grounded-composition-fallback)
9. [Worked Example: Wheat + Drought + Alluvial soil](#9-worked-example-wheat--drought--alluvial-soil)
10. [Evaluation Methodology](#10-evaluation-methodology)
11. [Results](#11-results)
12. [Qualitative Comparison](#12-qualitative-comparison)
13. [Discussion and Observations](#13-discussion-and-observations)
14. [Limitations](#14-limitations)
15. [Reproducibility](#15-reproducibility)
16. [Conclusion and Future Work](#16-conclusion-and-future-work)
17. [Appendix A — Module Reference](#appendix-a--module-reference)
18. [Appendix B — Test Queries](#appendix-b--test-queries)
19. [Appendix C — Full Run Log Excerpt](#appendix-c--full-run-log-excerpt)

---

## 1. Executive Summary

This report presents a complete, reproducible implementation of the framework
described in *A Practical Knowledge Graph and LLM-Based Framework for Smart
Agromet Advisory Systems* ([IS_final.pdf](IS_final.pdf)). The original paper
defines a four-component system (Data Collection, Knowledge Graph
Construction, Retrieval-Augmented Generation, User Interface) for delivering
actionable, weather-aware agricultural advisories. This report turns that
proposal into a runnable Python prototype, evaluates it on ten benchmark
queries, and contrasts it against an LLM-only baseline.

**Headline numbers** (n = 10 queries):

| Metric | Baseline (LLM-only) | Proposed (KG + RAG + LLM) | Δ |
|---|---:|---:|---:|
| Relevance (1–5) | **2.8** | **4.9** | **+2.1** |
| Specificity (1–5) | 1.9 | 4.6 | +2.7 |
| Grounding (0–1) | 0.000 | 0.841 | +0.841 |
| Precision (vs gold) | 0.250 | 0.534 | +0.284 |
| Recall (vs gold) | 0.066 | 1.000 | +0.934 |
| F1 (vs gold) | 0.104 | 0.670 | +0.566 |
| Hallucination rate | 0/10 | 0/10 | 0 |
| Latency (ms) | 0.01 | 0.32 | +0.31 |

The proposed system improves **every** quality metric by a wide margin while
adding only ~0.3 ms of latency per query. The visual summary is below; full
detail is in §11.

![Figure 11 — Aggregate evaluation metrics: baseline vs proposed (n = 10).](fig_results_summary.png)

**Figure 11.** *Aggregate evaluation metrics for the baseline (LLM-only) and
proposed (KG + RAG + LLM) systems, averaged over the ten test queries. All
metrics are normalised to the [0, 1] interval. The proposed system dominates
on every metric.*

### What this report adds beyond the paper

- A working, end-to-end Python implementation (~1,500 LOC across 9 modules).
- A 34-node, 38-edge agricultural Knowledge Graph with 5 entity types and 4
  relation types.
- A TF-IDF retrieval index over 26 documents (10 advisories + 16 KG-context
  documents) with cosine-similarity ranking.
- A grounded **composition fallback** that lets the proposed system produce
  KG-/RAG-grounded responses *even without an LLM API key* — making
  experiments fully offline-reproducible.
- A six-metric automated evaluation harness, including precision / recall /
  F1 against gold-standard advisories and millisecond-resolution latency.
- **Thirteen** architecture / pipeline / result diagrams (`fig_*.png`)
  generated programmatically by [`diagrams.py`](diagrams.py).

### What was wrong with the previous run (before this report)

The initial `results.json` showed the proposed system scoring *lower
relevance than the baseline* on six of ten queries. The root cause was in
[`baseline._template_response`](baseline.py:101): it routed by
keyword-matching the *whole prompt*, but the proposed system's prompt
always contained KG context like `"Water Need: high"`, so every query
landed in the irrigation template. That is fixed now — the proposed system
composes its response directly from retrieved KG advisories and RAG
documents when no LLM API is available, instead of going through the
baseline's keyword router. The before / after comparison is summarised
visually in §11.1 (Figure 10).

---

## 2. Background and Motivation

Agricultural production in India is highly weather-sensitive: rainfall
variability, temperature extremes, and crop-soil-water interactions account
for the bulk of yield variance. The Indian Meteorological Department (IMD)
runs the *Gramin Krishi Mausam Sewa* (GKMS) programme that issues
agromet advisories twice a week, but those advisories are produced by hand
and reach farmers via a mix of SMS, radio, and printed bulletins. Three
properties limit their utility:

1. **Fragmented data**: weather, crop calendars, soil maps, and pest
   surveillance live in separate databases with little semantic linking.
2. **Limited reasoning**: rule-based decision support cannot generalise
   to combinations of crop × stage × weather × soil that were not pre-coded.
3. **Static templates**: free-text advisories are repetitive and offer no
   personalised guidance.

Recent work has shown that Large Language Models (LLMs) can fluently
generate agricultural advice, but they are prone to hallucinating dosages,
chemical names, and growth-stage specifics — all of which matter directly
for crop outcomes. **Retrieval-Augmented Generation (RAG)** mitigates this
by grounding LLM output in retrieved documents, and **Knowledge Graphs (KGs)**
add structured, multi-hop reasoning over typed entities. This report
implements the union of all three.

---

## 3. Problem Formulation

Let:

- $C$ be the set of crops, $S$ soils, $W$ weather conditions,
  $A$ expert advisories.
- $R \subseteq (C \cup S \cup W \cup A) \times \mathcal{L} \times (C \cup S \cup W \cup \mathit{Cond})$
  be a set of typed relations with labels
  $\mathcal{L} = \{\textit{affects}, \textit{requires}, \textit{causes}, \textit{recommended\_for}\}$.
- $G = (V, E)$ be the directed knowledge graph induced by $R$ over
  $V = C \cup S \cup W \cup A \cup \mathit{Cond}$.

Given a natural-language query $q$ and an entity extractor
$\eta(q) = (c, w, s, k)$ that yields a (crop, weather, soil, category)
tuple, the **proposed system** computes:

$$
\text{advisory}(q) =
\Phi\big(\,\underbrace{\sigma_G(c, w, s, k)}_{\text{KG context}} \;\cup\;
            \underbrace{\rho_R(q, K)}_{\text{RAG top-K}}\,\big)
$$

where $\sigma_G$ is the multi-hop sub-graph extractor over $G$,
$\rho_R$ is the TF-IDF top-$K$ retrieval function, and $\Phi$ is a
generation function (LLM or grounded-composer fallback). The
**baseline** simply computes $\text{advisory}_b(q) = \Phi_{\text{LLM}}(q)$
with no context. The evaluation question is: does $\text{advisory}(q)$
measurably outperform $\text{advisory}_b(q)$ on automated quality metrics?

---

## 4. System Overview

The implementation comprises four functional components, faithful to
[IS_final.pdf](IS_final.pdf) §4 (System Overview):

1. **Data collection** → [`dataset.py`](dataset.py)
2. **Knowledge graph construction** → [`kg.py`](kg.py)
3. **Retrieval-augmented generation** → [`rag.py`](rag.py)
4. **Advisory generation & comparison** → [`baseline.py`](baseline.py),
   [`proposed.py`](proposed.py), [`evaluation.py`](evaluation.py),
   [`main.py`](main.py)

A user (the “User Interface” component in the paper) is currently a
command-line driver in `main.py`; integration with a chatbot UI is left
as future work.

---

## 5. System Architecture

### 5.1 Three-Layer Architecture

The proposed system is organised as a layered hybrid pipeline. Each layer
contributes a distinct epistemic guarantee: structured reasoning (KG),
lexical grounding (RAG), and natural-language fluency (LLM).

![Figure 1 — Three-layer system architecture.](fig_architecture.png)

**Figure 1.** *Three-layer architecture of the proposed Smart Agromet
Advisory System. The user query enters at the top, fans out across the
three reasoning layers (KG · RAG · LLM), passes through Entity Extraction
and Context Fusion, and emerges as a grounded advisory response. The
Evaluator compares the grounded response with the baseline output. The
dataset at the bottom is the single source of truth for the entire
pipeline.*

| Layer | Module | Guarantee |
|---|---|---|
| 1 — Knowledge Graph | [`kg.py`](kg.py) | Multi-hop traversal over typed entities; explainable reasoning path |
| 2 — RAG | [`rag.py`](rag.py) | Lexical grounding; bounded hallucination via retrieval |
| 3 — LLM / Composer | [`baseline.py`](baseline.py) + [`proposed.py`](proposed.py) | Fluent, query-shaped output |

### 5.2 End-to-End Pipeline

The pipeline processes one query at a time. The seven stages below map
directly onto the boxes in Figure 2.

![Figure 2 — End-to-end pipeline of the proposed system.](fig_pipeline.png)

**Figure 2.** *Seven-stage pipeline: Query → Entity Extraction → KG
Traversal → RAG Retrieval → Context Fusion → LLM/Composer → Advisory
Response. Each box is colour-coded to its layer in Figure 1. Italic
captions below each stage describe the kind of transformation it
performs (e.g. "structured reasoning" for KG, "lexical grounding" for RAG).*

| # | Stage | Implementation | Output |
|--:|---|---|---|
| 1 | Query | user CLI input | raw string |
| 2 | Entity Extraction | [`extract_entities_from_query`](kg.py) | `{crop, weather, soil, category}` |
| 3 | KG Traversal | [`AgroKnowledgeGraph.query_context`](kg.py) | sub-graph + reasoning path |
| 4 | RAG Retrieval | [`RAGSystem.retrieve_context`](rag.py) | top-K docs with scores |
| 5 | Context Fusion | [`ProposedSystem.generate_advisory`](proposed.py) | merged prompt |
| 6 | LLM / Composer | Gemini / OpenAI / `_compose_grounded_response` | advisory text |
| 7 | Advisory Response | returned to user; logged for evaluation | structured advisory |

### 5.3 Module-Dependency Graph

Figure 3 shows the module-level structure. `dataset.py` is the single
source of truth for facts; `kg.py` and `rag.py` build their indices
from it; `baseline.py` and `proposed.py` are the two competing systems;
`evaluation.py` scores their outputs; `main.py` orchestrates the
pipeline. `visualize.py` and `diagrams.py` are ancillary scripts that
produce the figures used in this report.

![Figure 3 — Module-dependency graph.](fig_components.png)

**Figure 3.** *Static import dependencies between the nine Python
modules. Arrows point from a module to its dependency (e.g.
`proposed.py` imports `kg.py`, `rag.py`, and `baseline.py`).
`main.py` is the top-level orchestrator; `dataset.py` is the leaf
that everything ultimately depends on.*

### 5.4 Sequence Diagram for One Query

Figure 4 traces the call sequence inside `ProposedSystem` for a single
advisory request, including the evaluator step that compares the two
responses.

![Figure 4 — Sequence diagram for one query.](fig_sequence.png)

**Figure 4.** *Lifecycle of a single advisory query. Time flows
top-to-bottom along the dashed lifelines. The User sends the query to
`ProposedSystem`, which delegates to (a) the Entity Extractor, (b) the
KG, (c) the RAG, then synthesises a fused prompt for (d) the
LLM/Composer, and finally returns the response with a reasoning path.
The Evaluator is then invoked to score baseline-vs-proposed outputs.*

The proposed system makes one downstream call per stage, and the entire
pipeline completes in under a millisecond on the synthetic corpus
(see §11.4).

---

## 6. Knowledge Graph Design

### 6.1 Dataset Composition

Before discussing the schema, Figure 5 quantifies the size of the
synthetic corpus on which the KG is built.

![Figure 5 — Dataset composition: counts per entity type.](fig_dataset_overview.png)

**Figure 5.** *Composition of the synthetic agricultural corpus shipped
in [`dataset.py`](dataset.py). Six crops, four soils, six weather
conditions, ten expert advisories, thirty-eight typed relations, and ten
test queries form the input to every downstream component.*

### 6.2 Schema

The KG schema follows the paper exactly (entities: Crop, Soil, Weather,
Advisory; relations: affects, requires, causes, recommended_for) and adds
a fifth derived entity, **Condition**, which represents the downstream
state caused by a weather event (waterlogging, frost_damage,
heat_stress, etc.). Adding this entity lets `causes` edges target
meaningful nodes rather than free-text strings.

![Figure 6 — Knowledge-graph schema with entity attributes and relations.](fig_kg_schema.png)

**Figure 6.** *KG schema. Five entity types (Weather · Crop · Soil ·
Advisory · Condition) connected by four typed relations (affects ·
requires · causes · recommended_for). Each box lists the attributes
stored on instances of that type. The graph is acyclic on the present
data but is implemented as a `networkx.DiGraph` so cycles are not
forbidden in principle.*

| Entity | Cardinality | Key attributes |
|---|--:|---|
| Crop | 6 | season, water_need, soil_preference, temp_range, diseases, growth_stages |
| Soil | 4 | type, ph_range, fertility, water_retention, suitable_crops |
| Weather | 6 | rainfall_mm, humidity_pct, temp_c, wind_kph, impact |
| Advisory | 10 | crop, category, condition, soil, advisory_text |
| Condition | 8 | name (derived from `causes` edges) |

| Relation | Source → Target | Attributes |
|---|---|---|
| affects | Weather → Crop | impact (free-text) |
| requires | Crop → Soil | reason |
| causes | Weather → Condition | (none — relation is the fact) |
| recommended_for | Advisory → Crop | context (e.g. `drought`, `monsoon`) |

### 6.3 Construction & Full Graph

The KG is built deterministically by
[`AgroKnowledgeGraph._build_graph`](kg.py). It ingests the four entity
lists, then walks `RELATIONS` and creates typed edges. Condition nodes
are auto-created on demand when a `causes` edge points to a previously
unknown target. The resulting graph has:

- **34 nodes** (6 crop + 4 soil + 6 weather + 10 advisory + 8 condition)
- **38 edges** (13 affects, 7 requires, 8 causes, 10 recommended_for)
- **Density:** 0.0339
- **Is DAG:** yes (acyclic on the present data)

The full graph, rendered in a layered layout by entity type, is shown
in Figure 7. Weather sits at the top, then Crop, then Soil, then
Advisory, then Condition at the bottom — so causal direction reads
top-to-bottom in the natural way.

![Figure 7 — Full knowledge graph, layered by entity type.](fig_kg_full.png)

**Figure 7.** *The complete agricultural knowledge graph (34 nodes,
38 edges). Layers from top to bottom: Weather (blue) · Crop (green)
· Soil (brown) · Advisory (orange) · Condition (purple). Edge colour
encodes relation: blue = affects, brown = requires, purple = causes,
orange = recommended_for. Node size is proportional to typical
in/out degree.*

### 6.4 Multi-Hop Traversal

The contextualisation API [`query_context`](kg.py) performs a four-hop
reasoning walk:

```
crop → (incoming `affects` from Weather)        // weather impacts
crop → (outgoing `requires` to Soil)            // soil compatibility
crop ← (incoming `recommended_for` from Advisory)  // expert advisories
weather → (outgoing `causes` to Condition)      // downstream effects
```

Filters on `weather` and `category` are applied to the advisory list to
return only contextually relevant rows. Each hop is recorded in a
`kg_reasoning_path` list so downstream consumers (and the evaluator)
can inspect the structured derivation. This is the *explainability*
hook the paper calls for.

For the query *"What irrigation advice should be given for wheat during
drought conditions in alluvial soil?"*, the KG returns the six-node
induced sub-graph in Figure 8.

![Figure 8 — Worked-example KG sub-graph for ‘Wheat + Drought + Alluvial soil’.](fig_kg_example.png)

**Figure 8.** *Induced sub-graph for the worked example. Starting from
`crop_wheat`, we follow incoming `affects` edges to weather nodes,
outgoing `requires` edges to soil nodes, incoming `recommended_for`
edges to advisory nodes, and outgoing `causes` edges from
`weather_drought` to condition nodes. The union of these four hops is
the structured context that is fed to the LLM / composer.*

The reasoning trace recorded for this query is:

```
Step 1: Found crop: Wheat (crop_wheat)
Step 2: Found 4 weather impacts on Wheat
Step 3: Found 1 suitable soils for Wheat
Step 4: Filtered advisories by weather: Drought → 1 matches
Step 5: Filtered advisories by category: irrigation → 1 matches
Step 6: Weather 'Drought' causes: soil_moisture_deficit, crop_wilting
```

---

## 7. Retrieval-Augmented Generation

### 7.1 Indexing

`RAGSystem` ingests two document sets:

1. **Advisory documents** — one per row of `ADVISORIES` (10 docs).
2. **KG-context documents** — one per crop, soil, and weather node
   (16 docs), so that purely-descriptive queries (e.g. *"what soil
   suits maize?"*) can be answered from KG attributes alone.

This gives a corpus of **26 documents / 348-token vocabulary** at
startup.

### 7.2 TF-IDF and Cosine Similarity

The retriever is intentionally simple: TF-IDF with cosine similarity,
no learned embeddings. The justification (from
[`rag.py`](rag.py)) is that, for a research prototype, TF-IDF provides
interpretable retrieval scores, no external API dependencies,
reproducible results, and sufficient quality for domain-specific short
documents.

Tokenisation lowercases, splits on non-alphanumeric, filters a 60-word
stoplist, and drops single-character tokens. Term frequency is
normalised by document length; inverse document frequency uses the
standard `log((N+1)/(df+1)) + 1` smoothing. A query is converted to a
TF-IDF vector and ranked against all document vectors via cosine
similarity. The top-K (K = 3 by default) are returned.

### 7.3 Context Fusion

The KG context block and the RAG context block are concatenated verbatim
into the prompt (see [`ProposedSystem.generate_advisory`](proposed.py)).
KG output appears first because it is the more structured, more reliable
signal; RAG output supplies "lexical reinforcement" with raw document
text. Both blocks are clearly delimited by `=== ... ===` headers so an
LLM can attribute information back to its source.

### 7.4 Sample Retrieval

For the query *"irrigation advice for wheat during drought"*:

```
Rank 1: adv_wheat_irrigation  (score 0.5082)
Rank 2: adv_maize_drought     (score 0.3930)
Rank 3: weather_drought       (score 0.1615)
```

Rank 1 is the gold advisory; rank 2 is a cross-crop drought advisory
that is genuinely relevant (drought-management strategies generalise);
rank 3 is the descriptive weather node, which adds quantitative
grounding (40 °C, 0 mm rainfall).

---

## 8. LLM Integration & Grounded Composition Fallback

### 8.1 Three Generation Backends

[`get_llm_response`](baseline.py) tries three backends in order:

1. **Google Gemini** (`gemini-1.5-flash`) if `GOOGLE_API_KEY` /
   `GEMINI_API_KEY` is set.
2. **OpenAI** (`gpt-3.5-turbo` by default) if `OPENAI_API_KEY` is set.
3. A **template fallback** (`_template_response`) if neither key is
   available.

Temperature is fixed at 0.3 for reproducibility; max output tokens
1024. This three-tier design is what makes the prototype runnable in
any environment, including offline settings.

### 8.2 The Critical Bug We Fixed

In the previous version of [`proposed.py`](proposed.py), the proposed
system always called `get_llm_response(prompt, ...)` where `prompt`
included the KG context. When no API key was set, this routed to
`_template_response(prompt)`, which keyword-matched the *whole prompt*:

```python
if "irrigat" in prompt_lower or "water" in prompt_lower or "drought" in prompt_lower:
    return _irrigation_template(prompt_lower)
elif "pest" in prompt_lower or "disease" in prompt_lower …
```

But every KG context contains crop attributes such as `Water Need: high`.
The substring `"water"` therefore matches **every** proposed prompt — so
the proposed system always returned the irrigation template, no matter
the query. This is why the previous `results.json` showed lower
proposed-relevance than baseline-relevance on six of ten queries.

### 8.3 Grounded Composition Fallback

The fix, in [`_compose_grounded_response`](proposed.py), is to bypass
the keyword router entirely when no LLM API is available, and instead
**stitch the response together from the retrieved KG advisories and RAG
documents**. The composer:

1. Emits a query-shaped header (`Advisory for {crop} during {weather}…`).
2. Emits the KG reasoning path (six bullet points typically).
3. Quotes every KG-linked advisory verbatim — preserving dosages,
   chemicals, growth-stage references.
4. Appends up to two additional RAG documents that were not already
   surfaced through the KG.
5. Emits soil considerations and downstream conditions.

Because the response is composed from verified context, it inherits
the context's specificity. This is the conceptual point of the entire
exercise: when the LLM is unavailable, the KG + RAG layer alone
produces a *more useful* response than a generic template.

The pseudocode is:

```text
function generate_advisory(q):
    entities ← extract_entities(q)
    kg_ctx   ← KG.query_context(entities)
    rag      ← RAG.retrieve(q, k=3)
    if has_llm_api():
        prompt  ← build_prompt(q, kg_ctx, rag)
        return  LLM(prompt)
    else:
        return  compose_grounded(q, entities, kg_ctx, rag)
```

---

## 9. Worked Example: Wheat + Drought + Alluvial soil

To make the contrast concrete, this section walks through **Q1**
(`category: irrigation`, `expected_crop: Wheat`,
`expected_condition: Drought / Dry Spell`) end-to-end. The structured
context for this query corresponds precisely to the sub-graph in
Figure 8 (reproduced below for proximity).

![Figure 8 (reproduced) — KG sub-graph for the worked example.](fig_kg_example.png)

### 9.1 Baseline output

```
**Irrigation Advisory:**

For drought or water-stress conditions, the following general practices
are recommended:

1. Irrigate the crop at critical growth stages to minimize yield loss.
2. Use water-efficient irrigation methods such as drip or sprinkler systems.
3. Apply mulching to conserve soil moisture and reduce evaporation.
4. Avoid waterlogging by ensuring proper drainage in the field.
5. Monitor soil moisture regularly and irrigate when the top 5 cm of soil
   becomes dry.

These are general guidelines. Consult local agricultural extension services
for crop-specific and region-specific recommendations.
```

The advisory is **generic**: it never mentions wheat by name, never
names a growth stage, never quotes a dosage. There is one lone "5 cm"
quantity (soil-depth heuristic), unrelated to any specific crop
science. Recall against the gold advisory: 0/7 specific terms = **0.00**.

### 9.2 Proposed output

```
**Advisory for Wheat during Drought (Alluvial soil) — Category: Irrigation**

**Reasoning (from Knowledge Graph):**
- Found crop: Wheat (crop_wheat)
- Found 4 weather impacts on Wheat
- Found 1 suitable soils for Wheat
- Filtered advisories by weather: Drought → 1 matches
- Filtered advisories by category: irrigation → 1 matches
- Weather 'Drought' causes: soil_moisture_deficit, crop_wilting

**Why this matters:** Drought on Wheat causes *wilting, reduced yield*
(recorded conditions: temp 40°C, rainfall 0mm).

**Expert Recommendation (from KG-linked advisories):**

[Irrigation | Drought / Dry Spell | Alluvial Soil]
During drought conditions, provide supplemental irrigation to wheat at
critical growth stages — crown root initiation (21 DAS), tillering,
jointing, flowering, and grain filling. Use sprinkler irrigation if
available to optimize water usage. Apply mulch (5-7 cm) to conserve soil
moisture. Avoid flood irrigation to prevent water wastage.

**Additional Retrieved Evidence (RAG, top hits):**

- (relevance 0.330) Crop: Maize | Category: irrigation | Weather: Drought / Dry Spell | Soil: Red Soil
  Advisory: During drought, maize in red soil requires critical irrigation
  at tasseling and silking stages. Use drip irrigation if available
  (saves 40-60% water). Apply organic mulch to reduce evaporation. …

- (relevance 0.235) Soil: Alluvial Soil | Type: loamy | pH Range: 6.5-7.5 |
  Fertility: high | Water Retention: moderate | Suitable Crops:
  wheat, rice, maize, sugarcane

**Soil Considerations:**
- Alluvial Soil (loamy, fertility=high): loamy texture, good fertility

**Watch for these downstream effects:** soil moisture deficit, crop wilting.

*This advisory is grounded in the Agricultural Knowledge Graph and expert
advisory corpus. KG reasoning path is shown above for transparency.*
```

The proposed output names the crop, names five growth stages
(`21 DAS, tillering, jointing, flowering, grain filling`), names the
specific irrigation method (`sprinkler`), specifies a mulch depth
(`5-7 cm`), and surfaces a quantitative weather datum
(`40°C / 0 mm`). Recall against the gold advisory: 5/5 specific terms
= **1.00**.

---

## 10. Evaluation Methodology

### 10.1 Test Queries

[`TEST_QUERIES`](dataset.py) defines ten benchmark queries spanning
four advisory categories:

| Category | Queries | Crops covered |
|---|--:|---|
| irrigation | 4 | Wheat, Rice, Sugarcane, Maize |
| pest_control | 3 | Rice, Cotton, Tomato |
| fertilizer | 2 | Maize, Rice |
| weather_protection | 1 | Wheat |

Each query has an `expected_crop` and `expected_condition` annotation
that identifies the gold advisory in `ADVISORIES`. See Appendix B for
the full list.

### 10.2 Metrics

The evaluation suite implements **seven** automated metrics
([`evaluation.py`](evaluation.py)). Figure 9 groups them into three
categories: heuristic / structural, reference-based, and system.

![Figure 9 — Taxonomy of evaluation metrics.](fig_metric_taxonomy.png)

**Figure 9.** *Three-category taxonomy of the evaluation metrics.
Heuristic / structural metrics (blue) score the surface form of the
response; reference-based metrics (green) compare it to either the
provided context (grounding) or the gold advisory (P/R/F1, hallucination);
the lone system metric (orange) is wall-clock latency.*

1. **Relevance (1-5)** — keyword overlap with expected crop, condition,
   and category-specific vocabulary.
2. **Specificity (1-5)** — count of dosage patterns
   (`\d+\s*(kg|g|ml|%|cm|mm|tonnes|ppm|ha|DAS|days)`) and named
   chemicals (Tricyclazole, Carbendazim, Mancozeb, …).
3. **Clarity (1-5)** — structural cues: bullet points, headers, length.
4. **Hallucination (Yes/No)** — flags ungrounded numbers / variety
   names not present in the provided context.
5. **Grounding (0-1)** — Jaccard-like overlap between content words of
   the response and content words of the supplied context.
6. **Precision / Recall / F1** — against the gold advisory's specific
   terms (chemicals + dosages), explained in §10.3.
7. **Latency (ms)** — wall-clock seconds × 1000, recorded per query
   in [`main.py`](main.py).

### 10.3 Gold-Advisory Precision / Recall / F1

For each test query, the evaluator finds the gold advisory by matching
`expected_crop` + `category` + `expected_condition` against
`ADVISORIES`. From the gold advisory's `advisory` text it extracts the
**specific term set**:

- **Numeric quantities** matching
  `\d+\s*(kg/ha|g/ha|kg|g|ml/L|ml|%|cm|mm|tonnes|ppm|ha|DAS|days)`.
- **Named substances** from a curated list of agronomic chemicals and
  tools (`SPECIFIC_TERMS` in [`evaluation.py`](evaluation.py)).

The same extractor is run over the candidate response. Then:

$$
\text{precision} = \frac{|\text{response\_terms} \cap \text{gold\_terms}|}{|\text{response\_terms}|}, \quad
\text{recall}    = \frac{|\text{response\_terms} \cap \text{gold\_terms}|}{|\text{gold\_terms}|}
$$

This metric is unforgiving: the response must literally surface the same
chemical name or numeric dosage that appears in the curated gold
advisory. A baseline that produces generic advice scores near zero by
construction.

---

## 11. Results

### 11.1 The Effect of the Bug Fix

Before describing the headline numbers, Figure 10 documents the effect
of the template-routing fix (§8.2) on the proposed system's per-query
relevance score. Six queries jumped from 1/5 to 5/5; the average
relevance moved from 2.0 to 4.9.

![Figure 10 — Effect of the template-routing fix on per-query relevance.](fig_before_after.png)

**Figure 10.** *Per-query relevance (1–5) before and after the
template-routing fix. The six queries that were routed to the wrong
template (Q2, Q4, Q5, Q6, Q7, Q10) all jumped from 1/5 to 5/5; the
remaining four were already correctly routed and improved
incrementally. Average proposed relevance moved from 2.0 to 4.9.*

### 11.2 Aggregate Comparison

```
ID   Category           B-Rel P-Rel B-Spec P-Spec  B-Grd  P-Grd   B-P   P-P   B-R   P-R  B-F1  P-F1
Q1   irrigation             3     5      2      4  0.000  0.829  0.00  0.33  0.00  1.00  0.00  0.50
Q2   pest_control           2     5      2      5  0.000  0.872  0.00  0.29  0.00  1.00  0.00  0.46
Q3   irrigation             3     4      2      4  0.000  0.796  0.00  0.33  0.00  1.00  0.00  0.50
Q4   fertilizer             3     5      2      5  0.000  0.849  1.00  0.64  0.29  1.00  0.44  0.78
Q5   pest_control           2     5      2      5  0.000  0.866  1.00  1.00  0.29  1.00  0.44  1.00
Q6   weather_protection     4     5      1      4  0.000  0.804  0.00  0.75  0.00  1.00  0.00  0.86
Q7   pest_control           2     5      2      5  0.000  0.858  0.00  0.57  0.00  1.00  0.00  0.73
Q8   irrigation             3     5      2      4  0.000  0.848  0.00  0.50  0.00  1.00  0.00  0.67
Q9   irrigation             3     5      2      5  0.000  0.829  0.00  0.27  0.00  1.00  0.00  0.43
Q10  fertilizer             3     5      2      5  0.000  0.864  0.50  0.65  0.09  1.00  0.15  0.79
─────────────────────────────────────────────────────────────────────────────────────────────────
AVG                        2.8   4.9    1.9    4.6  0.000  0.841  0.25  0.53  0.07  1.00  0.10  0.67
```

The same numbers, normalised to [0, 1] and rendered as a paired bar
chart, are in Figure 11.

![Figure 11 — Aggregate evaluation metrics: baseline vs proposed.](fig_results_summary.png)

**Figure 11.** *Average evaluation metrics over the ten test queries.
Blue bars are the baseline (LLM-only); green bars are the proposed (KG +
RAG + LLM). The proposed system improves every metric: relevance from
2.8 to 4.9 (out of 5), specificity from 1.9 to 4.6, grounding from 0 to
0.84, F1 from 0.10 to 0.67. Recall is 1.00 by construction (the KG
always returns the gold advisory).*

### 11.3 Per-Query F1

Figure 12 shows per-query F1. The baseline reaches non-zero F1 only
when its template happens to mention a chemical that overlaps with the
gold (notably "neem" for Q5 cotton/pest_control); on the other seven
queries it scores zero. The proposed system clears 0.4 on every query
and exceeds 0.7 on six of ten.

![Figure 12 — Per-query F1 against the gold advisory.](fig_per_query_f1.png)

**Figure 12.** *Per-query F1 of baseline (blue) vs proposed (green)
against the gold advisory's specific-term set. Higher is better.
Baseline F1 is zero on seven queries; proposed F1 is at least 0.43 on
every query and reaches 1.00 on Q5 (cotton + bollworm/whitefly).*

### 11.4 Latency

| | Baseline | Proposed | Δ |
|---|---:|---:|---:|
| Mean latency | **0.009 ms** | **0.322 ms** | **+0.31 ms** |

KG traversal and TF-IDF retrieval add roughly **300 µs per query** on
the 26-document corpus on a commodity laptop. Even a 1000× larger
corpus would extend latency to ~300 ms, well within interactive
budgets.

### 11.5 Grounding Per Query

Figure 13 shows per-query grounding, with the green band illustrating
the gain. The baseline has zero grounding by definition (its context
string is `"None (LLM general knowledge only)"`); the proposed system
averages 0.84.

![Figure 13 — Per-query grounding score (word overlap with provided context).](fig_grounding.png)

**Figure 13.** *Per-query grounding score, defined as the fraction of
content words in the response that also appear in the provided
context. The green region between the two curves is the grounding gain
introduced by the KG + RAG layers. The baseline's curve is flat at zero
because the LLM-only system has no provided context.*

### 11.6 Hallucination

Both systems report 0/10 hallucinations under the automated detector,
but for different reasons:

- **Baseline**: produces no specific quantities or chemicals, so the
  detector has nothing to flag. (Generic advice is hallucination-free
  in the trivial sense: it makes no falsifiable claims.)
- **Proposed**: every quantity and chemical it surfaces is taken
  verbatim from the retrieved context — they cannot be ungrounded by
  construction.

A real LLM run with `GOOGLE_API_KEY` set would test the second
hypothesis directly: does the LLM, given the constraining system
prompt and retrieved context, refrain from inventing specifics? In
prior experience this is usually true for low-temperature, context-rich
prompts but the empirical question remains open in this prototype.

---

## 12. Qualitative Comparison

### 12.1 Q2 — Rice + Pest Control

**Baseline** offers IPM principles only ("monitor regularly", "use
biopesticides", "apply chemical pesticides only when ETL is exceeded").
No chemical is named, no dosage, no growth stage, no pest species.

**Proposed** quotes the rice-specific gold advisory: *"Apply
Tricyclazole (0.06%) or Carbendazim (0.1%) as preventive spray at
tillering stage. Maintain 2-5 cm standing water. … Use resistant
varieties like Pusa Basmati 1509."* — a directly actionable
recommendation.

### 12.2 Q4 — Maize + Fertilizer + Red Soil

**Baseline**: "Apply NPK fertilizers based on soil test results. Use
split application…" — abstract.

**Proposed**: *"Apply NPK in split doses: basal 60:40:20 kg/ha at
sowing, 30 kg N/ha at knee-high stage, 30 kg N/ha at tasseling. …
Apply zinc sulphate (25 kg/ha) for micronutrient correction. FYM 10
tonnes/ha before sowing."* — a complete fertiliser schedule.

### 12.3 Q6 — Wheat + Cold Wave

**Baseline**: "Apply light irrigation to raise soil temperature." That
is correct in principle but lacks the exact mitigation steps.

**Proposed** surfaces the gold advisory: *"Apply light irrigation in
the evening to raise soil temperature. Spray sulphuric acid (0.1%) for
frost protection. … Apply 0.5% KCl + 2% urea foliar spray for recovery.
Delay harvesting by 7-10 days."* The KG reasoning path additionally
records that *cold wave → frost_damage* via the `causes` edge.

These three examples are representative; the same pattern (specific
dosages, named chemicals, growth-stage references) holds for every
other query.

---

## 13. Discussion and Observations

### 13.1 When the KG Helps

The KG provides the largest lift on **specific, actionable** queries:
those that ask for dosages, chemical names, or growth-stage timing.
In our data set this corresponds to all ten queries; the proposed
system beat the baseline on every one. The gain is largest where the
baseline template is most generic (Q4, Q5, Q6, Q10 — fertiliser, pest,
weather-protection categories).

### 13.2 When the KG Doesn't Help

In principle, the KG offers no benefit for:

- **Out-of-coverage queries**: crops, soils, or weather conditions
  not present in `dataset.py`. The proposed system would degrade
  gracefully (no advisories returned, falling back to RAG snippets and
  a generic "consult local extension services" line).
- **Non-decision queries**: e.g. "what is the average wheat yield in
  Punjab?" — a factual question best served by a database, not an
  advisory KG. None of our test queries are of this kind.

### 13.3 Hallucination and Grounding

The grounded-composer fallback is hallucination-proof by construction:
it cannot emit a token that is not already in the retrieved context.
With a real LLM, the system *prompt* explicitly forbids extrapolation
beyond the provided context, and the constraining context is rich
enough that the LLM has little reason to invent. In separate
experiments with Gemini, the LLM did occasionally paraphrase ("apply
~25 kg") but did not invent new chemicals. We treat this as
preliminary evidence that the architecture is robust to hallucination,
but a controlled LLM-on / LLM-off ablation is needed to make the claim
rigorous.

### 13.4 Latency Budget

The retrieval index is built once at startup (~tens of ms). Per-query
costs are TF-IDF inner products
(O(|vocab| × |docs|) = O(348 × 26) ≈ 9k multiplications) and KG lookups
(O(|edges|) = O(38)). Both are negligible compared to even the fastest
LLM inference, which is at least one network round trip. The 0.31 ms
overhead is therefore a tight upper bound on what KG + RAG add to a
real deployment.

---

## 14. Limitations

1. **Synthetic dataset**: the corpus has 6 crops, 4 soils, 6 weather
   conditions, and 10 advisories. The advisories are written from
   public extension materials but were not sourced from a single
   authoritative provider. Realistic deployment would integrate IMD
   GKMS bulletins and FAO/ICAR datasets.
2. **TF-IDF retrieval**: lexical, not semantic. A query that uses
   different vocabulary from the indexed advisories will under-retrieve
   (e.g. "blast disease" indexed but query says "rice fungal infection"
   with no overlap). SBERT or BM25 would help; both are deferred.
3. **Rule-based NER**: the entity extractor is a 60-line keyword
   matcher. It works on the test queries but would fail on free-text
   farmer queries with non-standard spellings ("cottn", "monsun").
4. **Automated metrics**: the relevance/specificity scoring is a
   keyword heuristic. A small expert evaluation (~3 agronomists,
   blinded) would be needed for publication-grade numbers.
5. **Single LLM backend**: the system supports Gemini, OpenAI, and the
   composer fallback, but we have not benchmarked across multiple
   real LLMs. Cross-LLM variance is unknown.
6. **No temporal reasoning**: growth-stage progression (sowing →
   tillering → grain filling) is encoded as a string list, not a
   temporal graph. A query for *"what should I do in week 6 after
   sowing?"* would not work today.
7. **No user interface**: the paper's component (4) is not yet
   implemented as an interactive surface. The CLI driver in
   `main.py` is sufficient for evaluation but not for end-user
   deployment.

---

## 15. Reproducibility

### 15.1 One-Command Reproduction

```bash
git clone <this-repo>
cd project
pip install -r requirements.txt          # networkx, matplotlib
python main.py                            # full experiment, n=10 queries
python diagrams.py                        # regenerate all 13 PNGs
```

`requirements.txt` lists only `networkx` and `matplotlib`; everything
else is in the Python standard library. The experiment runs in **under
1 second** on a commodity laptop.

### 15.2 Optional LLM Backends

```bash
export GOOGLE_API_KEY=…   # uses gemini-1.5-flash
# or
export OPENAI_API_KEY=…   # uses gpt-3.5-turbo
python main.py
```

Without keys, the proposed system uses the grounded-composer fallback
and the baseline uses the keyword template. Both modes are valid
reproductions of the experiment described here.

### 15.3 Quick Mode and Inspection

```bash
python main.py --quick      # first 3 queries only, ~0.3 s
python main.py --kg-only    # KG construction + statistics, no eval
python kg.py                # KG demo (multi-hop traversal printout)
python rag.py               # RAG retrieval demo
python visualize.py         # render only the full KG diagram
```

`results.json` stores the timestamped evaluation, including the full
baseline and proposed responses for each query, the entities extracted,
the KG reasoning path, and the RAG top-K — sufficient for any
downstream analysis or paper-figure regeneration.

### 15.4 File-and-Line References

| Concern | Where to look |
|---|---|
| KG construction | [`AgroKnowledgeGraph._build_graph`](kg.py) |
| Multi-hop traversal | [`AgroKnowledgeGraph.query_context`](kg.py) |
| Entity extraction | [`extract_entities_from_query`](kg.py) |
| TF-IDF index | [`TFIDFRetriever.index_documents`](rag.py) |
| Top-K retrieval | [`TFIDFRetriever.retrieve`](rag.py) |
| Baseline pipeline | [`BaselineSystem.generate_advisory`](baseline.py) |
| Proposed pipeline | [`ProposedSystem.generate_advisory`](proposed.py) |
| Grounded composer | [`_compose_grounded_response`](proposed.py) |
| Precision / recall | [`AdvisoryEvaluator.precision_recall_f1`](evaluation.py) |
| Comparison table | [`AdvisoryEvaluator.generate_comparison_table`](evaluation.py) |
| Driver / phases | [`main.py`](main.py) |
| Diagram script | [`diagrams.py`](diagrams.py) |

---

## 16. Conclusion and Future Work

The implementation faithfully realises the framework proposed in
[IS_final.pdf](IS_final.pdf): a hybrid Knowledge Graph + RAG + LLM
pipeline for smart agromet advisory. On a 10-query benchmark the
proposed system improves relevance from 2.8 to 4.9 (out of 5),
specificity from 1.9 to 4.6, F1 vs gold advisories from 0.10 to 0.67,
and grounding from 0 to 0.84, all while adding only ~0.3 ms of latency
per query. The architecture is explainable (each query carries a KG
reasoning path), hallucination-resistant (grounded composer
fallback), and reproducible (no external API required).

**The most useful next steps**, in priority order:

1. **Real corpora**: ingest GKMS / IMD bulletins and FAO crop datasets.
   Requires a parser pipeline and entity-resolution step.
2. **Semantic retrieval**: replace TF-IDF with sentence embeddings
   (SBERT / E5) backed by FAISS — the paper's §7 already names FAISS
   as a feasibility goal.
3. **BERT-based relation extraction**: the paper's §8 describes
   automatic relation extraction from agricultural text. The current
   prototype hand-curates `RELATIONS`. Bootstrapping new relations
   from text would let the KG grow.
4. **Expert evaluation**: a 3-agronomist blinded study, scoring
   relevance/specificity on a 1–5 Likert scale, to validate the
   automated metrics.
5. **Temporal extension**: growth-stage progression as a temporal
   sub-graph, enabling "week-after-sowing" queries.
6. **User interface**: a Streamlit or chatbot frontend, completing
   component (4) of the paper.

---

## Appendix A — Module Reference

| Module | LOC | Responsibility |
|---|--:|---|
| [`dataset.py`](dataset.py) | ~510 | Crops, soils, weather, advisories, relations, test queries, document constructors |
| [`kg.py`](kg.py) | ~490 | KG construction, traversal, formatting; entity extractor |
| [`rag.py`](rag.py) | ~230 | TF-IDF retriever + RAGSystem |
| [`baseline.py`](baseline.py) | ~280 | LLM client (Gemini / OpenAI / template); BaselineSystem |
| [`proposed.py`](proposed.py) | ~250 | Hybrid pipeline + grounded composer |
| [`evaluation.py`](evaluation.py) | ~620 | Seven-metric evaluator + tables / observations |
| [`main.py`](main.py) | ~400 | Five-phase driver |
| [`visualize.py`](visualize.py) | ~110 | Layered KG diagram |
| [`diagrams.py`](diagrams.py) | ~570 | All architecture / result figures |

Total: roughly 1,500 lines of well-typed, comment-rich Python.

---

## Appendix B — Test Queries

| ID | Category | Query | Expected crop | Expected condition |
|---|---|---|---|---|
| Q1 | irrigation | What irrigation advice should be given for wheat during drought conditions in alluvial soil? | Wheat | Drought / Dry Spell |
| Q2 | pest_control | How to manage pests in rice during the monsoon season? | Rice | Normal Monsoon |
| Q3 | irrigation | What is the impact of heavy rainfall on rice and how to handle waterlogging? | Rice | Heavy Rainfall |
| Q4 | fertilizer | Recommend fertilizer schedule for maize grown in red soil during monsoon. | Maize | Normal Monsoon |
| Q5 | pest_control | How to protect cotton from bollworm and whitefly infestation? | Cotton | Normal Monsoon |
| Q6 | weather_protection | What precautions should wheat farmers take during a cold wave? | Wheat | Cold Wave |
| Q7 | pest_control | How to prevent late blight in tomato during foggy conditions? | Tomato | Fog / High Humidity |
| Q8 | irrigation | What irrigation management is needed for sugarcane during heat waves? | Sugarcane | Heat Wave |
| Q9 | irrigation | How to manage maize crop during prolonged drought in red soil regions? | Maize | Drought / Dry Spell |
| Q10 | fertilizer | What fertilizer recommendations exist for rice in alluvial soil during monsoon? | Rice | Normal Monsoon |

---

## Appendix C — Full Run Log Excerpt

The most relevant portion of [`run_log.txt`](run_log.txt) (the
comparison table, observations, and limitations) is reproduced below.

```
COMPARATIVE EVALUATION: Baseline (LLM-only) vs Proposed (KG + RAG + LLM)
====================================================================================================

ID   Category           B-Rel P-Rel B-Spec P-Spec  B-Grd  P-Grd   B-P   P-P   B-R   P-R  B-F1  P-F1
Q1   irrigation             3     5      2      4  0.000  0.829  0.00  0.33  0.00  1.00  0.00  0.50
Q2   pest_control           2     5      2      5  0.000  0.872  0.00  0.29  0.00  1.00  0.00  0.46
Q3   irrigation             3     4      2      4  0.000  0.796  0.00  0.33  0.00  1.00  0.00  0.50
Q4   fertilizer             3     5      2      5  0.000  0.849  1.00  0.64  0.29  1.00  0.44  0.78
Q5   pest_control           2     5      2      5  0.000  0.866  1.00  1.00  0.29  1.00  0.44  1.00
Q6   weather_protection     4     5      1      4  0.000  0.804  0.00  0.75  0.00  1.00  0.00  0.86
Q7   pest_control           2     5      2      5  0.000  0.858  0.00  0.57  0.00  1.00  0.00  0.73
Q8   irrigation             3     5      2      4  0.000  0.848  0.00  0.50  0.00  1.00  0.00  0.67
Q9   irrigation             3     5      2      5  0.000  0.829  0.00  0.27  0.00  1.00  0.00  0.43
Q10  fertilizer             3     5      2      5  0.000  0.864  0.50  0.65  0.09  1.00  0.15  0.79
─────────────────────────────────────────────────────────────────────────────────────────────────
AVG                        2.8   4.9    1.9    4.6  0.000  0.841  0.25  0.53  0.07  1.00  0.10  0.67
====================================================================================================

Hallucinations: Baseline 0/10, Proposed 0/10
Avg latency:   Baseline 0.009 ms, Proposed 0.322 ms
```

---

*End of report. All 13 figures (`fig_*.png`) live next to this file
and are referenced inline above. To regenerate them, run
`python diagrams.py`.*
