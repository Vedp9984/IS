"""
kg.py — Knowledge Graph Module
================================
Builds and queries a domain-specific agricultural knowledge graph using NetworkX.

Node types: crop, soil, weather, advisory, condition
Edge types: affects, requires, causes, recommended_for

Key capabilities:
  - Graph construction from dataset
  - Multi-hop traversal for context retrieval
  - Subgraph extraction for specific queries
  - Graph statistics and visualization
"""

import networkx as nx
import json
from typing import List, Dict, Tuple, Optional
from dataset import CROPS, SOILS, WEATHER_CONDITIONS, ADVISORIES, RELATIONS


class AgroKnowledgeGraph:
    """Agricultural Knowledge Graph built on NetworkX."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """Construct the KG from the dataset."""
        # ── Add Crop Nodes ──
        for crop in CROPS:
            self.graph.add_node(
                crop["id"],
                type="crop",
                name=crop["name"],
                season=crop["season"],
                water_need=crop["water_need"],
                soil_preference=crop["soil_preference"],
                temp_range=crop["temp_range_c"],
                diseases=crop["diseases"],
                growth_stages=crop["growth_stages"],
            )

        # ── Add Soil Nodes ──
        for soil in SOILS:
            self.graph.add_node(
                soil["id"],
                type="soil",
                name=soil["name"],
                soil_type=soil["type"],
                ph_range=soil["ph_range"],
                fertility=soil["fertility"],
                water_retention=soil["water_retention"],
                suitable_crops=soil["suitable_crops"],
            )

        # ── Add Weather Nodes ──
        for w in WEATHER_CONDITIONS:
            self.graph.add_node(
                w["id"],
                type="weather",
                condition=w["condition"],
                rainfall_mm=w["rainfall_mm"],
                humidity_pct=w["humidity_pct"],
                temp_c=w["temp_c"],
                wind_kph=w["wind_kph"],
                impact=w["impact"],
            )

        # ── Add Advisory Nodes ──
        for adv in ADVISORIES:
            self.graph.add_node(
                adv["id"],
                type="advisory",
                crop=adv["crop"],
                category=adv["category"],
                condition=adv["condition"],
                soil=adv["soil"],
                advisory_text=adv["advisory"],
            )

        # ── Add Edges ──
        for src, rel, tgt, attrs in RELATIONS:
            # For "causes" relations, target is a string condition, add as node if not present
            if rel == "causes" and tgt not in self.graph:
                self.graph.add_node(tgt, type="condition", name=tgt)
            self.graph.add_edge(src, tgt, relation=rel, **attrs)

    # ──────────────────── Query Methods ────────────────────

    def get_node_info(self, node_id: str) -> Optional[Dict]:
        """Get all attributes of a node."""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def get_neighbors(self, node_id: str, relation: str = None) -> List[Tuple[str, Dict]]:
        """Get neighbors of a node, optionally filtered by relation type."""
        neighbors = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation:
                target_data = dict(self.graph.nodes[target])
                neighbors.append((target, {**target_data, "edge_data": data}))
        return neighbors

    def get_incoming(self, node_id: str, relation: str = None) -> List[Tuple[str, Dict]]:
        """Get incoming edges to a node."""
        incoming = []
        for source, _, data in self.graph.in_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation:
                source_data = dict(self.graph.nodes[source])
                incoming.append((source, {**source_data, "edge_data": data}))
        return incoming

    def find_crop_node(self, crop_name: str) -> Optional[str]:
        """Find crop node ID by name (case-insensitive)."""
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "crop" and data.get("name", "").lower() == crop_name.lower():
                return node_id
        return None

    def find_weather_node(self, condition: str) -> Optional[str]:
        """Find weather node by condition keyword."""
        condition_lower = condition.lower()
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "weather":
                if condition_lower in data.get("condition", "").lower():
                    return node_id
        return None

    def find_soil_node(self, soil_name: str) -> Optional[str]:
        """Find soil node by name keyword."""
        soil_lower = soil_name.lower()
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "soil":
                if soil_lower in data.get("name", "").lower():
                    return node_id
        return None

    def get_advisories_for_crop(self, crop_name: str) -> List[Dict]:
        """Get all advisories linked to a crop via 'recommended_for' relation."""
        crop_id = self.find_crop_node(crop_name)
        if not crop_id:
            return []

        advisories = []
        incoming = self.get_incoming(crop_id, relation="recommended_for")
        for src_id, src_data in incoming:
            if src_data.get("type") == "advisory":
                advisories.append({
                    "id": src_id,
                    "advisory_text": src_data.get("advisory_text", ""),
                    "category": src_data.get("category", ""),
                    "condition": src_data.get("condition", ""),
                    "soil": src_data.get("soil", ""),
                    "edge_context": src_data.get("edge_data", {}),
                })
        return advisories

    def get_weather_impacts_on_crop(self, crop_name: str) -> List[Dict]:
        """Get weather conditions that affect a specific crop."""
        crop_id = self.find_crop_node(crop_name)
        if not crop_id:
            return []

        impacts = []
        incoming = self.get_incoming(crop_id, relation="affects")
        for src_id, src_data in incoming:
            if src_data.get("type") == "weather":
                impacts.append({
                    "weather_id": src_id,
                    "condition": src_data.get("condition", ""),
                    "impact": src_data.get("edge_data", {}).get("impact", ""),
                    "temp_c": src_data.get("temp_c"),
                    "rainfall_mm": src_data.get("rainfall_mm"),
                })
        return impacts

    def get_soil_for_crop(self, crop_name: str) -> List[Dict]:
        """Get soil types suitable for a crop."""
        crop_id = self.find_crop_node(crop_name)
        if not crop_id:
            return []

        soils = []
        neighbors = self.get_neighbors(crop_id, relation="requires")
        for tgt_id, tgt_data in neighbors:
            if tgt_data.get("type") == "soil":
                soils.append({
                    "soil_id": tgt_id,
                    "name": tgt_data.get("name", ""),
                    "soil_type": tgt_data.get("soil_type", ""),
                    "fertility": tgt_data.get("fertility", ""),
                    "reason": tgt_data.get("edge_data", {}).get("reason", ""),
                })
        return soils

    def query_context(self, crop_name: str = None, weather: str = None,
                      soil: str = None, category: str = None) -> Dict:
        """
        Multi-hop KG query: gather comprehensive context for a scenario.
        This is the key method that provides structured reasoning context.

        Returns a dictionary with:
          - crop_info: details about the crop
          - weather_impacts: how weather affects the crop
          - soil_info: soil compatibility
          - relevant_advisories: matching advisories
          - kg_reasoning_path: the traversal path (for explainability)
        """
        context = {
            "crop_info": None,
            "weather_impacts": [],
            "soil_info": [],
            "relevant_advisories": [],
            "related_conditions": [],
            "kg_reasoning_path": [],
        }

        # Step 1: Find crop
        if crop_name:
            crop_id = self.find_crop_node(crop_name)
            if crop_id:
                context["crop_info"] = self.get_node_info(crop_id)
                context["kg_reasoning_path"].append(f"Found crop: {crop_name} ({crop_id})")

                # Step 2: Get weather impacts
                impacts = self.get_weather_impacts_on_crop(crop_name)
                context["weather_impacts"] = impacts
                context["kg_reasoning_path"].append(
                    f"Found {len(impacts)} weather impacts on {crop_name}"
                )

                # Step 3: Get soil requirements
                soils = self.get_soil_for_crop(crop_name)
                context["soil_info"] = soils
                context["kg_reasoning_path"].append(
                    f"Found {len(soils)} suitable soils for {crop_name}"
                )

                # Step 4: Get advisories
                advisories = self.get_advisories_for_crop(crop_name)

                # Filter by weather condition if specified
                if weather:
                    weather_lower = weather.lower()
                    filtered = [a for a in advisories
                                if weather_lower in a["condition"].lower()]
                    if filtered:
                        advisories = filtered
                        context["kg_reasoning_path"].append(
                            f"Filtered advisories by weather: {weather} → {len(filtered)} matches"
                        )

                # Filter by category if specified
                if category:
                    cat_lower = category.lower()
                    filtered = [a for a in advisories
                                if cat_lower in a["category"].lower()]
                    if filtered:
                        advisories = filtered
                        context["kg_reasoning_path"].append(
                            f"Filtered advisories by category: {category} → {len(filtered)} matches"
                        )

                context["relevant_advisories"] = advisories

        # Step 5: Get weather-caused conditions
        if weather:
            weather_id = self.find_weather_node(weather)
            if weather_id:
                caused = self.get_neighbors(weather_id, relation="causes")
                context["related_conditions"] = [
                    {"condition": tgt_data.get("name", tgt_id)}
                    for tgt_id, tgt_data in caused
                ]
                context["kg_reasoning_path"].append(
                    f"Weather '{weather}' causes: "
                    + ", ".join(c["condition"] for c in context["related_conditions"])
                )

        return context

    def format_context_for_llm(self, context: Dict) -> str:
        """Format KG query results into a structured text prompt for the LLM."""
        parts = []

        if context["crop_info"]:
            ci = context["crop_info"]
            parts.append(
                f"=== CROP INFORMATION (from Knowledge Graph) ===\n"
                f"Crop: {ci.get('name')} | Season: {ci.get('season')} | "
                f"Water Need: {ci.get('water_need')}\n"
                f"Temperature Range: {ci.get('temp_range', 'N/A')}°C\n"
                f"Common Diseases: {', '.join(ci.get('diseases', []))}\n"
                f"Growth Stages: {', '.join(ci.get('growth_stages', []))}"
            )

        if context["weather_impacts"]:
            parts.append("\n=== WEATHER IMPACTS (from Knowledge Graph) ===")
            for wi in context["weather_impacts"]:
                parts.append(
                    f"- {wi['condition']}: {wi['impact']} "
                    f"(Temp: {wi.get('temp_c')}°C, Rain: {wi.get('rainfall_mm')}mm)"
                )

        if context["soil_info"]:
            parts.append("\n=== SOIL INFORMATION (from Knowledge Graph) ===")
            for si in context["soil_info"]:
                parts.append(
                    f"- {si['name']} ({si['soil_type']}): "
                    f"Fertility={si['fertility']}, Reason={si['reason']}"
                )

        if context["related_conditions"]:
            parts.append("\n=== CAUSED CONDITIONS (from Knowledge Graph) ===")
            for rc in context["related_conditions"]:
                parts.append(f"- {rc['condition']}")

        if context["relevant_advisories"]:
            parts.append("\n=== RELEVANT EXPERT ADVISORIES (from Knowledge Graph) ===")
            for adv in context["relevant_advisories"]:
                parts.append(
                    f"\n[{adv['category'].upper()}] ({adv['condition']} | {adv['soil']})\n"
                    f"{adv['advisory_text']}"
                )

        if context["kg_reasoning_path"]:
            parts.append("\n=== KG REASONING PATH ===")
            for i, step in enumerate(context["kg_reasoning_path"], 1):
                parts.append(f"  Step {i}: {step}")

        return "\n".join(parts)

    # ──────────────────── Statistics ────────────────────

    def get_stats(self) -> Dict:
        """Return graph statistics."""
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            r = data.get("relation", "unknown")
            edge_types[r] = edge_types.get(r, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "density": nx.density(self.graph),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
        }

    def export_for_visualization(self) -> Dict:
        """Export graph data for visualization / paper diagrams."""
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": data.get("name", data.get("condition", node_id)),
                "type": data.get("type", "unknown"),
            })

        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            edges.append({
                "source": src,
                "target": tgt,
                "relation": data.get("relation", ""),
            })

        return {"nodes": nodes, "edges": edges}


# ──────────────────── Utility Functions ────────────────────

def extract_entities_from_query(query: str) -> Dict:
    """
    Simple entity extraction from natural language query.
    Returns crop, weather, soil, and category mentions.
    """
    query_lower = query.lower()

    # Crop detection
    crop_keywords = {
        "wheat": "Wheat", "rice": "Rice", "maize": "Maize",
        "cotton": "Cotton", "tomato": "Tomato", "sugarcane": "Sugarcane",
    }
    detected_crop = None
    for kw, name in crop_keywords.items():
        if kw in query_lower:
            detected_crop = name
            break

    # Weather detection
    weather_keywords = {
        "drought": "Drought", "dry spell": "Drought", "dry": "Drought",
        "heavy rain": "Heavy Rainfall", "heavy rainfall": "Heavy Rainfall",
        "flood": "Heavy Rainfall", "waterlog": "Heavy Rainfall",
        "cold wave": "Cold Wave", "frost": "Cold Wave", "cold": "Cold Wave",
        "heat wave": "Heat Wave", "heatwave": "Heat Wave", "heat": "Heat Wave",
        "fog": "Fog", "foggy": "Fog", "humid": "Fog",
        "monsoon": "Normal Monsoon",
    }
    detected_weather = None
    for kw, name in weather_keywords.items():
        if kw in query_lower:
            detected_weather = name
            break

    # Soil detection
    soil_keywords = {
        "alluvial": "Alluvial", "black soil": "Black",
        "red soil": "Red", "laterite": "Laterite",
    }
    detected_soil = None
    for kw, name in soil_keywords.items():
        if kw in query_lower:
            detected_soil = name
            break

    # Category detection
    category_keywords = {
        "irrigat": "irrigation", "water": "irrigation",
        "pest": "pest_control", "disease": "pest_control",
        "insect": "pest_control", "blight": "pest_control",
        "bollworm": "pest_control", "whitefly": "pest_control",
        "fertiliz": "fertilizer", "nutrient": "fertilizer", "npk": "fertilizer",
        "cold wave": "weather_protection", "frost": "weather_protection",
        "protect": "weather_protection",
    }
    detected_category = None
    for kw, cat in category_keywords.items():
        if kw in query_lower:
            detected_category = cat
            break

    return {
        "crop": detected_crop,
        "weather": detected_weather,
        "soil": detected_soil,
        "category": detected_category,
    }


# ──────────────────── Main (demo + stats) ────────────────────

if __name__ == "__main__":
    kg = AgroKnowledgeGraph()

    print("=" * 60)
    print("AGRICULTURAL KNOWLEDGE GRAPH — Statistics")
    print("=" * 60)

    stats = kg.get_stats()
    print(f"\nTotal Nodes: {stats['total_nodes']}")
    print(f"Total Edges: {stats['total_edges']}")
    print(f"Density:     {stats['density']:.4f}")
    print(f"Is DAG:      {stats['is_dag']}")

    print("\nNode Types:")
    for t, count in stats["node_types"].items():
        print(f"  {t}: {count}")

    print("\nEdge Types:")
    for r, count in stats["edge_types"].items():
        print(f"  {r}: {count}")

    print("\n" + "=" * 60)
    print("DEMO: Multi-hop Query for 'Wheat + Drought'")
    print("=" * 60)

    entities = extract_entities_from_query(
        "What irrigation advice for wheat during drought in alluvial soil?"
    )
    print(f"\nExtracted entities: {json.dumps(entities, indent=2)}")

    context = kg.query_context(
        crop_name=entities["crop"],
        weather=entities["weather"],
        category=entities["category"],
    )
    print("\n" + kg.format_context_for_llm(context))
