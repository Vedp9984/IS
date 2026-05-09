"""
dataset.py
==========
Synthetic Agricultural Advisory Dataset
Provides structured data for Knowledge Graph construction and RAG retrieval.

Entities: Crop, Soil, Weather, Advisory
Relations: affects, requires, causes, recommended_for
"""

# ─────────────────────────── Crop Profiles ───────────────────────────
CROPS = [
    {
        "id": "crop_wheat",
        "name": "Wheat",
        "season": "Rabi",
        "water_need": "moderate",
        "soil_preference": "loamy",
        "temp_range_c": (10, 25),
        "diseases": ["rust", "smut", "blight"],
        "growth_stages": ["germination", "tillering", "heading", "maturity"],
    },
    {
        "id": "crop_rice",
        "name": "Rice",
        "season": "Kharif",
        "water_need": "high",
        "soil_preference": "clayey",
        "temp_range_c": (20, 35),
        "diseases": ["blast", "sheath_blight", "brown_spot"],
        "growth_stages": ["nursery", "transplanting", "tillering", "flowering", "maturity"],
    },
    {
        "id": "crop_maize",
        "name": "Maize",
        "season": "Kharif",
        "water_need": "moderate",
        "soil_preference": "loamy",
        "temp_range_c": (18, 32),
        "diseases": ["downy_mildew", "stem_borer", "fall_armyworm"],
        "growth_stages": ["germination", "vegetative", "tasseling", "maturity"],
    },
    {
        "id": "crop_cotton",
        "name": "Cotton",
        "season": "Kharif",
        "water_need": "moderate",
        "soil_preference": "black_soil",
        "temp_range_c": (21, 35),
        "diseases": ["bollworm", "whitefly", "leaf_curl"],
        "growth_stages": ["germination", "squaring", "flowering", "boll_opening"],
    },
    {
        "id": "crop_tomato",
        "name": "Tomato",
        "season": "Rabi",
        "water_need": "moderate",
        "soil_preference": "sandy_loam",
        "temp_range_c": (15, 30),
        "diseases": ["early_blight", "late_blight", "leaf_curl_virus"],
        "growth_stages": ["seedling", "vegetative", "flowering", "fruiting"],
    },
    {
        "id": "crop_sugarcane",
        "name": "Sugarcane",
        "season": "Kharif",
        "water_need": "very_high",
        "soil_preference": "loamy",
        "temp_range_c": (20, 35),
        "diseases": ["red_rot", "smut", "top_borer"],
        "growth_stages": ["germination", "tillering", "grand_growth", "maturity"],
    },
]

# ─────────────────────────── Soil Types ───────────────────────────
SOILS = [
    {
        "id": "soil_alluvial",
        "name": "Alluvial Soil",
        "type": "loamy",
        "ph_range": (6.5, 7.5),
        "fertility": "high",
        "water_retention": "moderate",
        "suitable_crops": ["wheat", "rice", "maize", "sugarcane"],
    },
    {
        "id": "soil_black",
        "name": "Black Soil (Regur)",
        "type": "clayey",
        "ph_range": (7.0, 8.5),
        "fertility": "high",
        "water_retention": "high",
        "suitable_crops": ["cotton", "wheat", "sugarcane"],
    },
    {
        "id": "soil_red",
        "name": "Red Soil",
        "type": "sandy_loam",
        "ph_range": (5.5, 6.5),
        "fertility": "low",
        "water_retention": "low",
        "suitable_crops": ["maize", "tomato"],
    },
    {
        "id": "soil_laterite",
        "name": "Laterite Soil",
        "type": "sandy",
        "ph_range": (5.0, 6.0),
        "fertility": "low",
        "water_retention": "low",
        "suitable_crops": ["rice", "tomato"],
    },
]

# ─────────────────────────── Weather Conditions ───────────────────────────
WEATHER_CONDITIONS = [
    {
        "id": "weather_heavy_rain",
        "condition": "Heavy Rainfall",
        "rainfall_mm": 150,
        "humidity_pct": 90,
        "temp_c": 28,
        "wind_kph": 20,
        "impact": "waterlogging, disease spread, nutrient leaching",
    },
    {
        "id": "weather_drought",
        "condition": "Drought / Dry Spell",
        "rainfall_mm": 0,
        "humidity_pct": 25,
        "temp_c": 40,
        "wind_kph": 15,
        "impact": "wilting, stunted growth, soil cracking",
    },
    {
        "id": "weather_normal_monsoon",
        "condition": "Normal Monsoon",
        "rainfall_mm": 60,
        "humidity_pct": 75,
        "temp_c": 30,
        "wind_kph": 10,
        "impact": "favorable for kharif crops",
    },
    {
        "id": "weather_cold_wave",
        "condition": "Cold Wave",
        "rainfall_mm": 5,
        "humidity_pct": 60,
        "temp_c": 4,
        "wind_kph": 25,
        "impact": "frost damage, delayed growth",
    },
    {
        "id": "weather_heatwave",
        "condition": "Heat Wave",
        "rainfall_mm": 0,
        "humidity_pct": 20,
        "temp_c": 45,
        "wind_kph": 10,
        "impact": "heat stress, pollen sterility, increased water demand",
    },
    {
        "id": "weather_foggy",
        "condition": "Fog / High Humidity",
        "rainfall_mm": 2,
        "humidity_pct": 95,
        "temp_c": 12,
        "wind_kph": 5,
        "impact": "fungal diseases, reduced photosynthesis",
    },
]

# ─────────────────────────── Expert Advisories ───────────────────────────
ADVISORIES = [
    {
        "id": "adv_wheat_irrigation",
        "crop": "Wheat",
        "category": "irrigation",
        "condition": "Drought / Dry Spell",
        "soil": "Alluvial Soil",
        "advisory": (
            "During drought conditions, provide supplemental irrigation to wheat at "
            "critical growth stages — crown root initiation (21 DAS), tillering, "
            "jointing, flowering, and grain filling. Use sprinkler irrigation if "
            "available to optimize water usage. Apply mulch (5-7 cm) to conserve "
            "soil moisture. Avoid flood irrigation to prevent water wastage."
        ),
    },
    {
        "id": "adv_rice_pest",
        "crop": "Rice",
        "category": "pest_control",
        "condition": "Normal Monsoon",
        "soil": "Alluvial Soil",
        "advisory": (
            "During monsoon, rice is susceptible to blast and sheath blight. "
            "Monitor fields weekly for lesion symptoms. Apply Tricyclazole (0.06%) "
            "or Carbendazim (0.1%) as preventive spray at tillering stage. Maintain "
            "2-5 cm standing water. Remove and destroy infected plant debris. "
            "Use resistant varieties like Pusa Basmati 1509 where possible."
        ),
    },
    {
        "id": "adv_rice_heavy_rain",
        "crop": "Rice",
        "category": "irrigation",
        "condition": "Heavy Rainfall",
        "soil": "Alluvial Soil",
        "advisory": (
            "During heavy rainfall, ensure proper drainage channels are functional "
            "to prevent prolonged waterlogging beyond 48 hours. Waterlogging at "
            "flowering stage causes significant yield loss. After water recedes, "
            "apply foliar spray of 2% urea to revive stressed plants. Check for "
            "brown spot disease which thrives in post-flood conditions."
        ),
    },
    {
        "id": "adv_maize_fertilizer",
        "crop": "Maize",
        "category": "fertilizer",
        "condition": "Normal Monsoon",
        "soil": "Red Soil",
        "advisory": (
            "For maize in red soil, apply NPK fertilizer in split doses: "
            "basal dose of 60:40:20 kg/ha at sowing, followed by 30 kg N/ha at "
            "knee-high stage and 30 kg N/ha at tasseling. Red soil is typically "
            "deficient in nitrogen and phosphorus — supplement with FYM "
            "(10 tonnes/ha) before sowing. Use zinc sulphate (25 kg/ha) to "
            "correct micronutrient deficiency."
        ),
    },
    {
        "id": "adv_cotton_pest",
        "crop": "Cotton",
        "category": "pest_control",
        "condition": "Normal Monsoon",
        "soil": "Black Soil (Regur)",
        "advisory": (
            "Monitor cotton fields for bollworm and whitefly from squaring stage "
            "onwards. Install pheromone traps (5/ha) for bollworm monitoring. "
            "Spray Neem oil (5 ml/L) as first line of defense. If pest population "
            "exceeds ETL (Economic Threshold Level), apply Emamectin benzoate "
            "(0.2 g/L). For whitefly, use yellow sticky traps and spray "
            "Thiamethoxam (0.2 g/L) if needed."
        ),
    },
    {
        "id": "adv_wheat_cold",
        "crop": "Wheat",
        "category": "weather_protection",
        "condition": "Cold Wave",
        "soil": "Alluvial Soil",
        "advisory": (
            "During cold wave alerts, apply light irrigation in the evening to "
            "raise soil temperature and protect wheat from frost damage. Spray "
            "sulphuric acid (0.1%) solution for frost protection. Avoid nitrogen "
            "top-dressing during extreme cold. If frost damage has occurred, "
            "apply foliar spray of 0.5% KCl and 2% urea for recovery. "
            "Delay harvesting of frost-affected crop by 7-10 days."
        ),
    },
    {
        "id": "adv_tomato_blight",
        "crop": "Tomato",
        "category": "pest_control",
        "condition": "Fog / High Humidity",
        "soil": "Red Soil",
        "advisory": (
            "Foggy and humid conditions favor late blight (Phytophthora infestans) "
            "in tomato. Apply Mancozeb (2.5 g/L) or Metalaxyl + Mancozeb "
            "combination as preventive spray at 10-day intervals. Ensure proper "
            "plant spacing (60x45 cm) for air circulation. Stake plants to keep "
            "foliage off the soil. Remove lower leaves touching the ground. "
            "Avoid overhead irrigation."
        ),
    },
    {
        "id": "adv_sugarcane_heat",
        "crop": "Sugarcane",
        "category": "irrigation",
        "condition": "Heat Wave",
        "soil": "Alluvial Soil",
        "advisory": (
            "During heat waves, increase irrigation frequency to every 7 days "
            "for sugarcane. Apply trash mulching (dried leaves, 10 cm layer) "
            "between rows to reduce soil temperature by 3-5°C. Irrigate during "
            "early morning or late evening. Avoid field operations during peak "
            "heat hours (12-3 PM). Spray 2% kaolin solution on leaves to reduce "
            "heat stress. Grand growth stage is most sensitive to heat stress."
        ),
    },
    {
        "id": "adv_maize_drought",
        "crop": "Maize",
        "category": "irrigation",
        "condition": "Drought / Dry Spell",
        "soil": "Red Soil",
        "advisory": (
            "During drought, maize in red soil requires critical irrigation at "
            "tasseling and silking stages. Use drip irrigation if available "
            "(saves 40-60% water). Apply organic mulch to reduce evaporation. "
            "Spray anti-transpirant (kaolin 6%) to reduce water loss from leaves. "
            "If prolonged drought is expected, consider short-duration varieties "
            "for the next season."
        ),
    },
    {
        "id": "adv_rice_fertilizer",
        "crop": "Rice",
        "category": "fertilizer",
        "condition": "Normal Monsoon",
        "soil": "Alluvial Soil",
        "advisory": (
            "For rice in alluvial soil during normal monsoon, apply NPK at "
            "120:60:40 kg/ha in splits — 50% N as basal, 25% at tillering, "
            "25% at panicle initiation. All P and K as basal. Use Leaf Color "
            "Chart (LCC) for need-based nitrogen application. Apply zinc "
            "sulphate (25 kg/ha) in zinc-deficient areas. Incorporate green "
            "manure (Sesbania/Dhaincha) 45 days before transplanting."
        ),
    },
]

# ─────────────────────────── Relations (for KG edges) ───────────────────────────
RELATIONS = [
    # Weather AFFECTS Crop
    ("weather_heavy_rain", "affects", "crop_rice", {"impact": "waterlogging risk, disease spread"}),
    ("weather_heavy_rain", "affects", "crop_wheat", {"impact": "lodging, root rot risk"}),
    ("weather_drought", "affects", "crop_wheat", {"impact": "wilting, reduced yield"}),
    ("weather_drought", "affects", "crop_maize", {"impact": "pollen sterility, poor grain fill"}),
    ("weather_cold_wave", "affects", "crop_wheat", {"impact": "frost damage at flowering"}),
    ("weather_cold_wave", "affects", "crop_tomato", {"impact": "chilling injury, stunted growth"}),
    ("weather_heatwave", "affects", "crop_sugarcane", {"impact": "heat stress, increased water demand"}),
    ("weather_heatwave", "affects", "crop_cotton", {"impact": "flower drop, boll shedding"}),
    ("weather_foggy", "affects", "crop_tomato", {"impact": "fungal disease outbreak"}),
    ("weather_foggy", "affects", "crop_wheat", {"impact": "rust and smut risk"}),
    ("weather_normal_monsoon", "affects", "crop_rice", {"impact": "favorable growth conditions"}),
    ("weather_normal_monsoon", "affects", "crop_maize", {"impact": "good vegetative growth"}),
    ("weather_normal_monsoon", "affects", "crop_cotton", {"impact": "pest pressure increase"}),

    # Crop REQUIRES Soil
    ("crop_wheat", "requires", "soil_alluvial", {"reason": "loamy texture, good fertility"}),
    ("crop_rice", "requires", "soil_alluvial", {"reason": "water retention, nutrient-rich"}),
    ("crop_maize", "requires", "soil_red", {"reason": "well-drained, moderately fertile"}),
    ("crop_maize", "requires", "soil_alluvial", {"reason": "alternative: high fertility"}),
    ("crop_cotton", "requires", "soil_black", {"reason": "high water retention, self-mulching"}),
    ("crop_tomato", "requires", "soil_red", {"reason": "well-drained sandy loam"}),
    ("crop_sugarcane", "requires", "soil_alluvial", {"reason": "deep loamy soil, nutrient-rich"}),

    # Weather CAUSES condition
    ("weather_heavy_rain", "causes", "waterlogging", {}),
    ("weather_heavy_rain", "causes", "nutrient_leaching", {}),
    ("weather_drought", "causes", "soil_moisture_deficit", {}),
    ("weather_drought", "causes", "crop_wilting", {}),
    ("weather_cold_wave", "causes", "frost_damage", {}),
    ("weather_heatwave", "causes", "heat_stress", {}),
    ("weather_heatwave", "causes", "pollen_sterility", {}),
    ("weather_foggy", "causes", "fungal_outbreak", {}),

    # Advisory RECOMMENDED_FOR Crop
    ("adv_wheat_irrigation", "recommended_for", "crop_wheat", {"context": "drought"}),
    ("adv_rice_pest", "recommended_for", "crop_rice", {"context": "monsoon"}),
    ("adv_rice_heavy_rain", "recommended_for", "crop_rice", {"context": "heavy_rain"}),
    ("adv_maize_fertilizer", "recommended_for", "crop_maize", {"context": "monsoon, red_soil"}),
    ("adv_cotton_pest", "recommended_for", "crop_cotton", {"context": "monsoon"}),
    ("adv_wheat_cold", "recommended_for", "crop_wheat", {"context": "cold_wave"}),
    ("adv_tomato_blight", "recommended_for", "crop_tomato", {"context": "foggy"}),
    ("adv_sugarcane_heat", "recommended_for", "crop_sugarcane", {"context": "heatwave"}),
    ("adv_maize_drought", "recommended_for", "crop_maize", {"context": "drought"}),
    ("adv_rice_fertilizer", "recommended_for", "crop_rice", {"context": "monsoon, alluvial"}),
]

# ─────────────────────────── Test Queries ───────────────────────────
TEST_QUERIES = [
    {
        "id": "Q1",
        "query": "What irrigation advice should be given for wheat during drought conditions in alluvial soil?",
        "category": "irrigation",
        "expected_crop": "Wheat",
        "expected_condition": "Drought / Dry Spell",
    },
    {
        "id": "Q2",
        "query": "How to manage pests in rice during the monsoon season?",
        "category": "pest_control",
        "expected_crop": "Rice",
        "expected_condition": "Normal Monsoon",
    },
    {
        "id": "Q3",
        "query": "What is the impact of heavy rainfall on rice and how to handle waterlogging?",
        "category": "irrigation",
        "expected_crop": "Rice",
        "expected_condition": "Heavy Rainfall",
    },
    {
        "id": "Q4",
        "query": "Recommend fertilizer schedule for maize grown in red soil during monsoon.",
        "category": "fertilizer",
        "expected_crop": "Maize",
        "expected_condition": "Normal Monsoon",
    },
    {
        "id": "Q5",
        "query": "How to protect cotton from bollworm and whitefly infestation?",
        "category": "pest_control",
        "expected_crop": "Cotton",
        "expected_condition": "Normal Monsoon",
    },
    {
        "id": "Q6",
        "query": "What precautions should wheat farmers take during a cold wave?",
        "category": "weather_protection",
        "expected_crop": "Wheat",
        "expected_condition": "Cold Wave",
    },
    {
        "id": "Q7",
        "query": "How to prevent late blight in tomato during foggy conditions?",
        "category": "pest_control",
        "expected_crop": "Tomato",
        "expected_condition": "Fog / High Humidity",
    },
    {
        "id": "Q8",
        "query": "What irrigation management is needed for sugarcane during heat waves?",
        "category": "irrigation",
        "expected_crop": "Sugarcane",
        "expected_condition": "Heat Wave",
    },
    {
        "id": "Q9",
        "query": "How to manage maize crop during prolonged drought in red soil regions?",
        "category": "irrigation",
        "expected_crop": "Maize",
        "expected_condition": "Drought / Dry Spell",
    },
    {
        "id": "Q10",
        "query": "What fertilizer recommendations exist for rice in alluvial soil during monsoon?",
        "category": "fertilizer",
        "expected_crop": "Rice",
        "expected_condition": "Normal Monsoon",
    },
]


def get_all_documents():
    """Return all advisories as text documents for RAG indexing."""
    docs = []
    for adv in ADVISORIES:
        doc_text = (
            f"Crop: {adv['crop']} | Category: {adv['category']} | "
            f"Weather: {adv['condition']} | Soil: {adv['soil']}\n"
            f"Advisory: {adv['advisory']}"
        )
        docs.append({"id": adv["id"], "text": doc_text, "metadata": adv})
    return docs


def get_kg_context_documents():
    """Return crop+soil+weather descriptions as additional context docs."""
    docs = []
    for crop in CROPS:
        text = (
            f"Crop: {crop['name']} | Season: {crop['season']} | "
            f"Water Need: {crop['water_need']} | "
            f"Preferred Soil: {crop['soil_preference']} | "
            f"Temperature Range: {crop['temp_range_c'][0]}-{crop['temp_range_c'][1]}°C | "
            f"Common Diseases: {', '.join(crop['diseases'])} | "
            f"Growth Stages: {', '.join(crop['growth_stages'])}"
        )
        docs.append({"id": crop["id"], "text": text})

    for soil in SOILS:
        text = (
            f"Soil: {soil['name']} | Type: {soil['type']} | "
            f"pH Range: {soil['ph_range'][0]}-{soil['ph_range'][1]} | "
            f"Fertility: {soil['fertility']} | "
            f"Water Retention: {soil['water_retention']} | "
            f"Suitable Crops: {', '.join(soil['suitable_crops'])}"
        )
        docs.append({"id": soil["id"], "text": text})

    for w in WEATHER_CONDITIONS:
        text = (
            f"Weather: {w['condition']} | Rainfall: {w['rainfall_mm']}mm | "
            f"Humidity: {w['humidity_pct']}% | Temp: {w['temp_c']}°C | "
            f"Wind: {w['wind_kph']}kph | Impact: {w['impact']}"
        )
        docs.append({"id": w["id"], "text": text})

    return docs


if __name__ == "__main__":
    print(f"Dataset Summary:")
    print(f"  Crops:      {len(CROPS)}")
    print(f"  Soils:      {len(SOILS)}")
    print(f"  Weather:    {len(WEATHER_CONDITIONS)}")
    print(f"  Advisories: {len(ADVISORIES)}")
    print(f"  Relations:  {len(RELATIONS)}")
    print(f"  Test Queries: {len(TEST_QUERIES)}")
    print(f"\n  Advisory Documents: {len(get_all_documents())}")
    print(f"  Context Documents:  {len(get_kg_context_documents())}")
