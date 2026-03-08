from typing import Dict, Any
from src.data.data_loader import HotpotQALoader

JWST_DOCS = [
    {"id": "doc1", "text": "The James Webb Space Telescope (JWST) is a space telescope designed to conduct infrared astronomy. Its high-resolution and high-sensitivity instruments allow it to view objects too old, distant, or faint for the Hubble Space Telescope. The primary scientific instruments on board include the Near-Infrared Camera (NIRCam) and the Mid-Infrared Instrument (MIRI). These tools are essential for studying the formation of early galaxies. Additionally, the telescope requires a massive sunshield to keep its instruments cold."},
    {"id": "doc2", "text": "The Hubble Space Telescope was launched into low Earth orbit in 1990 and remains in operation. It features a 2.4-meter mirror and observes primarily in the visible and ultraviolet spectra. Hubble has recorded some of the most detailed visible-light images ever, allowing a deep view into space. It does not possess the advanced mid-infrared capabilities of newer observatories. Astronauts have visited Hubble multiple times to repair and upgrade its systems."},
    {"id": "doc3", "text": "One of the major goals of modern astronomy is the study of exoplanets and their atmospheres. JWST uses its Near-Infrared Spectrograph (NIRSpec) to analyze the chemical composition of exoplanetary atmospheres. By observing the transit of a planet across its host star, scientists can detect signatures of water and carbon dioxide. This method of transit spectroscopy is highly dependent on the telescope's stable orbit. Future missions will continue this search for habitable worlds."},
    {"id": "doc4", "text": "JWST was launched on December 25, 2021, on an Ariane 5 rocket from Kourou, French Guiana. It took a month to reach its destination, the Sun-Earth L2 Lagrange point. This orbit allows the telescope to stay in line with Earth as it moves around the Sun. The L2 point is approximately 1.5 million kilometers away from our planet. Operations and data processing are managed by the Space Telescope Science Institute."}
]

def load_demo_datasets(hotpot_path: str = "data/hotpotqa/dev") -> Dict[str, Any]:
    """Returns the available datasets, their queries, and documents."""
    datasets = {
        "jwst": {
            "name": "James Webb Space Telescope (Curated)",
            "queries": [
                "What are the specific instruments on the James Webb Space Telescope used for observation?",
                "Where was the JWST launched from and what is its destination?",
                "How does JWST study exoplanets?"
            ],
            "documents": JWST_DOCS
        },
        "hotpotqa": {
            "name": "HotpotQA (Dev Subset)",
            "queries": [],
            "documents": []
        }
    }
    
    # Safely try to load a few samples from HotpotQA
    try:
        loader = HotpotQALoader(hotpot_path)
        # Grab first 3 queries for demo purposes
        for i in range(min(3, len(loader))):
            example = loader[i]
            datasets["hotpotqa"]["queries"].append(example.question)
            
            # Add documents to the corpus pool (avoiding duplicates)
            existing_titles = {d.get('id') for d in datasets["hotpotqa"]["documents"]}
            for doc in example.documents:
                if doc.title not in existing_titles:
                    datasets["hotpotqa"]["documents"].append({
                        "id": doc.title,
                        "text": doc.text
                    })
                    existing_titles.add(doc.title)
    except Exception as e:
        print(f"Warning: Could not load HotpotQA dataset for demo: {e}")

    return datasets