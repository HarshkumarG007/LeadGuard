"""Generate frozen 1000-world corpus for F7.5 optimizer benchmark."""

import numpy as np
import json
from pathlib import Path

def generate_frozen_corpus(output_path="data/test_fixtures/f7_regret_corpus_v1.json"):
    np.random.seed(42)
    N = 10
    worlds = []
    
    for i in range(1000):
        # Randomize world
        p_lead = np.random.uniform(0.01, 0.99, size=N)
        insp_costs = np.random.uniform(100, 1000, size=N)
        interv_values = np.random.uniform(5000, 15000, size=N)
        interv_costs = np.random.uniform(1000, 5000, size=N)
        budget = np.random.uniform(500, 5000)
        
        world = {
            "world_id": i,
            "N": N,
            "p_lead": p_lead.tolist(),
            "insp_costs": insp_costs.tolist(),
            "interv_values": interv_values.tolist(),
            "interv_costs": interv_costs.tolist(),
            "budget": float(budget)
        }
        worlds.append(world)
        
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(worlds, f, indent=2)
        
    print(f"Generated {len(worlds)} worlds and saved to {output_path}")

if __name__ == "__main__":
    generate_frozen_corpus()
