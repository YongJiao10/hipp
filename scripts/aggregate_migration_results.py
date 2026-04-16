import json
from pathlib import Path

base_dir = Path("/Users/jy/Documents/HippoMaps-network-first/outputs_migration/hipp_functional_parcellation_network")
methods = [
    "network-gradient",
    "network-prob-cluster-nonneg",
    "network-spectral",
    "network-spectral-nonneg",
    "intrinsic-spectral",
    "intrinsic-spectral-nonneg",
]
atlases = ["kong2019", "lynch2024"]
subjects = ["sub-100610", "sub-102311", "sub-102816"]

results = []

for method in methods:
    for atlas in atlases:
        for sub in subjects:
            summary_path = base_dir / method / atlas / sub / "final_selection_summary.json"
            if summary_path.exists():
                with open(summary_path, "r") as f:
                    data = json.load(f)
                
                for smooth in ["2mm", "4mm"]:
                    for hemi in ["L", "R"]:
                        hemi_data = data["per_smooth"].get(smooth, {}).get("hemis", {}).get(hemi)
                        if not hemi_data:
                            continue
                        
                        k_final = hemi_data.get("k_final")
                        annotations = hemi_data.get("cluster_annotations", [])
                        
                        # Store cluster info
                        clusters = []
                        for ann in annotations:
                            clusters.append({
                                "name": ann.get("cluster_name"),
                                "fraction": ann.get("cluster_fraction"),
                                "network": ann.get("dominant_network")
                            })
                        
                        results.append({
                            "method": method,
                            "atlas": atlas,
                            "subject": sub,
                            "smooth": smooth,
                            "hemi": hemi,
                            "k": k_final,
                            "clusters": clusters
                        })

# Print a summary table
print(f"{'Method':<25} | {'Atlas':<10} | {'Sm':<4} | {'Hemi':<4} | {'Sub-100610':<15} | {'Sub-102311':<15} | {'Sub-102816':<15}")
print("-" * 115)

for method in methods:
    for atlas in atlases:
        for smooth in ["2mm", "4mm"]:
            for hemi in ["L", "R"]:
                ks = []
                for sub in subjects:
                    sub_res = [r for r in results if r["method"] == method and r["atlas"] == atlas and r["smooth"] == smooth and r["subject"] == sub and r["hemi"] == hemi]
                    if sub_res:
                        ks.append(str(sub_res[0]["k"]))
                    else:
                        ks.append("N/A")
                
                # Print K values as a first check of consistency
                print(f"{method:<25} | {atlas:<10} | {smooth:<4} | {hemi:<4} | {ks[0]:<15} | {ks[1]:<15} | {ks[2]:<15}")

print("\nDetailed Cluster Fractions (Dominant Network):")
for method in methods:
    for atlas in atlases:
        print(f"\n--- {method} / {atlas} ---")
        for smooth in ["2mm", "4mm"]:
            for hemi in ["L", "R"]:
                print(f"Smooth {smooth}, Hemi {hemi}:")
                for sub in subjects:
                    sub_res = [r for r in results if r["method"] == method and r["atlas"] == atlas and r["smooth"] == smooth and r["subject"] == sub and r["hemi"] == hemi]
                    if sub_res:
                        c_info = ", ".join([f"{c['network']}({c['fraction']:.2f})" for c in sub_res[0]["clusters"]])
                        print(f"  {sub}: K={sub_res[0]['k']} [{c_info}]")
