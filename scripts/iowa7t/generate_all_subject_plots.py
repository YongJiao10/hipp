#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd

# Add the scripts/core_workflow/lib directory to the path so we can import our new module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_workflow'))

from lib.fc_visualization import plot_combined_fc

def main():
    subjects = ["06112026", "06122026"]
    variants = ["NoXCPD", "36P"]
    branches = ["multiband", "multi_echo"]
    net_types = ["8net", "17net"]
    
    base_dir = "/home/yojiao/LSS/jianglab/3_Data_Working/fmriprep_processed_Iowa7T"
    
    count = 0
    total_processed = 0
    for subj in subjects:
        for var in variants:
            for branch in branches:
                # Path construction
                subj_dir = os.path.join(base_dir, f"DataOrg_PFM_{var}", "hipp_outputs", "subjects", f"sub-{subj}", branch)
                ts_path = os.path.join(subj_dir, "cortex", "kong2022", "cortex_roi_parcel_timeseries.npy")
                tsv_path = os.path.join(subj_dir, "cortex", "kong2022", "cortex_roi_parcels.tsv")
                output_dir = os.path.join(subj_dir, "figures")
                
                if not os.path.exists(ts_path) or not os.path.exists(tsv_path):
                    print(f"Warning: Missing files for sub-{subj} {var} {branch}. Skipping...")
                    continue
                    
                total_processed += 1
                print(f"\n[{total_processed}] Processing sub-{subj} | Pipeline: {var} | Scan: {branch}...")
                print(f"  Loading timeseries: {ts_path}")
                ts = np.load(ts_path)
                
                print(f"  Loading parcels: {tsv_path}")
                df = pd.read_csv(tsv_path, sep='\t')
                
                # Compute correlation matrix and apply Fisher-z transform
                r = np.corrcoef(ts)
                r_clipped = np.clip(r, -0.9999, 0.9999)
                fc_matrix = np.arctanh(r_clipped)
                # Set diagonal elements to NaN to prevent colorbar scaling skew
                np.fill_diagonal(fc_matrix, np.nan)
                
                for net in net_types:
                    # Output path: fc_heatmap_8net or fc_heatmap_17net
                    save_path = os.path.join(output_dir, f"fc_heatmap_{net}")
                    print(f"  Drawing {net} combined diagram to {save_path}.png...")
                    
                    title = f"Subject {subj} ({var} {branch}) FC Heatmap & Chord ({net})"
                    plot_combined_fc(
                        fc_matrix=fc_matrix,
                        parcels_df=df,
                        net_type=net,
                        dark_mode=True,
                        title=title,
                        scale=0.7,
                        cb_title="Fisher's $z$",
                        save_path=save_path
                    )
                    count += 1
                
    print(f"\nSuccessfully generated combined plots. Total configurations: {total_processed}, total images: {count}.")

if __name__ == "__main__":
    main()
