#!/usr/bin/env python3.10

import sys
import shutil
import os
import glob
import torch
from timm.models import create_model
import modeling_finetune
import fcsparser
import numpy as np
import json
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing

print("starting program")

# Global variables for the model and device
model = None
args = None
device = 'cpu'

INPUT_DIR = os.getenv('INPUT_DIR', './data/input')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './data/output')
NUM_PROCESSES = int(os.getenv('NUM_PROCESSES', '4'))  # Default to 4 processes if not specified

def read_fcs(fcs_file):
    meta, data = fcsparser.parse(fcs_file, reformat_meta=True)
    return meta, data

def init_worker():
    """Initializer function for each worker process to load the model."""
    global model
    global args
    global device

    device = "cpu"
    print("loading model")
    checkpoint = torch.load(
        "model/cymae_30D_6L_pretrained0.25R_fold0_0.0064lr_200epoch_checkpoint-best.pth",
        map_location=torch.device(device),
        weights_only=False
    )
    print("model loaded")
    
    args = checkpoint['args']
    
    print("create model")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    ).to(device)
    print("model created")

    print("loading model state")
    model.load_state_dict(checkpoint['model'])
    print("model state loaded")
    del checkpoint
    
    model.eval()

# Mapping from indices to class names
idx_to_class = {
    0: 'Plasmablast', 1: 'Th2/activated', 2: 'Treg/activated', 3: 'CD8Naive',
    4: 'Treg', 5: 'EarlyNK', 6: 'CD66bnegCD45lo', 7: 'CD4Naive', 8: 'Th2',
    9: 'CD8TEM2', 10: 'Th17', 11: 'IgDposMemB', 12: 'CD8Naive/activated',
    13: 'CD8TEMRA/activated', 14: 'Eosinophil', 15: 'CD8TEM3/activated',
    16: 'DPT', 17: 'MAITNKT', 18: 'gdT', 19: 'CD8TEM2/activated',
    20: 'nnCD4CXCR5pos/activated', 21: 'IgDnegMemB', 22: 'CD45hiCD66bpos',
    23: 'LateNK', 24: 'Neutrophil', 25: 'DNT', 26: 'Basophil', 27: 'pDC',
    28: 'CD8TEM1/activated', 29: 'mDC', 30: 'Th1', 31: 'DNT/activated',
    32: 'Th1/activated', 33: 'CD8TEMRA', 34: 'CD8TCM/activated',
    35: 'CD8TEM1', 36: 'CD4Naive/activated', 37: 'NaiveB', 38: 'ILC',
    39: 'CD8TEM3', 40: 'Th17/activated', 41: 'CD8TCM', 42: 'ClassicalMono',
    43: 'DPT/activated', 44: 'nnCD4CXCR5pos', 45: 'TotalMonocyte'
}

# Map detailed cell types to broader categories
cell_type_mapping = {
    # B cells
    'NaiveB': 'B cells',
    'IgDposMemB': 'B cells',
    'IgDnegMemB': 'B cells',
    'Plasmablast': 'B cells',
    # Basophil
    'Basophil': 'Basophil',
    # CD4 T cells
    'CD4Naive': 'CD4 T cells',
    'Th1': 'CD4 T cells',
    'Th1/activated': 'CD4 T cells',
    'Th2': 'CD4 T cells',
    'Th2/activated': 'CD4 T cells',
    'Th17': 'CD4 T cells',
    'Th17/activated': 'CD4 T cells',
    'Treg': 'CD4 T cells',
    'Treg/activated': 'CD4 T cells',
    'CD4Naive/activated': 'CD4 T cells',
    'DPT': 'CD4 T cells',
    'DPT/activated': 'CD4 T cells',
    'nnCD4CXCR5pos': 'CD4 T cells',
    'nnCD4CXCR5pos/activated': 'CD4 T cells',
    # CD45hiCD66bpos
    'CD45hiCD66bpos': 'CD45hiCD66bpos',
    # CD66bnegCD45lo
    'CD66bnegCD45lo': 'CD66bnegCD45lo',
    # CD8 T cells
    'CD8Naive': 'CD8 T cells',
    'CD8Naive/activated': 'CD8 T cells',
    'CD8TEM1': 'CD8 T cells',
    'CD8TEM1/activated': 'CD8 T cells',
    'CD8TEM2': 'CD8 T cells',
    'CD8TEM2/activated': 'CD8 T cells',
    'CD8TEM3': 'CD8 T cells',
    'CD8TEM3/activated': 'CD8 T cells',
    'CD8TEMRA': 'CD8 T cells',
    'CD8TEMRA/activated': 'CD8 T cells',
    'CD8TCM': 'CD8 T cells',
    'CD8TCM/activated': 'CD8 T cells',
    # Eosinophils
    'Eosinophil': 'Eosinophil',
    # ILC
    'ILC': 'ILC',
    # Monocytes/mDC
    'ClassicalMono': 'Monocytes/mDC',
    'TotalMonocyte': 'Monocytes/mDC',
    'mDC': 'Monocytes/mDC',
    # NK cell
    'EarlyNK': 'NK cell',
    'LateNK': 'NK cell',
    # Neutrophil
    'Neutrophil': 'Neutrophil',
    # Other T cells
    'MAITNKT': 'Other T cells',
    'gdT': 'Other T cells',
    'DNT': 'Other T cells',
    'DNT/activated': 'Other T cells',
    # pDC
    'pDC': 'pDC',
}

# Cell types of interest (for highlighting in the UMAP plot)
cell_types_of_interest = [
    key for key, value in cell_type_mapping.items() if value == 'CD4 T cells'
]

# Input marker order should follow this list
marker_list = [
    '89Y_CD45', '141Pr_CD196_CCR6', '143Nd_CD123_IL-3R', '144Nd_CD19', '145Nd_CD4',
    '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD16', '149Sm_CD45RO', '150Nd_CD45RA',
    '151Eu_CD161', '152Sm_CD194_CCR4', '153Eu_CD25_IL-2Ra', '154Sm_CD27', '155Gd_CD57',
    '156Gd_CD183_CXCR3', '158Gd_CD185_CXCR5', '160Gd_CD28', '161Dy_CD38', '163Dy_CD56_NCAM',
    '164Dy_TCRgd', '166Er_CD294', '167Er_CD197_CCR7', '168Er_CD14', '170Er_CD3',
    '171Yb_CD20', '172Yb_CD66b', '173Yb_HLA-DR', '174Yb_IgD', '176Yb_CD127_IL-7Ra'
]

def process_file(input_path):
    global model
    global device
    print("start processing")
    try:
        print("file reading")
        _, input_data = read_fcs(input_path)
        print("file read")

        missing_markers = [marker for marker in marker_list if marker not in input_data.columns]
        if missing_markers:
            print("missing markers")
            raise ValueError(f"The FCS file must contain all the markers in {marker_list}.\n\nMissing markers: {', '.join(missing_markers)}")
        else:
            print("data processing")
            input_data = input_data[marker_list].values
            input_data = torch.tensor(np.arcsinh(input_data)).to(device)  # torch.tensor (C, 30)

            batch_size = 1024
            preds = []
            
            print("running model")
            with torch.no_grad():
                for i in range(0, input_data.size(0), batch_size):
                    batch_data = input_data[i:i + batch_size]
                    batch_preds = model(batch_data)
                    batch_preds = torch.max(batch_preds, 1)[1]
                    preds.extend([idx_to_class[idx.item()] for idx in batch_preds])
            print("model finished")
            json.dump(preds, open(OUTPUT_DIR + '/' + os.path.basename(input_path).replace(".fcs", "/CyMAE_raw_output.json"), 'w'))

            # Fit UMAP on the input data
            print("fitting UMAP")
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(input_data.cpu().numpy())
            print("UMAP fitted")

            # Only color cell types of interest
            # Create a color palette
            print("creating color palette for interested cells")
            unique_types, counts = np.unique(preds, return_counts=True)
            unique_classes = list(unique_types)
            palette = sns.color_palette("tab20", len(cell_types_of_interest))
            class_to_color = {cls: palette[i % len(palette)] for i, cls in enumerate(cell_types_of_interest)}
            default_color = (0.8, 0.8, 0.8)  # Gray color for other cell types
            print("color palette created")

            # Plot UMAP embedding with colors based on interested categories
            print("plotting UMAP for interested cells")
            plt.figure(figsize=(12, 10))
            for cls in unique_classes:
                indices = [i for i, pred in enumerate(preds) if pred == cls]
                if cls in cell_types_of_interest:
                    plt.scatter(embedding[indices, 0], embedding[indices, 1], c=[class_to_color[cls]], label=cls, s=5)
                else:
                    plt.scatter(embedding[indices, 0], embedding[indices, 1], c=[default_color], label=None, s=5, alpha=0.5)

            plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title("Cells classified by CyMAE (Highlighted by interest)", fontsize=20)
            plt.xlabel("UMAP1", fontsize=15)
            plt.ylabel("UMAP2", fontsize=15)
            plt.tight_layout()
            print("UMAP plotted")

            # Save the plot as a PNG file
            print("saving image")
            output_image_path = OUTPUT_DIR + '/' + os.path.basename(input_path).replace(".fcs", "/CyMAE_(interest_cells).png")
            plt.savefig(output_image_path, dpi=300)
            plt.close()
            print("image saved")

            # Create plots of broad categories
            # Map detailed cell types to broader categories
            print("mapping broad categories")
            broad_preds = [cell_type_mapping.get(p, 'Others') for p in preds]
            print("broad categories mapped")

            # Create a color palette
            print("creating color palette for broad categories")
            unique_broad_classes = list(set(broad_preds))
            palette = sns.color_palette("tab20", len(unique_broad_classes))
            class_to_color = {cls: palette[i % len(palette)] for i, cls in enumerate(unique_broad_classes)}
            print("color palette created")

            # Plot UMAP embedding with colors based on broad categories
            print("plotting UMAP for broad categories")
            plt.figure(figsize=(12, 10))
            for cls in unique_broad_classes:
                indices = [i for i, pred in enumerate(broad_preds) if pred == cls]
                plt.scatter(embedding[indices, 0], embedding[indices, 1], c=[class_to_color[cls]], label=cls, s=5)

            plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title("Cells classified by CyMAE (Broad Categories)", fontsize=20)
            plt.xlabel("UMAP1", fontsize=15)
            plt.ylabel("UMAP2", fontsize=15)
            plt.tight_layout()
            print("UMAP plotted")

            # Save the plot as a PNG file
            print("saving image")
            output_image_path = OUTPUT_DIR + '/' + os.path.basename(input_path).replace(".fcs", "/CyMAE_(broad_types).png")
            plt.savefig(output_image_path, dpi=300)
            plt.close()
            print("image saved")

            print(f"Processing of file {input_path} completed.")
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")

if __name__ == '__main__':
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    fcs_files = glob.glob(f"{INPUT_DIR}/*.fcs")
    for file in fcs_files:
        os.makedirs(OUTPUT_DIR + f'/{os.path.basename(file).replace(".fcs", "")}', exist_ok=True)

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=init_worker) as pool:
        pool.map(process_file, fcs_files)
