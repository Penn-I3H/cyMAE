#!/usr/bin/env python3.11

import os
import glob
import torch
from timm.models import create_model
import modeling_finetune
import fcsparser
import numpy as np
import json

INPUT_DIR = os.getenv('INPUT_DIR', './service/data/input')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './service/data/output')


def read_fcs(fcs_file):
    meta, data = fcsparser.parse(fcs_file, reformat_meta=True)
    return meta, data


def main():
    device = "cpu"
    checkpoint = torch.load("model/cymae_30D_6L_pretrained0.25R_fold0_0.0064lr_200epoch_checkpoint-best.pth", map_location=torch.device(device), weights_only=False)
    args = checkpoint['args']
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
    model.load_state_dict(checkpoint['model'])
    model.eval()

    idx_to_class = {0: 'Plasmablast', 1: 'Th2/activated', 2: 'Treg/activated', 3: 'CD8Naive', 4: 'Treg', 5: 'EarlyNK', 6: 'CD66bnegCD45lo', 7: 'CD4Naive', 8: 'Th2', 9: 'CD8TEM2', 10: 'Th17', 11: 'IgDposMemB', 12: 'CD8Naive/activated', 13: 'CD8TEMRA/activated', 14: 'Eosinophil', 15: 'CD8TEM3/activated', 16: 'DPT', 17: 'MAITNKT', 18: 'gdT', 19: 'CD8TEM2/activated', 20: 'nnCD4CXCR5pos/activated', 21: 'IgDnegMemB', 22: 'CD45hiCD66bpos', 23: 'LateNK', 24: 'Neutrophil', 25: 'DNT', 26: 'Basophil', 27: 'pDC', 28: 'CD8TEM1/activated', 29: 'mDC', 30: 'Th1', 31: 'DNT/activated', 32: 'Th1/activated', 33: 'CD8TEMRA', 34: 'CD8TCM/activated', 35: 'CD8TEM1', 36: 'CD4Naive/activated', 37: 'NaiveB', 38: 'ILC', 39: 'CD8TEM3', 40: 'Th17/activated', 41: 'CD8TCM', 42: 'ClassicalMono', 43: 'DPT/activated', 44: 'nnCD4CXCR5pos', 45: 'TotalMonocyte'}


    # input marker order should follow this
    marker_list = ['89Y_CD45', '141Pr_CD196_CCR6', '143Nd_CD123_IL-3R', '144Nd_CD19', '145Nd_CD4',
                '146Nd_CD8a', '147Sm_CD11c', '148Nd_CD16', '149Sm_CD45RO', '150Nd_CD45RA',
                '151Eu_CD161', '152Sm_CD194_CCR4', '153Eu_CD25_IL-2Ra', '154Sm_CD27', '155Gd_CD57',
                '156Gd_CD183_CXCR3', '158Gd_CD185_CXCR5', '160Gd_CD28', '161Dy_CD38', '163Dy_CD56_NCAM',
                '164Dy_TCRgd', '166Er_CD294', '167Er_CD197_CCR7', '168Er_CD14', '170Er_CD3',
                '171Yb_CD20', '172Yb_CD66b', '173Yb_HLA-DR', '174Yb_IgD', '176Yb_CD127_IL-7Ra']
    
    fcs_files = glob.glob(f"{INPUT_DIR}/*.fcs")
    for input_path in fcs_files:
        
        _, input_data = read_fcs(input_path)

        missing_markers = [marker for marker in marker_list if marker not in input_data.columns]
        if missing_markers:
            raise ValueError(f"The FCS file must contain all the markers in {marker_list}.\n\nMissing markers: {', '.join(missing_markers)}")
        else:
            input_data = input_data[marker_list].values
            input_data = np.arcsinh(input_data)
            input_data = torch.tensor(input_data).to(device) # torch.tensor (C, 30)

            batch_size = 1024
            preds = []

            with torch.no_grad():
                for i in range(0, input_data.size(0), batch_size):
                    batch_data = input_data[i:i + batch_size]
                    batch_preds = model(batch_data)
                    batch_preds = torch.max(batch_preds, 1)[1]
                    batch_preds = [idx_to_class[idx.item()] for idx in batch_preds]
                    preds.extend(batch_preds)

            output_path = OUTPUT_DIR + '/' + input_path.split("/")[-1].replace(".fcs", "_cymae.json")
            json.dump(preds, open(output_path, "w"))


if __name__ == "__main__":
    main()