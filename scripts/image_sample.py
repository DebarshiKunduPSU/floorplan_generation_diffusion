"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import numpy as np
import torch as th
import io
import PIL.Image as Image
import drawSvg as drawsvg
import cairosvg
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_fid.fid_score import calculate_fid_given_paths
import webcolors
import networkx as nx
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos

from house_diffusion import dist_util, logger
from house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)
# Import both data loaders
from house_diffusion.rplanhg_datasets import load_rplanhg_data
from house_diffusion.floorset_datasets import load_floorset_data

# --- Generic Color Palette for FloorSet Visualization ---
FLOORSET_COLORS = [
    '#FFB6C1', '#87CEEB', '#98FB98', '#FFD700', '#FFA07A',
    '#20B2AA', '#DDA0DD', '#B0C4DE', '#FFDEAD', '#778899',
    '#F08080', '#ADD8E6', '#90EE90', '#F0E68C', '#E9967A'
]

def save_samples(
        sample, ext, model_kwargs, 
        tmp_count, dataset_name,
        save_gif=False, save_edges=False,
        door_indices = [11, 12, 13], ID_COLOR=None,
        is_syn=False, draw_graph=False, save_svg=False):
    """
    Visualizes and saves the generated floorplan samples.
    Adapted to handle both RPLAN and FloorSet datasets.
    """
    prefix = 'syn_' if is_syn else ''
    graph_errors = []
    if not save_gif:
        sample = sample[-1:]

    for i in tqdm(range(sample.shape[1])):
        resolution = 256
        images = []

        for k in range(sample.shape[0]):
            draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw_color.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='white'))
            
            polys = []
            types = []
            
            # This mask tells us which points are real vs. padding
            valid_points_mask = model_kwargs[f'{prefix}src_key_padding_mask'][i] == 0
            
            # Get room indices for valid points
            room_indices_tensor = model_kwargs[f'{prefix}room_indices'][i][valid_points_mask]
            
            # Find unique room indices and their counts to reconstruct the polygons
            unique_rooms, room_start_indices = th.unique_consecutive(room_indices_tensor, return_inverse=True, return_counts=True)
            
            points_sequence = sample[k][i][valid_points_mask].cpu().data.numpy()
            points_sequence = (points_sequence / 2 + 0.5) * resolution

            # Split the flat sequence of points back into per-room polygons
            points_split = th.split(th.from_numpy(points_sequence), tuple(room_start_indices.tolist()))

            for room_idx, poly_points in enumerate(points_split):
                polys.append(poly_points.tolist())
                # For FloorSet, we'll just use the index for coloring
                if dataset_name == 'floorset':
                    types.append(room_idx)
                else: # For RPLAN, use the semantic room type
                    room_type_onehot = model_kwargs[f'{prefix}room_types'][i][sum(room_start_indices[:room_idx])].cpu().numpy()
                    types.append(np.argmax(room_type_onehot))

            # Draw the polygons
            for poly, c_idx in zip(polys, types):
                if dataset_name == 'rplan' and c_idx in door_indices:
                    continue # Skip drawing doors for RPLAN as they are not closed polygons

                if dataset_name == 'floorset':
                    color = FLOORSET_COLORS[c_idx % len(FLOORSET_COLORS)]
                else: # RPLAN
                    color = ID_COLOR.get(c_idx, '#BEBEBE') # Default to grey if type is unknown

                if len(poly) > 1:
                    draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=color, fill_opacity=0.8, stroke='black', stroke_width=1))

            # Save the generated image
            if k == sample.shape[0] - 1:
                if save_svg:
                    draw_color.saveSvg(f'outputs/{ext}/{tmp_count+i}c_{k}_{ext}.svg')
                else:
                    png_data = cairosvg.svg2png(bytestring=draw_color.asSvg())
                    Image.open(io.BytesIO(png_data)).save(f'outputs/{ext}/{tmp_count+i}c_{ext}.png')

    return graph_errors


def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)

    dist_util.setup_dist()
    logger.configure(dir=f"outputs/{args.dataset}_samples")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    tmp_count = 0
    os.makedirs(f'outputs/pred', exist_ok=True)
    os.makedirs(f'outputs/gt', exist_ok=True)

    ID_COLOR = None
    if args.dataset == 'rplan':
        ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                    6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
                    13: '#785A67', 12: '#D3A2C7'}
        data = load_rplanhg_data(
            batch_size=args.batch_size,
            analog_bit=args.analog_bit,
            set_name=args.set_name,
            target_set=args.target_set,
        )
    elif args.dataset == 'floorset':
        data = load_floorset_data(
            batch_size=args.batch_size,
            analog_bit=args.analog_bit,
            set_name=args.set_name,
        )
    else:
        print("dataset does not exist!")
        assert False

    while tmp_count < args.num_samples:
        batch, model_kwargs = next(data)
        for key in model_kwargs:
            if isinstance(model_kwargs[key], th.Tensor):
                model_kwargs[key] = model_kwargs[key].to(dist_util.dev())

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        sample = sample_fn(
            model,
            batch.shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            analog_bit=args.analog_bit,
        )
        
        # Transpose to [time, batch, points, coords]
        sample = sample.permute([0, 1, 3, 2])
        
        # Prepare ground truth for visualization
        sample_gt = batch.to(dist_util.dev()).unsqueeze(0)
        sample_gt = sample_gt.permute([0, 1, 3, 2])

        save_samples(sample_gt, 'gt', model_kwargs, tmp_count, args.dataset, ID_COLOR=ID_COLOR, save_svg=args.save_svg)
        save_samples(sample, 'pred', model_kwargs, tmp_count, args.dataset, is_syn=True, ID_COLOR=ID_COLOR, save_svg=args.save_svg)
        
        tmp_count += batch.shape[0]

    logger.log("sampling complete")
    try:
        fid_score = calculate_fid_given_paths(['outputs/gt', 'outputs/pred'], 64, 'cuda', 2048)
        print(f'FID Score: {fid_score}')
    except Exception as e:
        print(f"Could not calculate FID score: {e}")


def create_argparser():
    defaults = dict(
        dataset='',
        clip_denoised=True,
        num_samples=64,
        batch_size=16,
        use_ddim=False,
        model_path="",
        draw_graph=False,
        save_svg=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
