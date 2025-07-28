"""
This script is adapted from rplanhg_datasets.py to load and process the FloorSet-Prime dataset
for use with the house-diffusion model.
"""
import os
import random
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
from prime_dataset import FloorplanDatasetPrime # Import the helper class

def floorplan_collate_fn(batch):
    """
    Custom collate function to correctly batch the output of our FloorSetPrimeDataset.
    It handles the tuple of (coordinates, conditioning_dictionary).
    """
    # Filter out None values which can occur if a sample fails to load
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None

    coords_list = [item[0] for item in batch]
    cond_dict_list = [item[1] for item in batch]

    batched_coords = np.stack(coords_list, axis=0)
    batched_coords = th.from_numpy(batched_coords)

    batched_cond = {}
    if cond_dict_list:
        keys = cond_dict_list[0].keys()
        for key in keys:
            items = [d[key] for d in cond_dict_list]
            if isinstance(items[0], np.ndarray):
                stacked_items = np.stack(items, axis=0)
                batched_cond[key] = th.from_numpy(stacked_items)
            elif isinstance(items[0], torch.Tensor):
                 batched_cond[key] = th.stack(items, dim=0)
            else:
                batched_cond[key] = items

    return batched_coords, batched_cond


def load_floorset_data(
    batch_size,
    analog_bit,
    set_name='train',
    data_dir = '/home/dqk5620/research/floorplanning/floorset_housediffusion/PrimeTensorData'
):
    """
    Creates a data loader for the FloorSet-Prime dataset.
    """
    print(f"Loading FloorSet-Prime '{set_name}' dataset from: {data_dir}")
    
    dataset = FloorSetPrimeDataset(data_dir, analog_bit, set_name)
    shuffle = True if set_name == 'train' else False

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=2, 
        drop_last=(set_name == 'train'),
        collate_fn=floorplan_collate_fn
    )
        
    while True:
        for data in loader:
            if data[0] is not None:
                yield data


def get_one_hot(x, z):
    """Creates a one-hot encoded vector."""
    return np.eye(z)[x]

class FloorSetPrimeDataset(Dataset):
    def __init__(self, data_dir, analog_bit, set_name, max_num_points=200):
        super().__init__()
        self.analog_bit = analog_bit
        self.set_name = set_name
        self.max_num_points = max_num_points
        self.num_coords = 2
        
        # Use the helper FloorplanDataset to handle file reading
        self.raw_dataset = FloorplanDatasetPrime(data_dir, set_name)

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        try:
            data_sample = self.raw_dataset[idx]
        except Exception as e:
            print(f"Warning: Skipping corrupted sample at index {idx}. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))

        fp_sol = data_sample['fp_sol'] 
        b2b_connectivity = data_sample['b2b_connectivity']

        all_corners_data = []
        corner_bounds = []
        num_points_so_far = 0

        for i, block_corners_tensor in enumerate(fp_sol):
            block_corners = block_corners_tensor.numpy()
            block_corners = block_corners[np.all(block_corners != -1, axis=1)]
            
            num_corners = len(block_corners)
            if num_corners == 0 or num_corners >= 32:
                continue

            block_corners = (block_corners - 0.5) * 2.0

            block_type_onehot = np.repeat(np.array([get_one_hot(0, 25)]), num_corners, 0)
            block_index_onehot = np.repeat(np.array([get_one_hot(i, 32)]), num_corners, 0)
            corner_index_onehot = np.array([get_one_hot(c, 32) for c in range(num_corners)])
            padding_mask = np.ones((num_corners, 1))
            connections = np.array([[j, (j + 1) % num_corners] for j in range(num_corners)])
            connections += num_points_so_far
            
            corner_bounds.append([num_points_so_far, num_points_so_far + num_corners])
            num_points_so_far += num_corners

            block_data = np.concatenate(
                (block_corners, block_type_onehot, corner_index_onehot, block_index_onehot, padding_mask, connections), 
                axis=1
            )
            all_corners_data.append(block_data)

        if not all_corners_data:
            return self.__getitem__((idx + 1) % len(self))

        house_layout = np.concatenate(all_corners_data, axis=0)
        num_total_points = len(house_layout)

        if num_total_points > self.max_num_points:
            return self.__getitem__((idx + 1) % len(self))
            
        padding_amount = self.max_num_points - num_total_points
        padding = np.zeros((padding_amount, house_layout.shape[1]))
        padding[:, self.num_coords + 25 + 32 + 32] = 0
        
        house_layout = np.concatenate((house_layout, padding), axis=0)

        num_blocks = len(corner_bounds)
        adj_matrix = np.zeros((num_blocks, num_blocks))
        for edge in b2b_connectivity:
            u, v, _ = edge.numpy().astype(int)
            if u < num_blocks and v < num_blocks:
                adj_matrix[u, v] = 1
                adj_matrix[v, u] = 1

        door_mask = np.ones((self.max_num_points, self.max_num_points))
        self_mask = np.ones((self.max_num_points, self.max_num_points))

        for i in range(num_blocks):
            start_i, end_i = corner_bounds[i]
            self_mask[start_i:end_i, start_i:end_i] = 0
            for j in range(i + 1, num_blocks):
                if adj_matrix[i, j] == 1:
                    start_j, end_j = corner_bounds[j]
                    door_mask[start_i:end_i, start_j:end_j] = 0
                    door_mask[start_j:end_j, start_i:end_i] = 0
        
        gen_mask = np.ones((self.max_num_points, self.max_num_points))
        gen_mask[:num_total_points, :num_total_points] = 0

        corner_coordinates = house_layout[:, :self.num_coords]
        
        cond = {
            'door_mask': door_mask.astype(np.float32),
            'self_mask': self_mask.astype(np.float32),
            'gen_mask': gen_mask.astype(np.float32),
            'room_types': house_layout[:, self.num_coords : self.num_coords + 25],
            'corner_indices': house_layout[:, self.num_coords + 25 : self.num_coords + 25 + 32],
            'room_indices': house_layout[:, self.num_coords + 25 + 32 : self.num_coords + 25 + 32 + 32],
            'src_key_padding_mask': 1 - house_layout[:, self.num_coords + 25 + 32 + 32],
            'connections': house_layout[:, self.num_coords + 25 + 32 + 32 + 1 :],
            'graph': b2b_connectivity.numpy()
        }

        if self.set_name == 'train':
            rotation = random.randint(0, 3)
            if rotation == 1:
                corner_coordinates[:, [0, 1]] = corner_coordinates[:, [1, 0]]
                corner_coordinates[:, 0] = -corner_coordinates[:, 0]
            elif rotation == 2:
                corner_coordinates[:, [0, 1]] = -corner_coordinates[:, [0, 1]]
            elif rotation == 3:
                corner_coordinates[:, [0, 1]] = corner_coordinates[:, [1, 0]]
                corner_coordinates[:, 1] = -corner_coordinates[:, 1]

        if self.analog_bit:
            ONE_HOT_RES = 256
            arr_onehot = np.zeros((self.max_num_points, 16))
            xs = ((corner_coordinates[:, 0] + 1) * (ONE_HOT_RES / 2)).astype(int)
            ys = ((corner_coordinates[:, 1] + 1) * (ONE_HOT_RES / 2)).astype(int)
            xs = np.clip(xs, 0, 255)
            ys = np.clip(ys, 0, 255)
            xs_bin = np.array([list(map(int, list(np.binary_repr(x, width=8)))) for x in xs])
            ys_bin = np.array([list(map(int, list(np.binary_repr(y, width=8)))) for y in ys])
            arr_onehot = np.concatenate([xs_bin, ys_bin], axis=1)
            arr_onehot[arr_onehot == 0] = -1
            corner_coordinates = arr_onehot

        return corner_coordinates.astype(np.float32).T, cond













# """
# This script is adapted from rplanhg_datasets.py to load and process the FloorSet-Prime dataset
# for use with the house-diffusion model.
# """
# import os
# import random
# import numpy as np
# import torch as th
# from torch.utils.data import DataLoader, Dataset
# from prime_dataset import FloorplanDatasetPrime # Import the helper class

# def floorplan_collate_fn(batch):
#     """
#     Custom collate function to correctly batch the output of our FloorSetPrimeDataset.
#     It handles the tuple of (coordinates, conditioning_dictionary).
#     """
#     # Filter out None values which can occur if a sample fails to load
#     batch = [b for b in batch if b is not None]
#     if not batch:
#         return None, None

#     coords_list = [item[0] for item in batch]
#     cond_dict_list = [item[1] for item in batch]

#     batched_coords = np.stack(coords_list, axis=0)
#     batched_coords = th.from_numpy(batched_coords)

#     batched_cond = {}
#     if cond_dict_list:
#         keys = cond_dict_list[0].keys()
#         for key in keys:
#             items = [d[key] for d in cond_dict_list]
#             if isinstance(items[0], np.ndarray):
#                 stacked_items = np.stack(items, axis=0)
#                 batched_cond[key] = th.from_numpy(stacked_items)
#             elif isinstance(items[0], torch.Tensor):
#                  batched_cond[key] = th.stack(items, dim=0)
#             else:
#                 batched_cond[key] = items

#     return batched_coords, batched_cond


# def load_floorset_data(
#     batch_size,
#     analog_bit,
#     set_name='train',
#     # --- EDIT: Updated the default path to your dataset location ---
#     data_dir = '/home/dqk5620/research/floorplanning/floorset_housediffusion/PrimeTensorData'
# ):
#     """
#     Creates a data loader for the FloorSet-Prime dataset.
#     """
#     print(f"Loading FloorSet-Prime '{set_name}' dataset from: {data_dir}")
    
#     dataset = FloorSetPrimeDataset(data_dir, analog_bit, set_name)
#     shuffle = True if set_name == 'train' else False

#     loader = DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         shuffle=shuffle, 
#         num_workers=2, 
#         drop_last=(set_name == 'train'),
#         collate_fn=floorplan_collate_fn
#     )
        
#     # This generator loop is necessary for the training script's structure
#     while True:
#         for data in loader:
#             if data[0] is not None:
#                 yield data


# def get_one_hot(x, z):
#     """Creates a one-hot encoded vector."""
#     return np.eye(z)[x]

# class FloorSetPrimeDataset(Dataset):
#     def __init__(self, data_dir, analog_bit, set_name, max_num_points=200):
#         super().__init__()
#         self.analog_bit = analog_bit
#         self.set_name = set_name
#         self.max_num_points = max_num_points
#         self.num_coords = 2
        
#         # Use the helper FloorplanDataset to handle file reading
#         self.raw_dataset = FloorplanDatasetPrime(data_dir, set_name)

#     def __len__(self):
#         return len(self.raw_dataset)

#     def __getitem__(self, idx):
#         try:
#             data_sample = self.raw_dataset[idx]
#         except Exception as e:
#             print(f"Warning: Skipping corrupted sample at index {idx}. Error: {e}")
#             return self.__getitem__((idx + 1) % len(self))

#         fp_sol = data_sample['fp_sol'] 
#         b2b_connectivity = data_sample['b2b_connectivity']

#         all_corners_data = []
#         corner_bounds = []
#         num_points_so_far = 0

#         for i, block_corners_tensor in enumerate(fp_sol):
#             block_corners = block_corners_tensor.numpy()
#             block_corners = block_corners[np.all(block_corners != -1, axis=1)]
            
#             num_corners = len(block_corners)
#             if num_corners == 0: continue

#             block_corners = (block_corners - 0.5) * 2.0

#             block_type_onehot = np.repeat(np.array([get_one_hot(0, 25)]), num_corners, 0)
#             block_index_onehot = np.repeat(np.array([get_one_hot(i, 32)]), num_corners, 0)
#             corner_index_onehot = np.array([get_one_hot(c, 32) for c in range(num_corners)])
#             padding_mask = np.ones((num_corners, 1))
#             connections = np.array([[j, (j + 1) % num_corners] for j in range(num_corners)])
#             connections += num_points_so_far
            
#             corner_bounds.append([num_points_so_far, num_points_so_far + num_corners])
#             num_points_so_far += num_corners

#             block_data = np.concatenate(
#                 (block_corners, block_type_onehot, corner_index_onehot, block_index_onehot, padding_mask, connections), 
#                 axis=1
#             )
#             all_corners_data.append(block_data)

#         if not all_corners_data:
#             return self.__getitem__((idx + 1) % len(self))

#         house_layout = np.concatenate(all_corners_data, axis=0)
#         num_total_points = len(house_layout)

#         if num_total_points > self.max_num_points:
#             return self.__getitem__((idx + 1) % len(self))
            
#         padding_amount = self.max_num_points - num_total_points
#         padding = np.zeros((padding_amount, house_layout.shape[1]))
#         padding[:, self.num_coords + 25 + 32 + 32] = 0
        
#         house_layout = np.concatenate((house_layout, padding), axis=0)

#         num_blocks = len(corner_bounds)
#         adj_matrix = np.zeros((num_blocks, num_blocks))
#         for edge in b2b_connectivity:
#             u, v, _ = edge.numpy().astype(int)
#             if u < num_blocks and v < num_blocks:
#                 adj_matrix[u, v] = 1
#                 adj_matrix[v, u] = 1

#         door_mask = np.ones((self.max_num_points, self.max_num_points))
#         self_mask = np.ones((self.max_num_points, self.max_num_points))

#         for i in range(num_blocks):
#             start_i, end_i = corner_bounds[i]
#             self_mask[start_i:end_i, start_i:end_i] = 0
#             for j in range(i + 1, num_blocks):
#                 if adj_matrix[i, j] == 1:
#                     start_j, end_j = corner_bounds[j]
#                     door_mask[start_i:end_i, start_j:end_j] = 0
#                     door_mask[start_j:end_j, start_i:end_i] = 0
        
#         gen_mask = np.ones((self.max_num_points, self.max_num_points))
#         gen_mask[:num_total_points, :num_total_points] = 0

#         corner_coordinates = house_layout[:, :self.num_coords]
        
#         cond = {
#             'door_mask': door_mask.astype(np.float32),
#             'self_mask': self_mask.astype(np.float32),
#             'gen_mask': gen_mask.astype(np.float32),
#             'room_types': house_layout[:, self.num_coords : self.num_coords + 25],
#             'corner_indices': house_layout[:, self.num_coords + 25 : self.num_coords + 25 + 32],
#             'room_indices': house_layout[:, self.num_coords + 25 + 32 : self.num_coords + 25 + 32 + 32],
#             'src_key_padding_mask': 1 - house_layout[:, self.num_coords + 25 + 32 + 32],
#             'connections': house_layout[:, self.num_coords + 25 + 32 + 32 + 1 :],
#             'graph': b2b_connectivity.numpy()
#         }

#         if self.set_name == 'train':
#             rotation = random.randint(0, 3)
#             if rotation == 1:
#                 corner_coordinates[:, [0, 1]] = corner_coordinates[:, [1, 0]]
#                 corner_coordinates[:, 0] = -corner_coordinates[:, 0]
#             elif rotation == 2:
#                 corner_coordinates[:, [0, 1]] = -corner_coordinates[:, [0, 1]]
#             elif rotation == 3:
#                 corner_coordinates[:, [0, 1]] = corner_coordinates[:, [1, 0]]
#                 corner_coordinates[:, 1] = -corner_coordinates[:, 1]

#         if self.analog_bit:
#             ONE_HOT_RES = 256
#             arr_onehot = np.zeros((self.max_num_points, 16))
#             xs = ((corner_coordinates[:, 0] + 1) * (ONE_HOT_RES / 2)).astype(int)
#             ys = ((corner_coordinates[:, 1] + 1) * (ONE_HOT_RES / 2)).astype(int)
#             xs = np.clip(xs, 0, 255)
#             ys = np.clip(ys, 0, 255)
#             xs_bin = np.array([list(map(int, list(np.binary_repr(x, width=8)))) for x in xs])
#             ys_bin = np.array([list(map(int, list(np.binary_repr(y, width=8)))) for y in ys])
#             arr_onehot = np.concatenate([xs_bin, ys_bin], axis=1)
#             arr_onehot[arr_onehot == 0] = -1
#             corner_coordinates = arr_onehot

#         return corner_coordinates.astype(np.float32).T, cond









# """
# This script is adapted from rplanhg_datasets.py to load and process the FloorSet-Prime dataset
# for use with the house-diffusion model.
# """
# import os
# import random
# import numpy as np
# import torch as th
# from torch.utils.data import DataLoader, Dataset
# from prime_dataset import FloorplanDatasetPrime, floorplan_collate # Assuming prime_dataset.py is in the python path

# def load_floorset_data(
#     batch_size,
#     analog_bit,
#     set_name='train',
#     data_dir = '../datasets/floorset' # Base directory for the FloorSet dataset
# ):
#     """
#     Creates a data loader for the FloorSet-Prime dataset.
#     """
#     print(f"Loading FloorSet-Prime dataset from: {data_dir}")
    
#     # The FloorSet loader uses a 'root' argument for the dataset path
#     dataset = FloorSetPrimeDataset(data_dir, analog_bit, set_name)

#     if set_name == 'train':
#         loader = DataLoader(
#             dataset, 
#             batch_size=batch_size, 
#             shuffle=True, 
#             num_workers=2, 
#             drop_last=True,
#             collate_fn=floorplan_collate_fn # Use the provided collate function
#         )
#     else: # For 'eval' or 'test'
#         loader = DataLoader(
#             dataset, 
#             batch_size=batch_size, 
#             shuffle=False, 
#             num_workers=2, 
#             drop_last=False,
#             collate_fn=floorplan_collate_fn
#         )
        
#     while True:
#         yield from loader

# def get_one_hot(x, z):
#     """Creates a one-hot encoded vector."""
#     return np.eye(z)[x]

# class FloorSetPrimeDataset(Dataset):
#     def __init__(self, data_dir, analog_bit, set_name, max_num_points=200):
#         super().__init__()
#         self.base_dir = data_dir
#         self.analog_bit = analog_bit
#         self.set_name = set_name
#         self.max_num_points = max_num_points
#         self.num_coords = 2
        
#         # Use the original FloorSet dataset loader
#         # This assumes the dataset is located at the specified path
#         self.floorset_dataset = FloorplanDatasetPrime(self.base_dir)

#     def __len__(self):
#         return len(self.floorset_dataset)

#     def __getitem__(self, idx):
#         # Retrieve a sample using the original FloorSet loader
#         # The collate function is handled by the DataLoader, so we get a single sample dict here
#         data_sample = self.floorset_dataset[idx]
        
#         fp_sol = data_sample['fp_sol'] # Polygon vertices for each block
#         b2b_connectivity = data_sample['b2b_connectivity'] # Block-to-block connections

#         all_corners = []
#         corner_bounds = []
#         num_points_so_far = 0

#         # Process each block (room) in the floorplan
#         for i, block_corners_tensor in enumerate(fp_sol):
#             # Convert tensor to numpy array
#             block_corners = block_corners_tensor.numpy()
            
#             # Filter out padding points (often marked as -1 or a large number in some datasets)
#             block_corners = block_corners[np.all(block_corners != -1, axis=1)]
            
#             num_corners = len(block_corners)
#             if num_corners == 0:
#                 continue

#             # --- Normalize coordinates to [-1, 1] ---
#             # FloorSet coordinates are typically in a normalized space already, but we ensure they are centered and scaled.
#             # This step might need adjustment based on the exact coordinate range of FloorSet.
#             # Assuming coordinates are in [0, 1], we shift to [-0.5, 0.5] and scale to [-1, 1]
#             block_corners = (block_corners - 0.5) * 2.0

#             # --- Prepare conditioning information ---
#             # Block type: FloorSet doesn't have semantic types, so we use a single type for all blocks.
#             block_type_onehot = np.repeat(np.array([get_one_hot(0, 25)]), num_corners, 0) # Using 25 to match rplan's dimension
            
#             # Block index and corner index
#             block_index_onehot = np.repeat(np.array([get_one_hot(i, 32)]), num_corners, 0)
#             corner_index_onehot = np.array([get_one_hot(c, 32) for c in range(num_corners)])
            
#             # Padding mask (1 for real data, 0 for padding)
#             padding_mask = np.ones((num_corners, 1))
            
#             # Connections (for drawing/post-processing, similar to rplan)
#             connections = np.array([[j, (j + 1) % num_corners] for j in range(num_corners)])
#             connections += num_points_so_far
            
#             # Store bounds for creating attention masks
#             corner_bounds.append([num_points_so_far, num_points_so_far + num_corners])
#             num_points_so_far += num_corners

#             # Combine all information for this block
#             block_data = np.concatenate(
#                 (block_corners, block_type_onehot, corner_index_onehot, block_index_onehot, padding_mask, connections), 
#                 axis=1
#             )
#             all_corners.append(block_data)

#         # Concatenate all blocks into a single sequence
#         if not all_corners: # Handle empty floorplans
#             return self.__getitem__((idx + 1) % len(self))

#         house_layout = np.concatenate(all_corners, axis=0)

#         # --- Padding ---
#         num_total_points = len(house_layout)
#         if num_total_points > self.max_num_points:
#             # If a floorplan exceeds the max points, you might skip it or truncate it.
#             # Skipping is safer to avoid losing information.
#             return self.__getitem__((idx + 1) % len(self))
            
#         padding_amount = self.max_num_points - num_total_points
#         # The number of columns should match the concatenated block_data
#         padding = np.zeros((padding_amount, house_layout.shape[1]))
        
#         # Set padding mask for padded elements to 0
#         padding[:, self.num_coords + 25 + 32 + 32] = 0 
        
#         house_layout = np.concatenate((house_layout, padding), axis=0)

#         # --- Build Graph and Attention Masks ---
#         num_blocks = len(corner_bounds)
#         adj_matrix = np.zeros((num_blocks, num_blocks))
#         for edge in b2b_connectivity:
#             u, v, _ = edge.numpy().astype(int)
#             if u < num_blocks and v < num_blocks:
#                 adj_matrix[u, v] = 1
#                 adj_matrix[v, u] = 1

#         door_mask = np.ones((self.max_num_points, self.max_num_points))
#         self_mask = np.ones((self.max_num_points, self.max_num_points))

#         for i in range(num_blocks):
#             start_i, end_i = corner_bounds[i]
#             # Self-mask: allow attention within a block
#             self_mask[start_i:end_i, start_i:end_i] = 0
#             for j in range(i + 1, num_blocks):
#                 if adj_matrix[i, j] == 1:
#                     start_j, end_j = corner_bounds[j]
#                     # Door-mask: allow attention between connected blocks
#                     door_mask[start_i:end_i, start_j:end_j] = 0
#                     door_mask[start_j:end_j, start_i:end_i] = 0
        
#         gen_mask = np.ones((self.max_num_points, self.max_num_points))
#         gen_mask[:num_total_points, :num_total_points] = 0

#         # --- Final Assembly ---
#         corner_coordinates = house_layout[:, :self.num_coords]
        
#         cond = {
#             'door_mask': door_mask.astype(np.float32),
#             'self_mask': self_mask.astype(np.float32),
#             'gen_mask': gen_mask.astype(np.float32),
#             'room_types': house_layout[:, self.num_coords : self.num_coords + 25],
#             'corner_indices': house_layout[:, self.num_coords + 25 : self.num_coords + 25 + 32],
#             'room_indices': house_layout[:, self.num_coords + 25 + 32 : self.num_coords + 25 + 32 + 32],
#             'src_key_padding_mask': 1 - house_layout[:, self.num_coords + 25 + 32 + 32],
#             'connections': house_layout[:, self.num_coords + 25 + 32 + 32 + 1 :],
#             'graph': b2b_connectivity.numpy() # Keep the original graph info
#         }

#         # Data augmentation for training set
#         if self.set_name == 'train':
#             rotation = random.randint(0, 3)
#             if rotation == 1: # 90 degrees
#                 corner_coordinates[:, [0, 1]] = corner_coordinates[:, [1, 0]]
#                 corner_coordinates[:, 0] = -corner_coordinates[:, 0]
#             elif rotation == 2: # 180 degrees
#                 corner_coordinates[:, [0, 1]] = -corner_coordinates[:, [0, 1]]
#             elif rotation == 3: # 270 degrees
#                 corner_coordinates[:, [0, 1]] = corner_coordinates[:, [1, 0]]
#                 corner_coordinates[:, 1] = -corner_coordinates[:, 1]

#         if self.analog_bit:
#             # Convert to binary representation if needed
#             ONE_HOT_RES = 256
#             arr_onehot = np.zeros((self.max_num_points, 16)) - 1
#             xs = ((corner_coordinates[:, 0] + 1) * (ONE_HOT_RES / 2)).astype(int)
#             ys = ((corner_coordinates[:, 1] + 1) * (ONE_HOT_RES / 2)).astype(int)
            
#             # Clamp values to be within [0, 255]
#             xs = np.clip(xs, 0, 255)
#             ys = np.clip(ys, 0, 255)

#             xs_bin = np.array([np.binary_repr(x, width=8) for x in xs])
#             ys_bin = np.array([np.binary_repr(y, width=8) for y in ys])
            
#             xs_int = np.array([list(map(int, list(b))) for b in xs_bin])
#             ys_int = np.array([list(map(int, list(b))) for b in ys_bin])

#             arr_onehot = np.concatenate([xs_int, ys_int], axis=1)
#             arr_onehot[arr_onehot == 0] = -1
#             corner_coordinates = arr_onehot

#         return corner_coordinates.astype(np.float32).T, cond
















# # """
# # This script is adapted from rplanhg_datasets.py to load and process the FloorSet-Prime dataset
# # for use with the house-diffusion model.
# # """
# # import os
# # import random
# # import numpy as np
# # import torch as th
# # from torch.utils.data import DataLoader, Dataset
# # from prime_dataset import FloorplanDataset, floorplan_collate_fn # Assuming prime_dataset.py is in the python path

# # def load_floorset_data(
# #     batch_size,
# #     analog_bit,
# #     set_name='train',
# #     data_dir = '../datasets/floorset' # Base directory for the FloorSet dataset
# # ):
# #     """
# #     Creates a data loader for the FloorSet-Prime dataset.
# #     """
# #     print(f"Loading FloorSet-Prime dataset from: {data_dir}")
    
# #     # The FloorSet loader uses a 'root' argument for the dataset path
# #     dataset = FloorSetPrimeDataset(data_dir, analog_bit, set_name)

# #     if set_name == 'train':
# #         loader = DataLoader(
# #             dataset, 
# #             batch_size=batch_size, 
# #             shuffle=True, 
# #             num_workers=2, 
# #             drop_last=True,
# #             collate_fn=floorplan_collate_fn # Use the provided collate function
# #         )
# #     else: # For 'eval' or 'test'
# #         loader = DataLoader(
# #             dataset, 
# #             batch_size=batch_size, 
# #             shuffle=False, 
# #             num_workers=2, 
# #             drop_last=False,
# #             collate_fn=floorplan_collate_fn
# #         )
        
# #     while True:
# #         yield from loader

# # def get_one_hot(x, z):
# #     """Creates a one-hot encoded vector."""
# #     return np.eye(z)[x]

# # class FloorSetPrimeDataset(Dataset):
# #     def __init__(self, data_dir, analog_bit, set_name, max_num_points=200):
# #         super().__init__()
# #         self.base_dir = data_dir
# #         self.analog_bit = analog_bit
# #         self.set_name = set_name
# #         self.max_num_points = max_num_points
# #         self.num_coords = 2
        
# #         # Use the original FloorSet dataset loader
# #         # This assumes the dataset is located at the specified path
# #         self.floorset_dataset = FloorplanDataset(self.base_dir)

# #     def __len__(self):
# #         return len(self.floorset_dataset)

# #     def __getitem__(self, idx):
# #         # Retrieve a sample using the original FloorSet loader
# #         # The collate function is handled by the DataLoader, so we get a single sample dict here
# #         data_sample = self.floorset_dataset[idx]
        
# #         fp_sol = data_sample['fp_sol'] # Polygon vertices for each block
# #         b2b_connectivity = data_sample['b2b_connectivity'] # Block-to-block connections

# #         all_corners = []
# #         corner_bounds = []
# #         num_points_so_far = 0

# #         # Process each block (room) in the floorplan
# #         for i, block_corners_tensor in enumerate(fp_sol):
# #             # Convert tensor to numpy array
# #             block_corners = block_corners_tensor.numpy()
            
# #             # Filter out padding points (often marked as -1 or a large number in some datasets)
# #             block_corners = block_corners[np.all(block_corners != -1, axis=1)]
            
# #             num_corners = len(block_corners)
# #             if num_corners == 0:
# #                 continue

# #             # --- Normalize coordinates to [-1, 1] ---
# #             # FloorSet coordinates are typically in a normalized space already, but we ensure they are centered and scaled.
# #             # This step might need adjustment based on the exact coordinate range of FloorSet.
# #             # Assuming coordinates are in [0, 1], we shift to [-0.5, 0.5] and scale to [-1, 1]
# #             block_corners = (block_corners - 0.5) * 2.0

# #             # --- Prepare conditioning information ---
# #             # Block type: FloorSet doesn't have semantic types, so we use a single type for all blocks.
# #             block_type_onehot = np.repeat(np.array([get_one_hot(0, 25)]), num_corners, 0) # Using 25 to match rplan's dimension
            
# #             # Block index and corner index
# #             block_index_onehot = np.repeat(np.array([get_one_hot(i, 32)]), num_corners, 0)
# #             corner_index_onehot = np.array([get_one_hot(c, 32) for c in range(num_corners)])
            
# #             # Padding mask (1 for real data, 0 for padding)
# #             padding_mask = np.ones((num_corners, 1))
            
# #             # Connections (for drawing/post-processing, similar to rplan)
# #             connections = np.array([[j, (j + 1) % num_corners] for j in range(num_corners)])
# #             connections += num_points_so_far
            
# #             # Store bounds for creating attention masks
# #             corner_bounds.append([num_points_so_far, num_points_so_far + num_corners])
# #             num_points_so_far += num_corners

# #             # Combine all information for this block
# #             block_data = np.concatenate(
# #                 (block_corners, block_type_onehot, corner_index_onehot, block_index_onehot, padding_mask, connections), 
# #                 axis=1
# #             )
# #             all_corners.append(block_data)

# #         # Concatenate all blocks into a single sequence
# #         if not all_corners: # Handle empty floorplans
# #             return self.__getitem__((idx + 1) % len(self))

# #         house_layout = np.concatenate(all_corners, axis=0)

# #         # --- Padding ---
# #         num_total_points = len(house_layout)
# #         if num_total_points > self.max_num_points:
# #             # If a floorplan exceeds the max points, you might skip it or truncate it.
# #             # Skipping is safer to avoid losing information.
# #             return self.__getitem__((idx + 1) % len(self))
            
# #         padding_amount = self.max_num_points - num_total_points
# #         # The number of columns should match the concatenated block_data
# #         padding = np.zeros((padding_amount, house_layout.shape[1]))
        
# #         # Set padding mask for padded elements to 0
# #         padding[:, self.num_coords + 25 + 32 + 32] = 0 
        
# #         house_layout = np.concatenate((house_layout, padding), axis=0)

# #         # --- Build Graph and Attention Masks ---
# #         num_blocks = len(corner_bounds)
# #         adj_matrix = np.zeros((num_blocks, num_blocks))
# #         for edge in b2b_connectivity:
# #             u, v, _ = edge.numpy().astype(int)
# #             if u < num_blocks and v < num_blocks:
# #                 adj_matrix[u, v] = 1
# #                 adj_matrix[v, u] = 1

# #         door_mask = np.ones((self.max_num_points, self.max_num_points))
# #         self_mask = np.ones((self.max_num_points, self.max_num_points))

# #         for i in range(num_blocks):
# #             start_i, end_i = corner_bounds[i]
# #             # Self-mask: allow attention within a block
# #             self_mask[start_i:end_i, start_i:end_i] = 0
# #             for j in range(i + 1, num_blocks):
# #                 if adj_matrix[i, j] == 1:
# #                     start_j, end_j = corner_bounds[j]
# #                     # Door-mask: allow attention between connected blocks
# #                     door_mask[start_i:end_i, start_j:end_j] = 0
# #                     door_mask[start_j:end_j, start_i:end_i] = 0
        
# #         gen_mask = np.ones((self.max_num_points, self.max_num_points))
# #         gen_mask[:num_total_points, :num_total_points] = 0

# #         # --- Final Assembly ---
# #         corner_coordinates = house_layout[:, :self.num_coords]
        
# #         cond = {
# #             'door_mask': door_mask.astype(np.float32),
# #             'self_mask': self_mask.astype(np.float32),
# #             'gen_mask': gen_mask.astype(np.float32),
# #             'room_types': house_layout[:, self.num_coords : self.num_coords + 25],
# #             'corner_indices': house_layout[:, self.num_coords + 25 : self.num_coords + 25 + 32],
# #             'room_indices': house_layout[:, self.num_coords + 25 + 32 : self.num_coords + 25 + 32 + 32],
# #             'src_key_padding_mask': 1 - house_layout[:, self.num_coords + 25 + 32 + 32],
# #             'connections': house_layout[:, self.num_coords + 25 + 32 + 32 + 1 :],
# #             'graph': b2b_connectivity.numpy() # Keep the original graph info
# #         }

# #         # Data augmentation for training set
# #         if self.set_name == 'train':
# #             rotation = random.randint(0, 3)
# #             if rotation == 1: # 90 degrees
# #                 corner_coordinates[:, [0, 1]] = corner_coordinates[:, [1, 0]]
# #                 corner_coordinates[:, 0] = -corner_coordinates[:, 0]
# #             elif rotation == 2: # 180 degrees
# #                 corner_coordinates[:, [0, 1]] = -corner_coordinates[:, [0, 1]]
# #             elif rotation == 3: # 270 degrees
# #                 corner_coordinates[:, [0, 1]] = corner_coordinates[:, [1, 0]]
# #                 corner_coordinates[:, 1] = -corner_coordinates[:, 1]

# #         if self.analog_bit:
# #             # Convert to binary representation if needed
# #             ONE_HOT_RES = 256
# #             arr_onehot = np.zeros((self.max_num_points, 16)) - 1
# #             xs = ((corner_coordinates[:, 0] + 1) * (ONE_HOT_RES / 2)).astype(int)
# #             ys = ((corner_coordinates[:, 1] + 1) * (ONE_HOT_RES / 2)).astype(int)
            
# #             # Clamp values to be within [0, 255]
# #             xs = np.clip(xs, 0, 255)
# #             ys = np.clip(ys, 0, 255)

# #             xs_bin = np.array([np.binary_repr(x, width=8) for x in xs])
# #             ys_bin = np.array([np.binary_repr(y, width=8) for y in ys])
            
# #             xs_int = np.array([list(map(int, list(b))) for b in xs_bin])
# #             ys_int = np.array([list(map(int, list(b))) for b in ys_bin])

# #             arr_onehot = np.concatenate([xs_int, ys_int], axis=1)
# #             arr_onehot[arr_onehot == 0] = -1
# #             corner_coordinates = arr_onehot

# #         return corner_coordinates.astype(np.float32).T, cond

