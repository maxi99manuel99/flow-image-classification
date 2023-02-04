import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from queue import Queue
from pathlib import Path
from threading import Thread
from typing import Type, Tuple
import cv2

MALWARE = ["Cridex", "Tinba", "Shifu", "Miuref", "Neris", "Nsis-ay", "Geodo", "Zeus", "Virut", "Htbot"]

class MalwareDataset(Dataset):
    def __init__(self):
        pass 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_tensor, associated_class = self.data[idx]
        class_id = self.class_map[associated_class]
        class_id = torch.tensor([class_id])
        return img_tensor, class_id

class Malware14(MalwareDataset):
    def __init__(self, set_type="Train", preprocessing_type="preprocessed"):
        if preprocessing_type == "preprocessed":
            path = "/home/trageser/datasets/FLOWS_PREPROCESSED_MID32_MAXID1000/14Classes/"
        elif preprocessing_type == "payload":
            path = "/home/trageser/datasets/FLOWS_PAYLOAD_MID32_MAXID1000/12Classes/"
        elif preprocessing_type == "pure":
            path = "/home/trageser/datasets/FLOWS_PURE_MID32_MAXID1000/14Classes/"

        path = path + set_type + "/"

        self.data = []
        self.class_map = {}
        self.classes = []

        file_queue = Queue()
        for file_path in Path(path).rglob('*.png'):
            file_queue.put(str(file_path.absolute()))

        for _ in range(300):
            thread = Thread(target=self.append_files_to_data, args=(file_queue, ))
            thread.daemon = True
            thread.start()
        
        file_queue.join()
        self.classes = set(self.classes)
        self.classes = list(self.classes)
        self.classes.sort()
    
        for i, class_name in enumerate(self.classes):
            self.class_map[class_name] = i

    def append_files_to_data(self, file_queue: Queue):
        while not file_queue.empty():
            file_path = file_queue.get()
            associated_class = file_path.split("/")[-2]
            if associated_class == "Neris" or associated_class == "Virut":
                associated_class = "NerisVirut"
            self.classes.append(associated_class)           
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img_tensor = torch.from_numpy(img) / 255.
            self.data.append((img_tensor, associated_class))
            file_queue.task_done()
    

class MalwareBinary(MalwareDataset):
    def __init__(self, set_type="Train", preprocessing_type="preprocessed"):
        if preprocessing_type == "preprocessed":
            path = "/home/trageser/datasets/FLOWS_PREPROCESSED_MID32_MAXID1000/14Classes/"
        elif preprocessing_type == "payload":
            path = "/home/trageser/datasets/FLOWS_PAYLOAD_MID32_MAXID1000/12Classes/"
        elif preprocessing_type == "pure":
            path = "/home/trageser/datasets/FLOWS_PURE_MID32_MAXID1000/14Classes/"

        path = path + set_type + "/"

        self.data = []
        self.class_map = {}

        file_queue = Queue()
        for file_path in Path(path).rglob('*.png'):
            file_queue.put(str(file_path.absolute()))

        for _ in range(300):
            thread = Thread(target=self.append_files_to_data, args=(file_queue, ))
            thread.daemon = True
            thread.start()
        
        file_queue.join()
        self.classes = ["Benign", "Malware"]
    
        for i, class_name in enumerate(self.classes):
            self.class_map[class_name] = i

    def append_files_to_data(self, file_queue: Queue):
        while not file_queue.empty():
            file_path = file_queue.get()
            associated_class = file_path.split("/")[-2]
            if associated_class in MALWARE:
                associated_class = "Malware"
            else:
                associated_class = "Benign"
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img_tensor = torch.from_numpy(img) / 255.
            self.data.append((img_tensor, associated_class))
            file_queue.task_done()

def collate_padded_batches(batch):
    dim1_lenghts = [item[0].shape[0] for item in batch]
    dim2_lenghts = [item[0].shape[1] for item in batch]
    max_dim1 = max(dim1_lenghts)
    max_dim2 = max(dim2_lenghts)
    data = torch.zeros((len(batch), max_dim1, max_dim2))
    for i in range(len(batch)):
        item = batch[i][0]
        data[i][:dim1_lenghts[i],:dim2_lenghts[i]] = item

    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def initialize_weights_for_balanced_set(dataset: MalwareDataset):
    dataset_size = len(dataset)
    class_counts = {}
    dataset_weights = [0] * len(dataset)

    for item in dataset:
        associated_class = item[1].item()
        if associated_class in class_counts.keys():
            class_counts[associated_class] = class_counts[associated_class] + 1
        else:
            class_counts[associated_class] = 1

    class_weights = [0] * len(class_counts)
    for key in class_counts.keys():
        class_weights[key] = dataset_size / class_counts[key]

    for i, item in enumerate(dataset):
        associated_class = item[1].item()
        dataset_weights[i] = class_weights[associated_class]
    
    return dataset_weights


def get_dataloader(dataset: MalwareDataset, batch_size:int, over_sampling:bool=False) -> DataLoader:
    if not over_sampling:
        return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_padded_batches, shuffle=True, pin_memory=True)
    else:
        weights = torch.DoubleTensor(initialize_weights_for_balanced_set(dataset))
        sampler = WeightedRandomSampler(weights, len(dataset))
        return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_padded_batches, sampler=sampler, pin_memory=True)

