from torch.utils.data import Dataset
import torch


class Translation_Data(Dataset):

    def __init__(self, data, voc_input, voc_output):
        self.data = data
        self.voc_input = voc_input
        self.voc_output = voc_output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]["EN"]
        output_text = self.data.iloc[idx]["FR"]

        input_indices = self.voc_input.sentence_to_idx(input_text)
        output_indices = self.voc_output.sentence_to_idx(output_text)

        return (
            torch.tensor(input_indices, dtype=torch.long),
            torch.tensor(output_indices, dtype=torch.long),
        )

#! Stop
def collate(batch, special_idx):
    """
    - takes tensor batches
    - gets max lengths of the input and output tensors
    - adds <pad> to make all the same length
    - add masks to ignore <pad> 
    return: all tensors and masks as dictionary
    """
    batch_size = len(batch)
    input_len_list = [len(input) for input, _ in batch]
    output_len_list = [len(output) for _, output in batch]
    max_input_len = max(input_len_list)
    max_output_len = max(output_len_list)

    input_batch = torch.full((batch_size, max_input_len), special_idx, dtype=torch.long)
    output_batch = torch.full((batch_size, max_output_len), special_idx, dtype=torch.long)

    input_mask = torch.zeros((batch_size, max_input_len), dtype=torch.bool)
    output_mask = torch.zeros((batch_size, max_output_len), dtype=torch.bool)

    for i, (input_seq, output_seq) in enumerate(batch):
            input_len, output_len = len(input_seq), len(output_seq)
            input_batch[i, :input_len] = input_seq
            output_batch[i, :output_len] = output_seq
            input_mask[i, :input_len] = True
            output_mask[i, :output_len] = True
    
    return {
         "input": input_batch,
         "output": output_batch,
         "input_mask": input_mask,
         "output_mask": output_mask
    }
