from torch.utils.data import Dataset
import torch


class Translation_Data(Dataset):

    def __init__(self, data, voc_input, voc_output):
        self.data = data
        self.voc_input = voc_input  # English vocabulary
        self.voc_output = voc_output  # French vocabulary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get English and French sentences
        input_text = self.data.iloc[idx]["EN"]
        output_text = self.data.iloc[idx]["FR"]

        # Convert text to indices
        input_indices = self.voc_input.sentence_to_idx(input_text)
        output_indices = self.voc_output.sentence_to_idx(output_text)

        return (
            torch.tensor(input_indices, dtype=torch.long),
            torch.tensor(output_indices, dtype=torch.long),
        )


def collate(batch, special_idx, max_len):
    
    batch_size = len(batch)

    # Initialize tensors with padding
    input_batch = torch.full((batch_size, max_len), special_idx, dtype=torch.long)
    output_batch = torch.full((batch_size, max_len), special_idx, dtype=torch.long)
    input_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    output_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    # Fill tensors with sequences
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
