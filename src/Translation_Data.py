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
