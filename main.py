import os
from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import evaluate
import matplotlib.pyplot as plt

from src.Vocab import Vocab
from src.Translation_Data import Translation_Data, collate
from src.models import Encoder, Decoder, Seq2Seq


# Hyperparameters
BATCH_SIZE = 128
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
N_DIM = 1
MAX_LEN = 12
LEARNRATE = 0.001
TEACHER_RATE = 0.5
NUM_EPOCHS = 50
DROP_RATE = 0.4
PATIENCE = 5
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = f"saved_models/model_{TIMESTAMP}"

def main():
    # todo bulid project structure
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH+"/attention_plots", exist_ok=True)
    os.makedirs(MODEL_PATH+"/checkpoints", exist_ok=True)

    # todo check if apple gpu is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: ", device)

    # todo ask if checkpoints should be loaded
    #! load_checkpoint = input("Do you want to load a checkpoint? (y/n): ")
    global LOAD_CHECKPOINT
    LOAD_CHECKPOINT = False
    #if load_checkpoint == "y":
    #    LOAD_CHECKPOINT = True

    # todo read in the data and train-test split
    data = pd.read_csv("data/rec05_small_en_fr.csv")
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # todo initialize counters for the vocabs
    en_counter = Counter()
    fr_counter = Counter()
    for sentence in data["EN"]:
        en_counter.update(sentence.split())
    for sentence in data["FR"]:
        fr_counter.update(sentence.split())

    # todo create vocabs for input and output
    vocabulary_en = Vocab()
    vocabulary_fr = Vocab()
    vocabulary_en.build(en_counter)
    vocabulary_fr.build(fr_counter)



    # todo initiate translator dataset
    train_translator_data = Translation_Data(train_data, vocabulary_en, vocabulary_fr)
    test_translator_data = Translation_Data(test_data, vocabulary_en, vocabulary_fr)

    train_loader = DataLoader(
        train_translator_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate(
            batch, special_idx=vocabulary_en.word2idx["<pad>"], max_len=MAX_LEN
        ),
    )
    test_loader = DataLoader(
        test_translator_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate(
            batch, special_idx=vocabulary_en.word2idx["<pad>"], max_len=MAX_LEN
        ),
    )

    # todo initiate the Objects for models
    encoder = Encoder(
        vocab_size=len(vocabulary_en),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        n_dim=N_DIM,
        dropout=DROP_RATE,
    ).to(device)

    pad_idx = vocabulary_en.word2idx["<pad>"]
    decoder = Decoder(
        vocab_size=len(vocabulary_fr),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        n_dim=N_DIM,
        dropout=DROP_RATE,
        pad_idx=pad_idx
    ).to(device)

    model = Seq2Seq(encoder, decoder, device).to(device)
    # Use French vocab's pad token for ignore_index
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary_fr.word2idx["<pad>"], label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNRATE, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    training(model, criterion, optimizer, scheduler, train_loader, test_loader, device, vocabulary_en, vocabulary_fr)

    # todo save the model
    torch.save(model.state_dict(), f"saved_models/model_{TIMESTAMP}/model_{TIMESTAMP}.pth")
    print(f"Model saved to saved_models/model_{TIMESTAMP}.pth")



def training(model, criterion, optimizer, scheduler, train_loader, test_loader, device, vocabulary_en, vocabulary_fr):
    model.train()
    start_epoch = 0
    best_bleu = 0.0
    epoch_patience = 0
    patience = PATIENCE

    # todo load the checkpoint if it exists
    if LOAD_CHECKPOINT:
        checkpoint = torch.load("saved_models/checkpoints/checkpoint.pth")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", unit="batch"):
            input = batch["input"].to(device)
            output = batch["output"].to(device)

            optimizer.zero_grad()
            outputs = model(input, output, teacher_forcing_rate=TEACHER_RATE)

            out_dim = outputs.size(-1)
            loss = criterion(
                outputs[:, 1:, :].reshape(-1, out_dim), output[:, 1:].reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_test_loss, bleu_score = evaluation(model, criterion, epoch, test_loader, device, vocabulary_en, vocabulary_fr)

        # print the results
        print("\n" + "=" * 50)                                  
        print(f"Epoche {epoch + 1}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Test Loss: {avg_test_loss}")
        print(f"BLEU Score: {bleu_score:.4f}")

        if bleu_score > best_bleu + 1e-4:
            best_bleu = bleu_score
            epoch_patience = 0
        else:
            epoch_patience += 1
            if epoch_patience >= patience:
                print("Early stopping triggered. No improvement in BLEU score for 5 epochs.")
                break
            
        scheduler.step(avg_test_loss)

        # save the checkpoint
        last_checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_train_loss,
        }
        torch.save(last_checkpoint, f"saved_models/model_{TIMESTAMP}/checkpoints/checkpoint.pth")


def evaluation(model, criterion, epoch, test_loader, device,vocabulary_en, vocabulary_fr):
    model.eval()
    test_loss = 0

    predictions = []
    references = []
    bleu = evaluate.load("bleu")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1} [Test]  ", unit="batch"):
            input = batch["input"].to(device)
            output = batch["output"].to(device)

            outputs = model(input, output, teacher_forcing_rate=0.0)
            out_dim = outputs.size(-1)

            loss = criterion(
                outputs[:, 1:, :].reshape(-1, out_dim), output[:, 1:].reshape(-1)
            )
            test_loss += loss.item()

            batch_size = input.size(0)
            for i in range(batch_size):
                single_sentence = input[i].unsqueeze(0)

                prediction_sentence = greedy_decode(single_sentence, 
                                                  model, 
                                                  vocab_input=vocabulary_en,
                                                  vocab_output=vocabulary_fr, 
                                                  device=device,
                                                  max_len=MAX_LEN)
                
                reference_idx = batch["output"][i].tolist()
                reference_tokens = vocabulary_fr.idx_to_sentence(reference_idx)
                # Remove special tokens
                reference_tokens = [t for t in reference_tokens if t not in ("<pad>", "<sos>", "<eos>", "<unk>")]
                reference_sentence = " ".join(reference_tokens)

                predictions.append(prediction_sentence)
                references.append([reference_sentence])


    avg_test_loss = test_loss / len(test_loader)

    bleu_results = bleu.compute(predictions=predictions, references=references)
    bleu_score = bleu_results["bleu"]

    return avg_test_loss, bleu_score


def greedy_decode(sentence_tensor: torch.Tensor,
                  model: torch.nn.Module,
                  vocab_input,    
                  vocab_output, 
                  max_len=MAX_LEN):

    model.eval()

    pad_idx = vocab_input.word2idx["<pad>"]
    src_mask = (sentence_tensor != pad_idx)  # Shape: (1, src_len)

    # Encoder: outputs (1, src_len, hid_dim), hidden (n_dim, 1, hid_dim)
    with torch.no_grad():
        enc_outputs, hidden = model.encoder(sentence_tensor)

    sos_idx = vocab_output.word2idx["<sos>"]
    input_token = torch.LongTensor([sos_idx]).to(device)  # Shape (1,)

    generated_indices = []

    for _ in range(max_len - 1):
        with torch.no_grad():
            # prediction: (1, vocab_size), new_hidden: (n_dim, 1, hid_dim)
            prediction, hidden, attn_weights = model.decoder(
                input_token, hidden, enc_outputs, src_mask
            )
        # prediction: (1, vocab_size)
        top1 = prediction.argmax(1).item()  # gibt Int
        
        if top1 == vocab_output.word2idx["<eos>"]:
            break

        generated_indices.append(top1)

        input_token = torch.LongTensor([top1]).to(device)


    generated_tokens = vocab_output.idx_to_sentence(generated_indices)
    generated_sentence = " ".join(generated_tokens)
    
    return generated_sentence



if __name__ == "__main__":
    main()