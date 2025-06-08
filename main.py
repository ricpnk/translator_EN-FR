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
import random

from torch.utils.tensorboard import SummaryWriter

from src.vocab import Vocab
from src.translation_data import Translation_Data, collate
from src.models import Encoder, Decoder, Seq2Seq

# Model hyperparameters
BATCH_SIZE = 256
EMBEDDING_DIM = 512
HIDDEN_DIM = 512
N_DIM = 3
MAX_LEN = 11
LEARNRATE = 0.001
TEACHER_RATE = 0.5
NUM_EPOCHS = 500
DROP_RATE = 0.5
PATIENCE = 20

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

    # todo ask if checkpoints should be loaded (deactivated)
    # load_checkpoint = input("Do you want to load a checkpoint? (y/n): ")
    global LOAD_CHECKPOINT
    LOAD_CHECKPOINT = False
    #if load_checkpoint == "y":
    #    LOAD_CHECKPOINT = True

    # todo read in the data and train-test split
    data = pd.read_csv("data/rec05_small_en_fr.csv")
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # todo build vocabulary
    en_counter = Counter()
    fr_counter = Counter()
    for sentence in data["EN"]:
        en_counter.update(sentence.split())
    for sentence in data["FR"]:
        fr_counter.update(sentence.split())
    vocabulary_en = Vocab()
    vocabulary_fr = Vocab()
    vocabulary_en.build(en_counter)
    vocabulary_fr.build(fr_counter)

    # todo test the vocab
    print("Vocab Test:")
    for sent in train_data["EN"][:5]:
        idxs    = vocabulary_en.sentence_to_idx(sent)
        tokens  = vocabulary_en.idx_to_sentence(idxs)
        print("INPUT:", sent.split())
        print("OUTPUT:", tokens)
        print("-"*50)

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
    
    # todo Initialize training components
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary_fr.word2idx["<pad>"], label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNRATE, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=2
    )

    # todo Shape testing
    batch = next(iter(train_loader))
    print("Batch input shape:",  batch["input"].shape)
    print("Batch output shape:", batch["output"].shape)
    print("Sentence 0: ", batch["input"][0])
    print("Input mask sum   :", batch["input_mask"][0].sum().item())
    print()

    # todo Model training
    writer = SummaryWriter(log_dir=f"runs/{TIMESTAMP}")
    training(model, criterion, optimizer, scheduler, train_loader, test_loader, device, vocabulary_en, vocabulary_fr, writer)

    #todo Save the final model
    torch.save(model.state_dict(), f"{MODEL_PATH}/model_{TIMESTAMP}.pth")
    print(f"Model saved to {MODEL_PATH}/model_{TIMESTAMP}.pth")

    writer.close()



#! Functions for training and evaluation

def training(model, criterion, optimizer, scheduler, train_loader, test_loader, device, vocabulary_en, vocabulary_fr, writer):
    model.train()
    best_bleu = 0.0
    epoch_patience = 0
    start_epoch = 0

    # load the checkpoint if it exists (deactivated)
    if LOAD_CHECKPOINT:
        checkpoint = torch.load("saved_models/checkpoints/checkpoint.pth")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1

    # train loop
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

        # calculate the average loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        avg_test_loss, bleu_score = evaluation(model, criterion, epoch, test_loader, device, vocabulary_en, vocabulary_fr)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/test", avg_test_loss, epoch)
        writer.add_scalar("BLEU", bleu_score, epoch)

        # Print results
        print("\n" + "=" * 50)                                  
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"BLEU Score: {bleu_score:.4f}")
        print("-" * 50 + "\n")

        # Print example translations
        print_example_translations(test_loader, model, vocabulary_en, vocabulary_fr, device)

        # Visualize attention every 5 epochs
        if epoch % 2 == 0 and epoch > 0:
            visualize_attention(test_loader, model, vocabulary_en, vocabulary_fr, device, epoch)

        # Early stopping check
        if bleu_score > best_bleu + 1e-4:
            best_bleu = bleu_score
            epoch_patience = 0
        else:
            epoch_patience += 1
            if epoch_patience >= PATIENCE:
                print("Early stopping triggered. No improvement in BLEU score for 5 epochs.")
                break
            
        scheduler.step(avg_test_loss)

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, avg_train_loss)

def evaluation(model, criterion, epoch, test_loader, device, vocabulary_en, vocabulary_fr):
    model.eval()
    test_loss = 0
    predictions = []
    references = []
    bleu = evaluate.load("bleu")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1} [Test]", unit="batch"):
            input = batch["input"].to(device)
            output = batch["output"].to(device)

            outputs = model(input, output, teacher_forcing_rate=0.0)
            out_dim = outputs.size(-1)

            loss = criterion(
                outputs[:, 1:, :].reshape(-1, out_dim), output[:, 1:].reshape(-1)
            )
            test_loss += loss.item()

            # Generate predictions for BLEU score
            batch_size = input.size(0)
            for i in range(batch_size):
                single_sentence = input[i].unsqueeze(0)
                prediction_sentence = decode(single_sentence, model, vocabulary_en, vocabulary_fr, device)
                
                reference_idx = batch["output"][i].tolist()
                reference_tokens = vocabulary_fr.idx_to_sentence(reference_idx)
                reference_tokens = [t for t in reference_tokens if t not in ("<pad>", "<sos>", "<eos>", "<unk>")]
                reference_sentence = " ".join(reference_tokens)

                predictions.append(prediction_sentence)
                references.append([reference_sentence])

    avg_test_loss = test_loss / len(test_loader)

    bleu_results = bleu.compute(predictions=predictions, references=references)
    bleu_score = bleu_results["bleu"]

    return avg_test_loss, bleu_score

def decode(sentence_tensor: torch.Tensor,
          model: torch.nn.Module,
          vocab_input: Vocab,    
          vocab_output: Vocab, 
          device: torch.device,
          max_len: int = MAX_LEN,
          return_attention: bool = False):

    model.eval()
    pad_idx = vocab_input.word2idx["<pad>"]
    src_mask = (sentence_tensor != pad_idx)

    with torch.no_grad():
        enc_outputs, hidden = model.encoder(sentence_tensor)

    sos_idx = vocab_output.word2idx["<sos>"]
    input_token = torch.LongTensor([sos_idx]).to(device)
    generated_indices = []
    all_attn = [] if return_attention else None

    for _ in range(max_len - 1):
        with torch.no_grad():
            prediction, hidden, attn_weights = model.decoder(
                input_token, hidden, enc_outputs, src_mask
            )
        
        if return_attention:
            all_attn.append(attn_weights.squeeze(0).cpu())
            
        top1 = prediction.argmax(1).item()
        if top1 == vocab_output.word2idx["<eos>"]:
            break

        generated_indices.append(top1)
        input_token = torch.LongTensor([top1]).to(device)

    if return_attention:
        t_steps = len(all_attn)
        if t_steps < max_len - 1:
            filler = torch.zeros((max_len - 1 - t_steps, enc_outputs.size(1)))
            all_attn.extend([filler[i] for i in range(filler.size(0))])
        all_attn = torch.stack(all_attn, dim=0)
        return generated_indices, all_attn

    generated_tokens = vocab_output.idx_to_sentence(generated_indices)
    return " ".join(generated_tokens)

def print_example_translations(test_loader, model, vocabulary_en, vocabulary_fr, device):
    print("Example translations:")
    sample_idx = random.sample(range(len(test_loader.dataset)), 5)
    for idx in sample_idx:
        input_sentence, output_sentence = test_loader.dataset[idx]
        input_tensor = input_sentence.unsqueeze(0).to(device)
        pred_sentence = decode(input_tensor, model, vocabulary_en, vocabulary_fr, device)
        
        input_tokens = vocabulary_en.idx_to_sentence(input_sentence.tolist())
        output_tokens = vocabulary_fr.idx_to_sentence(output_sentence.tolist())
        output_tokens = [t for t in output_tokens if t not in ("<pad>", "<sos>", "<eos>", "<unk>")]
        
        print(f"Input: {' '.join(input_tokens)}")
        print(f"Output: {' '.join(output_tokens)}")
        print(f"Predicted: {pred_sentence}")
        print("-" * 50)
    print("=" * 50 + "\n")

def visualize_attention(test_loader, model, vocabulary_en, vocabulary_fr, device, epoch):
    for idx in range(min(3, len(test_loader.dataset))):
        input_indices = test_loader.dataset[idx][0].unsqueeze(0).to(device)
        gen_indices, attn_matrix = decode(input_indices, model, vocabulary_en, vocabulary_fr, device, return_attention=True)
        input_tokens = vocabulary_en.idx_to_sentence(input_indices.squeeze(0).cpu().tolist())
        output_indices = test_loader.dataset[idx][1]
        output_tokens = vocabulary_fr.idx_to_sentence(output_indices.tolist())
        output_tokens = [t for t in output_tokens if t not in ("<pad>", "<sos>", "<eos>", "<unk>")]

        # Plot attention heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn_matrix[:len(output_tokens), :len(input_tokens)],
                      cmap="viridis", aspect="auto", vmin=0.0, vmax= 0.3)

        ax.set_xticks(range(len(input_tokens)))
        ax.set_xticklabels(input_tokens, rotation=45, ha='right')
        ax.set_yticks(range(len(output_tokens)))
        ax.set_yticklabels(output_tokens, rotation=0)
        ax.set_xlabel("Source (EN)")
        ax.set_ylabel("Target (FR)")
        ax.set_title(f"Attention Heatmap - Epoch {epoch} Example {idx}")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()

        fig.savefig(f"{MODEL_PATH}/attention_plots/epoch_{epoch}_example_{idx}.png")
        plt.close(fig)

def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, f"{MODEL_PATH}/checkpoints/checkpoint.pth")

if __name__ == "__main__":
    main()