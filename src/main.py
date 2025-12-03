from tqdm import tqdm
from constants import (
    SP_DIR,
    src_model_prefix,
    trg_model_prefix,
    seq_len,
    pad_id,
    learning_rate,
    device,
    num_epochs,
    ckpt_dir,
    start_epoch,
)
from constants import TRAIN_NAME, VALID_NAME
from custom_data import get_data_loader, pad_or_truncate
from transformer import Transformer
from torch import nn

import torch
import sys
import os
import numpy as np
import datetime
import sentencepiece as spm

class Manager:
    def __init__(self, is_train=True, ckpt_name=None):
        # Load vocabs
        print("Loading vocabs...")
        self.src_i2w = {}
        self.trg_i2w = {}

        with open(f"{SP_DIR}/{src_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split("\t")[0]
            self.src_i2w[i] = word

        with open(f"{SP_DIR}/{trg_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split("\t")[0]
            self.trg_i2w[i] = word

        print(
            f"The size of src vocab is {len(self.src_i2w)} and that of trg vocab is {len(self.trg_i2w)}."
        )

        # Load Transformer model & Adam optimizer
        print("Loading Transformer model & Adam optimizer...")
        self.model = Transformer(
            src_vocab_size=len(self.src_i2w), trg_vocab_size=len(self.trg_i2w)
        )
        if torch.cuda.device_count() > 1:
            print(f"Detecting {torch.cuda.device_count()} GPUs. Using DataParallel!")
            self.model = nn.DataParallel(self.model)
            self.model_core = self.model.module
        else:
            self.model_core = self.model
        self.model = self.model.to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_loss = sys.float_info.max
        self.start_epoch = start_epoch
        print(f"Starting from epoch {self.start_epoch}")

        if ckpt_name is not None:
            assert os.path.exists(f"{ckpt_dir}/{ckpt_name}"), (
                f"There is no checkpoint named {ckpt_name}."
            )

            print("Loading checkpoint...")
            checkpoint = torch.load(
                f"{ckpt_dir}/{ckpt_name}", map_location=device, weights_only=False
            )
            self.model_core.load_state_dict(checkpoint["model_state_dict"])
            self.optim.load_state_dict(checkpoint["optim_state_dict"])
            self.best_loss = checkpoint["loss"]

            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"]
                print(f"Resuming training from epoch {self.start_epoch}")
            else:
                print(
                    f"No epoch info in checkpoint, starting from epoch {self.start_epoch} (but with loaded weights)."
                )

        else:
            print("Initializing the model...")
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        if is_train:
            # Load loss function
            print("Loading loss function...")
            self.criterion = nn.NLLLoss(ignore_index=pad_id)

            # Load dataloaders
            print("Loading dataloaders...")
            self.train_loader = get_data_loader(TRAIN_NAME)
            self.valid_loader = get_data_loader(VALID_NAME)

        print("Setting finished.")

    def train(self):
        print("Training starts.")

        start_range = self.start_epoch
        end_range = self.start_epoch + num_epochs

        for epoch in range(start_range, end_range):
            self.model.train()

            train_losses = []
            start_time = datetime.datetime.now()

            for i, batch in tqdm(enumerate(self.train_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = (
                    src_input.to(device),
                    trg_input.to(device),
                    trg_output.to(device),
                )

                e_mask, d_mask = self.make_mask(src_input, trg_input)

                output = self.model(
                    src_input, trg_input, e_mask, d_mask
                )  # (B, L, vocab_size)

                trg_output_shape = trg_output.shape
                self.optim.zero_grad()
                loss = self.criterion(
                    output.view(-1, self.model_core.trg_vocab_size),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1]),
                )

                loss.backward()
                self.optim.step()

                train_losses.append(loss.item())

                del src_input, trg_input, trg_output, e_mask, d_mask, output

            end_time = datetime.datetime.now()
            training_time = end_time - start_time
            seconds = training_time.seconds
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60

            mean_train_loss = np.mean(train_losses)
            print(f"#################### Epoch: {epoch} ####################")
            print(
                f"Train loss: {mean_train_loss} || One epoch training time: {hours}hrs {minutes}mins {seconds}secs"
            )

            valid_loss, valid_time = self.validation()

            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)

            is_best = False
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                is_best = True
                print(
                    f"***** Epoch {epoch} has best valid loss: {self.best_loss} *****"
                )
            state_dict = {
                "model_state_dict": self.model_core.state_dict(),
                "optim_state_dict": self.optim.state_dict(),
                "loss": valid_loss,
                "best_loss": self.best_loss,
                "epoch": epoch,
            }
            torch.save(state_dict, f"{ckpt_dir}/ckpt_epoch{epoch}.tar")
            print(f"Saved checkpoint: ckpt_epoch{epoch}.tar")

            if is_best:
                torch.save(state_dict, f"{ckpt_dir}/best_ckpt.tar")
                print("***** Updated best_ckpt.tar *****")

            print(f"Best valid loss: {self.best_loss}")
            print(f"Valid loss: {valid_loss} || One epoch training time: {valid_time}")

        print("Training finished!")
    
    def validation(self):
        print("Validation processing...")
        self.model.eval()

        valid_losses = []
        start_time = datetime.datetime.now()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = (
                    src_input.to(device),
                    trg_input.to(device),
                    trg_output.to(device),
                )

                e_mask, d_mask = self.make_mask(src_input, trg_input)

                output = self.model(
                    src_input, trg_input, e_mask, d_mask
                )  # (B, L, vocab_size)

                trg_output_shape = trg_output.shape
                loss = self.criterion(
                    output.view(-1, self.model_core.trg_vocab_size),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1]),
                )

                valid_losses.append(loss.item())

                del src_input, trg_input, trg_output, e_mask, d_mask, output

        end_time = datetime.datetime.now()
        validation_time = end_time - start_time
        seconds = validation_time.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        mean_valid_loss = np.mean(valid_losses)

        return mean_valid_loss, f"{hours}hrs {minutes}mins {seconds}secs"
    
    def inference(self, input_sentence, method):
        print("Inference starts.")
        self.model.eval()

        print("Loading sentencepiece tokenizer...")
        src_sp = spm.SentencePieceProcessor()
        trg_sp = spm.SentencePieceProcessor()
        src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
        trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")

        print("Preprocessing input sentence...")
        tokenized = src_sp.EncodeAsIds(input_sentence)
        src_data = (
            torch.LongTensor(pad_or_truncate(tokenized)).unsqueeze(0).to(device)
        )  # (1, L)
        e_mask = (src_data != pad_id).unsqueeze(1).to(device)  # (1, 1, L)

        start_time = datetime.datetime.now()

        print("Encoding input sentence...")

        with torch.no_grad():
            src_data = self.model_core.src_embedding(src_data)
            src_data = self.model_core.positional_encoder(src_data)
            e_output = self.model_core.encoder(src_data, e_mask)  # (1, L, d_model)

            if method == "greedy":
                print("Greedy decoding selected.")
                result = self.greedy_search(e_output, e_mask, trg_sp)
            elif method == "beam":
                print("Beam search selected.")
                result = self.beam_search(e_output, e_mask, trg_sp)

        end_time = datetime.datetime.now()

        total_inference_time = end_time - start_time
        seconds = total_inference_time.seconds
        minutes = seconds // 60
        seconds = seconds % 60

        print(f"Input: {input_sentence}")
        print(f"Result: {result}")
        print(
            f"Inference finished! || Total inference time: {minutes}mins {seconds}secs"
        )

        return result
    
    def greedy_search(self):
        pass
    
    def beam_search(self):
        pass
    
    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1).to(device)
        d_mask = (trg_input != pad_id).unsqueeze(1).to(device)

        nopeak_mask = torch.tril(
            torch.ones((1, seq_len, seq_len), dtype=torch.bool, device=device)
        )  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask