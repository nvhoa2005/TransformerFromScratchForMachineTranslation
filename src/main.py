from tqdm import tqdm
from constants import (
    SP_DIR,
    src_model_prefix,
    trg_model_prefix,
    seq_len,
    pad_id,
    sos_id,
    eos_id,
    learning_rate,
    device,
    num_epochs,
    ckpt_dir,
    beam_size,
    start_epoch,
)
from constants import TRAIN_NAME, VALID_NAME
from custom_data import get_data_loader, pad_or_truncate
from transformer import Transformer
from torch import nn
import torch.nn.functional as F

import torch
import sys
import os
import numpy as np
import argparse
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

    def inference(self, input_sentence, method, beam_k=None):
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
                k = beam_k if beam_k is not None else beam_size
                print(f"Beam search selected with size {k}.")
                result = self.beam_search(e_output, e_mask, trg_sp, beam_size=k)

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

    def greedy_search(self, e_output, e_mask, trg_sp):
        last_words = torch.LongTensor([pad_id] * seq_len).to(device)  # (L)
        last_words[0] = sos_id  # (L)
        cur_len = 1

        for i in range(seq_len):
            d_mask = (
                (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device)
            )  # (1, 1, L)
            nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(
                device
            )  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
            d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

            trg_embedded = self.model_core.trg_embedding(last_words.unsqueeze(0))
            trg_positional_encoded = self.model_core.positional_encoder(trg_embedded)
            decoder_output = self.model_core.decoder(
                trg_positional_encoded, e_output, e_mask, d_mask
            )  # (1, L, d_model)

            output = self.model_core.softmax(
                self.model_core.output_linear(decoder_output)
            )  # (1, L, trg_vocab_size)

            output = torch.argmax(output, dim=-1)  # (1, L)
            last_word_id = output[0][i].item()

            if i < seq_len - 1:
                last_words[i + 1] = last_word_id
                cur_len += 1

            if last_word_id == eos_id:
                break

        if last_words[-1].item() == pad_id:
            decoded_output = last_words[1:cur_len].tolist()
        else:
            decoded_output = last_words[1:].tolist()
        decoded_output = trg_sp.decode_ids(decoded_output)

        return decoded_output

    def beam_search(self, e_output, e_mask, trg_sp, beam_size=beam_size, alpha=0.7):
        """
        Beam search implemented correctly.
        - e_output: encoder outputs (1, L_enc, d_model)
        - e_mask: encoder mask (1, 1, L_enc)
        - trg_sp: SentencePiece processor for decoding ids->text
        - beam_size: beam width
        - alpha: length normalization hyperparameter (common default ~0.7)
        """
        self.model.eval()

        # Each hypothesis: (tokens_list, cumulative_logprob, is_finished)
        # Initialize with single hypothesis [SOS]
        hypotheses = [([sos_id], 0.0, False)]

        for t in range(seq_len):
            all_candidates = []

            # If all hypotheses are finished, we can stop early
            if all(h[2] for h in hypotheses):
                break

            # Build batch of decoder inputs: for every hypothesis that is not finished,
            # we'll expand with top-k next token candidates. For finished hypos, keep them as-is.
            # We will run decoder on the batch of candidate sequences to get log-probs.
            for h_idx, (tokens, logp, finished) in enumerate(hypotheses):
                if finished:
                    # keep finished hypothesis as a candidate (carry over)
                    all_candidates.append((tokens, logp, True))
                else:
                    # We will expand this hypothesis; but first create its current input (padded)
                    # We'll ask the model for top-k next tokens; to do that efficiently we will
                    # create candidate inputs later.
                    # For now just note we will expand this hypothesis.
                    # We'll create the actual candidate inputs after we determine top-k per hypo.
                    pass

            # To get top-k for each hypothesis we need model output at position t given its tokens.
            # We'll create a batch of current hypotheses (one per non-finished hypo), run decoder,
            # and extract log-probs at time step t, then pick top-k per hypothesis.
            alive_hypos = [(idx, h) for idx, h in enumerate(hypotheses) if not h[2]]
            if len(alive_hypos) == 0:
                break

            # Prepare batch input: for each alive hypo, create padded tensor (seq_len) with its tokens
            batch_inputs = []
            hypo_map = []  # map from batch row -> hypothesis index
            for h_idx, (tokens, logp, finished) in alive_hypos:
                seq = tokens + [pad_id] * (seq_len - len(tokens))
                batch_inputs.append(seq)
                hypo_map.append(h_idx)

            batch_inputs = torch.LongTensor(batch_inputs).to(
                device
            )  # (B_alive, seq_len)

            # Create decoder mask for this batch
            d_mask = (
                (batch_inputs != pad_id).unsqueeze(1).to(device)
            )  # (B_alive, 1, seq_len)
            nopeak = torch.tril(
                torch.ones((1, seq_len, seq_len), dtype=torch.bool, device=device)
            )
            d_mask = d_mask & nopeak  # (B_alive, seq_len, seq_len) broadcasted

            # Run decoder for this batch (one forward)
            with torch.no_grad():
                trg_emb = self.model_core.trg_embedding(
                    batch_inputs
                )  # (B_alive, L, d_model)
                trg_emb = self.model_core.positional_encoder(
                    trg_emb
                )  # (B_alive, L, d_model)
                dec_out = self.model_core.decoder(
                    trg_emb,
                    e_output.repeat(len(batch_inputs), 1, 1)
                    if e_output.size(0) == 1
                    else e_output,
                    e_mask.repeat(len(batch_inputs), 1, 1)
                    if e_mask.size(0) == 1
                    else e_mask,
                    d_mask,
                )  # (B_alive, L, d_model)
                # Get log-probs (use model's output_linear then log_softmax to be safe)
                logits = self.model_core.output_linear(dec_out)  # (B_alive, L, V)
                log_probs = F.log_softmax(logits, dim=-1)  # (B_alive, L, V)

            # For each alive hypothesis, get top-k tokens at time step t
            B_alive = log_probs.size(0)
            V = log_probs.size(-1)
            topk = min(beam_size, V)

            for i in range(B_alive):
                hypo_idx = hypo_map[i]
                tokens, curr_logp, _ = hypotheses[hypo_idx]
                # get log-prob vector at time t
                logp_t = log_probs[i, t]  # (V,)
                top_vals, top_idx = torch.topk(logp_t, k=topk)  # both tensors
                top_vals = top_vals.cpu().tolist()
                top_idx = top_idx.cpu().tolist()
                for k_idx, token_id in enumerate(top_idx):
                    new_tokens = tokens + [token_id]
                    new_logp = curr_logp + top_vals[k_idx]  # cumulative log-prob
                    finished = token_id == eos_id
                    all_candidates.append((new_tokens, new_logp, finished))

            # Also include previous finished hypotheses (they were added earlier)

            # Now select top `beam_size` candidates among all_candidates by cumulative log-prob
            # Note: do NOT normalize length here (we keep cumulative log-prob for beam propagation).
            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            hypotheses = all_candidates[:beam_size]

            # If number of hypotheses < beam_size (possible if many finished), pad by carrying best finished
            # (not strictly necessary)

        # After finishing (either reached seq_len or all finished), choose best hypothesis with length normalization
        # Apply length normalization score = logp / (len(tokens) ** alpha)
        final_scores = []
        for tokens, logp, finished in hypotheses:
            length = len(tokens) - 1  # exclude SOS for length
            if length <= 0:
                length = 1.0
            score = logp / (length**alpha)
            final_scores.append((score, tokens, finished))

        final_scores = sorted(final_scores, key=lambda x: x[0], reverse=True)
        best_tokens = final_scores[0][1]

        # Remove leading SOS and trailing EOS if present
        if best_tokens and best_tokens[0] == sos_id:
            best_tokens = best_tokens[1:]
        if best_tokens and best_tokens[-1] == eos_id:
            best_tokens = best_tokens[:-1]

        return trg_sp.decode_ids(best_tokens)

    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1).to(device)
        d_mask = (trg_input != pad_id).unsqueeze(1).to(device)

        nopeak_mask = torch.tril(
            torch.ones((1, seq_len, seq_len), dtype=torch.bool, device=device)
        )  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="train or inference?")
    parser.add_argument("--ckpt_name", required=False, help="best checkpoint file")
    parser.add_argument(
        "--input", type=str, required=False, help="input sentence when inferencing"
    )
    parser.add_argument(
        "--decode", type=str, required=False, default="greedy", help="greedy or beam?"
    )

    args = parser.parse_args()

    if args.mode == "train":
        if args.ckpt_name is not None:
            manager = Manager(is_train=True, ckpt_name=args.ckpt_name)
        else:
            manager = Manager(is_train=True)

        manager.train()
    elif args.mode == "inference":
        assert args.ckpt_name is not None, (
            "Please specify the model file name you want to use."
        )
        assert args.input is not None, "Please specify the input sentence to translate."
        assert args.decode == "greedy" or args.decode == "beam", (
            "Please specify correct decoding method, either 'greedy' or 'beam'."
        )

        manager = Manager(is_train=False, ckpt_name=args.ckpt_name)
        manager.inference(args.input, args.decode)

    else:
        print("Please specify mode argument either with 'train' or 'inference'.")