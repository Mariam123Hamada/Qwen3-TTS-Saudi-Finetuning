
# coding=utf-8
# Qwen3-TTS 12Hz SFT fine-tuning
# Memory-optimized version for Tesla T4 (16GB)
# Model: Qwen3-TTS-12Hz-0.6B-Base

import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from huggingface_hub import snapshot_download
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from bitsandbytes.optim import AdamW8bit
from torch.utils.data import DataLoader
from transformers import AutoConfig


target_speaker_embedding = None


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    )

    parser.add_argument(
        "--output_model_path",
        type=str,
        default="output",
    )

    parser.add_argument(
        "--train_jsonl",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--speaker_name",
        type=str,
        default="my_speaker",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Accelerator configuration
    # ---------------------------------------------------------

    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="fp16",
        log_with="tensorboard",
        project_dir="/kaggle/working/logs",
    )

    if accelerator.is_main_process:
        print("Starting Qwen3-TTS fine-tuning")
        print(f"Model: {args.init_model_path}")
        print(f"Dataset: {args.train_jsonl}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Epochs: {args.num_epochs}")
        print(f"Speaker: {args.speaker_name}")

    # ---------------------------------------------------------
    # 2. Resolve model path
    # ---------------------------------------------------------

    MODEL_PATH = snapshot_download(
        repo_id=args.init_model_path,
    )

    if accelerator.is_main_process:
        print(f"Local model path: {MODEL_PATH}")

    # ---------------------------------------------------------
    # 3. Load model in FP32 and use SDPA
    #
    # Keep FP32 model parameters to avoid the previous
    # "Attempting to unscale FP16 gradients" error.
    #
    # FP16 mixed precision is handled by Accelerate.
    # ---------------------------------------------------------

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )

    config = AutoConfig.from_pretrained(MODEL_PATH)

    # ---------------------------------------------------------
    # 4. Load training dataset
    # ---------------------------------------------------------

    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        train_data = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if not train_data:
        raise ValueError("The training JSONL file is empty.")

    if accelerator.is_main_process:
        print(f"Training samples: {len(train_data)}")

    dataset = TTSDataset(
        train_data,
        qwen3tts.processor,
        config,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # 5. 8-bit Optimizer
    #
    # Uses less GPU memory than regular AdamW.
    # ---------------------------------------------------------

    optimizer = AdamW8bit(
        qwen3tts.model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # ---------------------------------------------------------
    # 6. Prepare model, optimizer and dataloader
    # ---------------------------------------------------------

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model,
        optimizer,
        train_dataloader,
    )

    # ---------------------------------------------------------
    # 7. Enable gradient checkpointing when supported
    # ---------------------------------------------------------

    unwrapped_model = accelerator.unwrap_model(model)

    if hasattr(unwrapped_model.talker, "gradient_checkpointing_enable"):

        unwrapped_model.talker.gradient_checkpointing_enable()

        accelerator.print(
            "Gradient checkpointing enabled for the talker."
        )

    else:

        accelerator.print(
            "Warning: Talker does not expose gradient checkpointing."
        )

    model.train()

    # ---------------------------------------------------------
    # 8. TensorBoard tracking
    # ---------------------------------------------------------

    accelerator.init_trackers(
        "qwen3-tts-saudi",
        config={
            "model": args.init_model_path,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs": args.num_epochs,
            "speaker": args.speaker_name,
            "mixed_precision": "fp16",
            "attention": "sdpa",
            "optimizer": "AdamW8bit",
            "gradient_accumulation_steps": 4,
        },
    )

    global_step = 0
    num_epochs = args.num_epochs

    # ---------------------------------------------------------
    # 9. Training loop
    # ---------------------------------------------------------

    for epoch in range(num_epochs):

        model.train()

        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(model):

                # ---------------------------------------------
                # Move batch tensors to model device
                # ---------------------------------------------

                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]

                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]

                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                # ---------------------------------------------
                # Speaker embedding
                # ---------------------------------------------

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(
                        device=model.device,
                        dtype=model.dtype,
                    )
                ).detach()

                if target_speaker_embedding is None:
                    target_speaker_embedding = (
                        speaker_embedding.detach()
                    )

                # ---------------------------------------------
                # Separate text and codec IDs
                # ---------------------------------------------

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # ---------------------------------------------
                # Text embedding projection
                # ---------------------------------------------

                input_text_embedding = model.talker.text_projection(
                    model.talker.model.text_embedding(input_text_ids)
                )

                input_text_embedding = (
                    input_text_embedding * text_embedding_mask
                )

                # ---------------------------------------------
                # Codec embedding
                # ---------------------------------------------

                input_codec_embedding = (
                    model.talker.model.codec_embedding(input_codec_ids)
                    * codec_embedding_mask
                )

                # Insert speaker embedding
                input_codec_embedding[:, 6, :] = speaker_embedding

                # ---------------------------------------------
                # Combine text and codec embeddings
                # ---------------------------------------------

                input_embeddings = (
                    input_text_embedding + input_codec_embedding
                )

                # ---------------------------------------------
                # Add sub-codec embeddings
                # ---------------------------------------------

                for i in range(1, 16):

                    codec_i_embedding = (
                        model.talker.code_predictor
                        .get_input_embeddings()[i - 1](
                            codec_ids[:, :, i]
                        )
                    )

                    codec_i_embedding = (
                        codec_i_embedding * codec_mask.unsqueeze(-1)
                    )

                    input_embeddings = (
                        input_embeddings + codec_i_embedding
                    )

                # ---------------------------------------------
                # Main talker forward pass
                # ---------------------------------------------

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                # ---------------------------------------------
                # Sub-talker loss
                # ---------------------------------------------

                hidden_states = outputs.hidden_states[0][-1]

                talker_hidden_states = hidden_states[
                    codec_mask[:, :-1]
                ]

                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = (
                    model.talker.forward_sub_talker_finetune(
                        talker_codec_ids,
                        talker_hidden_states,
                    )
                )

                # ---------------------------------------------
                # Combined loss
                # ---------------------------------------------

                loss = outputs.loss + 0.3 * sub_talker_loss

                # ---------------------------------------------
                # Backpropagation
                # ---------------------------------------------

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            # ---------------------------------------------
            # Logging
            # ---------------------------------------------

            if global_step % 10 == 0:

                current_loss = loss.detach().float().item()

                accelerator.print(
                    f"Epoch {epoch + 1}/{num_epochs} "
                    f"| Step {global_step} "
                    f"| Loss: {current_loss:.4f}"
                )

                accelerator.log(
                    {
                        "train/loss": current_loss,
                        "train/main_loss": (
                            outputs.loss.detach().float().item()
                        ),
                        "train/sub_talker_loss": (
                            sub_talker_loss.detach().float().item()
                        ),
                    },
                    step=global_step,
                )

            global_step += 1

        # -----------------------------------------------------
        # 10. Save checkpoint after each epoch
        # -----------------------------------------------------

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:

            output_dir = os.path.join(
                args.output_model_path,
                f"checkpoint-epoch-{epoch}",
            )

            os.makedirs(args.output_model_path, exist_ok=True)

            # Copy original model files
            shutil.copytree(
                MODEL_PATH,
                output_dir,
                dirs_exist_ok=True,
            )

            # ---------------------------------------------
            # Update model configuration
            # ---------------------------------------------

            input_config_file = os.path.join(
                MODEL_PATH,
                "config.json",
            )

            output_config_file = os.path.join(
                output_dir,
                "config.json",
            )

            with open(
                input_config_file,
                "r",
                encoding="utf-8",
            ) as f:
                config_dict = json.load(f)

            config_dict["tts_model_type"] = "custom_voice"

            talker_config = config_dict.get(
                "talker_config",
                {},
            )

            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }

            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }

            config_dict["talker_config"] = talker_config

            with open(
                output_config_file,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    config_dict,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            # ---------------------------------------------
            # Save trained model weights
            # ---------------------------------------------

            unwrapped_model = accelerator.unwrap_model(model)

            state_dict = {
                key: value.detach().to("cpu").contiguous()
                for key, value in unwrapped_model.state_dict().items()
            }

            # Remove speaker encoder weights
            drop_prefix = "speaker_encoder"

            keys_to_drop = [
                key
                for key in state_dict
                if key.startswith(drop_prefix)
            ]

            for key in keys_to_drop:
                del state_dict[key]

            # ---------------------------------------------
            # Save target speaker embedding at speaker ID 3000
            # ---------------------------------------------

            codec_embedding_key = (
                "talker.model.codec_embedding.weight"
            )

            if codec_embedding_key not in state_dict:
                raise KeyError(
                    f"Missing expected weight: {codec_embedding_key}"
                )

            if target_speaker_embedding is None:
                raise RuntimeError(
                    "Target speaker embedding was not initialized."
                )

            codec_embedding_weight = state_dict[
                codec_embedding_key
            ]

            codec_embedding_weight[3000] = (
                target_speaker_embedding[0]
                .detach()
                .to(
                    device="cpu",
                    dtype=codec_embedding_weight.dtype,
                )
            )

            # ---------------------------------------------
            # Write safetensors checkpoint
            # ---------------------------------------------

            save_path = os.path.join(
                output_dir,
                "model.safetensors",
            )

            save_file(
                state_dict,
                save_path,
            )

            accelerator.print(
                f"Checkpoint saved successfully: {output_dir}"
            )

    # ---------------------------------------------------------
    # 11. Finish tracking
    # ---------------------------------------------------------

    accelerator.wait_for_everyone()
    accelerator.end_training()

    if accelerator.is_main_process:
        print("Fine-tuning completed.")


if __name__ == "__main__":
    train()
