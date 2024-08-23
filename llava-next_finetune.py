# %%
MAX_LENGTH = 1000
USE_QLORA = False
MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"  # Download from HuggingFace
REPO_ID = ""
WANDB_PROJECT = ""
WANDB_NAME = ""
OUTPUT_DIR = ""
DATA_ROOT = ""
DATASET_JSONL = ""  # Eg. "general_perception.jsonl"

import os

# %%
import wandb
# os.environ["WANDB__SERVICE_WAIT"] = "300"
wandb.login()

# %% [markdown]
# ### Load processor

# %%
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

# %% [markdown]
# ### Load model

# %%
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration
import torch



## Load model
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="cuda:1"
    )
else:
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    )


# %% [markdown]
# ### Apply PEFT

# %%
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_model"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names: # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# %% [markdown]
# ### Create PyTorch dataset

# %%
import torch
from torch.utils.data import Dataset
import json_lines
import json
from PIL import Image
from itertools import islice
import os

class CodaDataset(Dataset):
    def __init__(self, data_root, split="Train", dataset_jsonl="general_perception.jsonl"):
        self.data_root = data_root
        self.split = split
        self.dataset_jsonl = dataset_jsonl

        self.get_json_file()

        self.dataset_len = len(self.dataset)


    def get_json_file(self):
        self.dataset = [json.loads(q) for q in open(os.path.expanduser(os.path.join(self.data_root, "CODA-LM", self.split, "vqa_anno", self.dataset_jsonl)), "r", encoding="utf-8")]

    def __len__(self):
        return self.dataset_len
        # return 10

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = Image.open(os.path.join(self.data_root, sample["image"]))
        return img, sample


# %%
train_dataset = CodaDataset(data_root=DATA_ROOT, split="Train", dataset_jsonl=DATASET_JSONL)
val_dataset = CodaDataset(data_root=DATA_ROOT, split="Val", dataset_jsonl=DATASET_JSONL)

# %% [markdown]
# ### Define collate functions

# %%
def train_collate_fn(examples):
    images = []
    texts = []
    for example in examples:
        image, sample = example
        images.append(image)

        if sample["question"] == "There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.":
            prompt = f"A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\n Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's color, position, status, implication, respones, and how they influence ego car. EXPERT: {sample['answer']}"
        elif sample["question"] == "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.":
            prompt = f"A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\nPlease describe the object inside the red rectangle in the image. Describe its color, position, status, implication, response, and explain why it affect ego car driving. EXPERT: {sample['answer']}"
        elif sample["question"] == "There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.":
            prompt = f"A chat between a curious human and an autonomous driving expert, specializing in providing specific and helpful driving suggestions. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. You must not discuss any objects that not show in the image. Please provide driving suggestions for the ego car based on the current scene. EXPERT: {sample['answer']}"
        else:
            prompt = ""
        assert prompt != ""

        texts.append(prompt)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, image_sizes, labels


def eval_collate_fn(examples):
    # we only feed the prompt to the model
    images = []
    texts = []
    answers = []
    for example in examples:
        image, sample = example
        images.append(image)
        
        if sample["question"] == "There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.":
            prompt = "A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\n Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, braking lights, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's color, position, status, implication, respones, and how they influence ego car. EXPERT:"
        elif sample["question"] == "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.":
            prompt = "A chat between a curious human and an autonomous driving expert, specializing in recognizing traffic scenes and making detailed explanation. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\nPlease describe the object inside the red rectangle in the image. Describe its color, position, status, implication, response, and explain why it affect ego car driving. EXPERT:"
        elif sample["question"] == "There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.":
            prompt = "A chat between a curious human and an autonomous driving expert, specializing in providing specific and helpful driving suggestions. The expert receives an image of traffic captured from the perspective of the ego car. USER: <image>\There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. You must not discuss any objects that not show in the image. Please provide driving suggestions for the ego car based on the current scene. EXPERT:"
        else:
            prompt = ""
        assert prompt != ""

        texts.append(prompt)
        answers.append(sample["answer"])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]

    return input_ids, attention_mask, pixel_values, image_sizes, answers

# %% [markdown]
# ### Define PyTorch LightningModule

# %%
import lightning as L
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np


class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, image_sizes, labels = batch

        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            labels=labels
                          )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, image_sizes, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, image_sizes=image_sizes, max_new_tokens=MAX_LENGTH)
        # turn them back into text, chopping of the prompt
        # important: we don"t skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

# %%
config = {"max_epochs": 20,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 1,
          # "seed":2022,
          "verbose": True,
}

model_module = LlavaModelPLModule(config, processor, model)


# %% [markdown]
# ### Train

# %%
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from huggingface_hub import HfApi

api = HfApi()

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        output_dir = f"{OUTPUT_DIR}/llava-next-fintune-epoch{trainer.current_epoch}"
        pl_module.model.save_pretrained(output_dir)
        pl_module.processor.save_pretrained(output_dir)
        # pl_module.model.push_to_hub(REPO_ID,
        #                             commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    # def on_train_end(self, trainer, pl_module):
    #     print(f"Pushing model to the hub after training")
    #     pl_module.processor.push_to_hub(REPO_ID,
    #                                 commit_message=f"Training done")
    #     pl_module.model.push_to_hub(REPO_ID,
    #                                 commit_message=f"Training done")

early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=10, verbose=False, mode="min")

# %%
from lightning.pytorch.loggers import WandbLogger


os.environ["WANDB__SERVICE_WAIT"] = "300"
wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
        accelerator="gpu",
        devices=[1],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[PushToHubCallback()]
)

#trainer.fit(model_module)
trainer.fit(model_module)




