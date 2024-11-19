import torch
from transformers import BertTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import os
import shutil

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# saves the model and the tokenizer for reloading
def save_model(model, tokenizer, save_dir, epoch):
    try:
        epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch + 1}")
        if os.path.exists(epoch_save_dir):
            shutil.rmtree(epoch_save_dir)
        os.makedirs(epoch_save_dir)
        
        model.save_pretrained(epoch_save_dir)
        tokenizer.save_pretrained(epoch_save_dir)
        
        print(f"Model and tokenizer saved successfully for epoch {epoch + 1}")
        return True
    except Exception as e:
        print(f"Error saving model for epoch {epoch + 1}: {str(e)}")
        return False

class MaskedLanguageModelingDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self.create_examples(text)

    def create_examples(self, text):
        # slitting the text into chunks using newline characters
        chunks = text.split('\n')
    
        #joining short chunks and split long ones
        processed_chunks = []
        current_chunk = ""
        for chunk in chunks:
            if len(current_chunk) + len(chunk) > self.max_length:
                processed_chunks.append(current_chunk.strip())
                current_chunk = chunk
            else:
                current_chunk += " " + chunk
        if current_chunk:
            processed_chunks.append(current_chunk.strip())
        
        # Tokenizing
        tokenized_chunks = [self.tokenizer(chunk, return_tensors="pt", max_length=self.max_length, 
                                           truncation=True, padding="max_length")
                            for chunk in processed_chunks if chunk.strip()]
        return tokenized_chunks

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = {key: val[0].clone().detach() for key, val in self.examples[idx].items()}
        labels = item['input_ids'].clone()
        
        # BERT random mask probability setup
        mask = torch.rand(item['input_ids'].shape) < 0.15
        mask[item['input_ids'] == self.tokenizer.pad_token_id] = False
        mask[item['input_ids'] == self.tokenizer.cls_token_id] = False
        mask[item['input_ids'] == self.tokenizer.sep_token_id] = False

        item['input_ids'][mask] = self.tokenizer.mask_token_id
        
        return item, labels

# main training function
def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, accumulation_steps, scaler, save_dir, tokenizer):
    model.to(device)
    best_val_loss = float('inf')
    
    # training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, (batch, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, labels in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = labels.to(device)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(**batch, labels=labels)
                    loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_model(model, tokenizer, save_dir, epoch):
                print(f"best model saved for epoch {epoch + 1}")
            else:
                print(f"Failed best model for epoch {epoch + 1}")

def main():
    wandb.init(project="New_Code", name="bert-mlm-50e-16b-2e5")
    
    # Hyperparameters
    max_length = 512
    batch_size = 16
    num_epochs = 50
    learning_rate = 2e-5
    accumulation_steps = 8
    warmup_steps = 1000
    weight_decay = 0.01

    wandb.config.update({
        "max_length": max_length,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "accumulation_steps": accumulation_steps,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model loading and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # loading the DSA corpus
    file_path = './dsa_corpus.txt'
    try:
        text = read_text_file(file_path)
        print(f"Successfully read file: {file_path}")
        print(f"Text length: {len(text)} characters")
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not text:
        print("Error: The text file is empty.")
        return

    dataset = MaskedLanguageModelingDataset(text, tokenizer, max_length)
    print(f"Dataset created. Number of examples: {len(dataset)}")

    if len(dataset) < 2:
        print("Error: The dataset doesn't have enough examples for train and val")
        return

    # train-val splitting
    train_size = max(1, int(0.8 * len(dataset)))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler()

    # saving the model
    save_dir = 'fine_tuned_bert_mlm'
    os.makedirs(save_dir, exist_ok=True)

    train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, accumulation_steps, scaler, save_dir, tokenizer)
    print("Model training completed.")
    
    wandb.finish()

if __name__ == "__main__":
    main()