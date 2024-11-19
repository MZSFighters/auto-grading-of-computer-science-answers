import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
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

class DSADataset(Dataset):
    def __init__(self, text, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self.create_examples(text)

    # def create_examples(self, text):
    #     # Split the text into chunks
    #     chunks = text.split('\n\n')  # Split on double newlines to separate larger sections
        
    #     tokenized_chunks = []
    #     for chunk in chunks:
    #         tokens = self.tokenizer.encode(chunk, add_special_tokens=True, max_length=self.max_length, truncation=True)
    #         if len(tokens) < 10:
    #             continue
            
    #         tokenized_chunks.append(torch.tensor(tokens))
        
    #     return tokenized_chunks

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
        tokenized_chunks = [self.tokenizer(chunk, 
                                        return_tensors="pt", 
                                        max_length=self.max_length, 
                                        truncation=True, 
                                        padding="max_length")
                            for chunk in processed_chunks if chunk.strip()]
        return tokenized_chunks

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]['input_ids'].squeeze(0)

# extra collator for batch due to GPT-2 tokenisation (padding)
def collate_batch(batch):
    # finding the maximum length in the batch and then padding sequences to that length
    max_len = max([item.size(0) for item in batch])
    padded_batch = []
    for item in batch:
        padded_item = torch.nn.functional.pad(item, (0, max_len - item.size(0)), value=tokenizer.pad_token_id)
        padded_batch.append(padded_item)
    
    # stacking all sequences
    stacked_batch = torch.stack(padded_batch)
    
    return {
        'input_ids': stacked_batch,
        'labels': stacked_batch.clone()
    }

# main training function
def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, accumulation_steps, scaler, save_dir, tokenizer):
    model.to(device)
    best_val_loss = float('inf')
    
    # training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(input_ids, labels=labels)
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
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(input_ids, labels=labels)
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
                print(f"New best model saved for epoch {epoch + 1}")
            else:
                print(f"Failed to save new best model for epoch {epoch + 1}")

def main():
    wandb.init(project="New_Code", name="gpt2-ntp-50e-b8-2e5")
    
    # Hyperparameters
    max_length = 1024  # GPT-2 can handle longer sequences
    batch_size = 8  # Smaller batch size due to larger model and sequences
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
    global tokenizer  # global tokenizer so it iss accessible in collate_batch
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Add padding token to GPT-2 tokenizer for end of sentence token
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

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

    dataset = DSADataset(text, tokenizer, max_length)
    print(f"Dataset created. Number of examples: {len(dataset)}")

    if len(dataset) < 2:
        print("Error: The dataset doesn't have enough examples for training and validation.")
        return

    # train-val splitting
    train_size = max(1, int(0.8 * len(dataset)))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch)

    # optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler()

    # saving the model
    save_dir = 'fine_tuned_gpt2'
    os.makedirs(save_dir, exist_ok=True)

    train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, accumulation_steps, scaler, save_dir, tokenizer)
    print("Model training completed.")
    wandb.finish()

if __name__ == "__main__":
    main()