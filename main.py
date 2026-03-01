# train_wikitext2_full.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
import json
import torch.nn.functional as F
import random

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_position=128, num_segments=2, dropout=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Embedding(max_position, embed_dim)
        self.segment = nn.Embedding(num_segments, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        x = self.token(input_ids) + self.position(pos_ids) + self.segment(segment_ids)
        return self.dropout(self.norm(x))

class SimpleBert(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.emb = BertEmbedding(vocab_size, embed_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlm_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim), nn.Linear(embed_dim, vocab_size))
        self.nsp_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))
    
    def forward(self, input_ids, segment_ids, attention_mask=None):
        emb = self.emb(input_ids, segment_ids)
        # Create padding mask for transformer
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # True for padding to ignore
        else:
            src_key_padding_mask = None
        encoded = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        cls_token = encoded[:, 0, :]  # [CLS] for NSP
        mlm_logits = self.mlm_head(encoded)
        nsp_logits = self.nsp_head(cls_token)
        return mlm_logits, nsp_logits

class FullBertDataset(Dataset):
    def __init__(self, filename):
        with open(filename) as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        return (torch.tensor(ex['input_ids'], dtype=torch.long),
                torch.tensor(ex['token_type_ids'], dtype=torch.long),
                torch.tensor(ex['attention_mask'], dtype=torch.long),
                torch.tensor(ex['mlm_labels'], dtype=torch.long),
                torch.tensor(ex['nsp_label'], dtype=torch.long))

def demonstrate_predictions(model, tokenizer, device, dataset, num_samples=3):
    print("\n--- Demonstration: Masked Token Predictions (MLM) & NSP ---")
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            # Pick random example
            ex_idx = random.randint(0, len(dataset) - 1)
            ex = dataset[ex_idx]  # Use dataset __getitem__ for consistency
            input_ids, segment_ids, attention_mask, mlm_labels, nsp_label = ex
            input_ids = input_ids.unsqueeze(0).to(device)
            segment_ids = segment_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)
            
            mlm_logits, nsp_logits = model(input_ids, segment_ids, attention_mask)
            mlm_preds = torch.argmax(mlm_logits[0], dim=-1).cpu()
            nsp_pred = torch.argmax(nsp_logits, dim=-1).item()
            
            # Decode for readability
            original_input = tokenizer.decode(input_ids[0].cpu(), skip_special_tokens=False)
            masked_pos = torch.where(mlm_labels != -100)[0]
            actual_tokens = [tokenizer.decode([mlm_labels[j].item()]) for j in masked_pos if mlm_labels[j] != -100]
            predicted_tokens = [tokenizer.decode([mlm_preds[j].item()]) for j in masked_pos if mlm_labels[j] != -100]
            
            print(f"\nSample {i+1} (NSP Label: {nsp_label.item()}, Predicted: {nsp_pred}):")
            print(f"Input (with masks): {original_input[:100]}...")  # Truncate for brevity
            print(f"Masked positions originals: {actual_tokens[:5]}...")  # First few
            print(f"Model predictions for masks: {predicted_tokens[:5]}...")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    global dataset  # For demo access
    dataset = FullBertDataset('preprocessed_wikitext2_full.json')
    loader = DataLoader(dataset, batch_size=16, shuffle=True)  # Smaller batch for stability
    model = SimpleBert(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # BERT-style optimizer/LR
    
    # Add LR scheduler (warmup for 10% steps, then decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, epochs=10, steps_per_epoch=len(loader))
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_criterion = nn.CrossEntropyLoss()
    epochs = 10  # For good convergence on full data
    
    for epoch in range(epochs):
        model.train()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        mlm_loss_sum, nsp_loss_sum = 0, 0
        mlm_correct, mlm_total, nsp_correct, total = 0, 0, 0, 0
        
        with tqdm(loader, desc=f"Epoch {epoch+1}", leave=True) as pbar:
            for input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels in pbar:
                input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels = (
                    input_ids.to(device), segment_ids.to(device), attention_mask.to(device),
                    mlm_labels.to(device), nsp_labels.to(device)
                )
                mlm_logits, nsp_logits = model(input_ids, segment_ids, attention_mask)
                
                # MLM loss: Flatten, ignore -100
                active_logits = mlm_logits.view(-1, vocab_size)
                active_labels = mlm_labels.view(-1)
                mlm_loss = mlm_criterion(active_logits, active_labels)
                
                # NSP loss
                nsp_loss = nsp_criterion(nsp_logits, nsp_labels)
                loss = mlm_loss + nsp_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping for stability
                optimizer.step()
                scheduler.step()  # LR scheduling
                
                # Metrics
                mlm_preds = torch.argmax(mlm_logits, dim=-1)
                mask = mlm_labels != -100
                mlm_correct += (mlm_preds[mask] == mlm_labels[mask]).sum().item()
                mlm_total += mask.sum().item()
                nsp_preds = torch.argmax(nsp_logits, dim=-1)
                nsp_correct += (nsp_preds == nsp_labels).sum().item()
                total += input_ids.size(0)
                mlm_loss_sum += mlm_loss.item()
                nsp_loss_sum += nsp_loss.item()
                
                pbar.set_postfix({
                    'Total Loss': f"{loss.item():.4f}",
                    'MLM Acc': f"{mlm_correct / mlm_total:.3f}" if mlm_total > 0 else "0.000",
                    'NSP Acc': f"{nsp_correct / total:.3f}"
                })
        
        avg_mlm_loss = mlm_loss_sum / len(loader)
        avg_nsp_loss = nsp_loss_sum / len(loader)
        print(f"Epoch {epoch+1} Summary: MLM Loss={avg_mlm_loss:.4f}, NSP Loss={avg_nsp_loss:.4f}")
        print(f"MLM Accuracy (masked only)={mlm_correct / mlm_total:.3f}, NSP Accuracy={nsp_correct / total:.3f}")
    
    demonstrate_predictions(model, tokenizer, device, dataset)
    torch.save(model.state_dict(), 'bert_wikitext2_full.pth')
    print("Training complete. Model saved as 'bert_wikitext2_full.pth'.")

if __name__ == "__main__":
    train()
