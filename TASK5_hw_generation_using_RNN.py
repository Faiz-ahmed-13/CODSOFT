import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm  # For progress bars

# ----------------- 1. Load Dataset with optimizations -------------------
def load_data(max_samples=None):
    data = np.load("deepwriting_training.npz", allow_pickle=True)
    strokes = data['strokes']
    char_labels = data['char_labels']
    alphabet = data['alphabet']
    
    # Optionally limit dataset size for faster testing
    if max_samples is not None:
        strokes = strokes[:max_samples]
        char_labels = char_labels[:max_samples]
    
    # Normalize strokes
    for stroke in strokes:
        stroke[:, :2] /= np.std(stroke[:, :2])
    
    # Convert to list of tensors for better memory efficiency
    strokes = [torch.tensor(s, dtype=torch.float32) for s in strokes]
    char_labels = [torch.tensor(c, dtype=torch.long) for c in char_labels]
    
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    return strokes, char_labels, char_to_idx, alphabet

# ----------------- 2. Optimized Dataset -------------------
class HandwritingDataset(Dataset):
    def __init__(self, strokes, char_labels):
        self.strokes = strokes  # Already converted to tensors in load_data
        self.char_labels = char_labels

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        return self.strokes[idx], self.char_labels[idx]

def collate_fn(batch):
    batch_strokes, batch_chars = zip(*batch)
    
    # Pad sequences
    padded_strokes = pad_sequence(batch_strokes, batch_first=True, padding_value=0)
    padded_chars = pad_sequence(batch_chars, batch_first=True, padding_value=0)
    
    # Get sequence lengths for packing
    stroke_lengths = torch.tensor([s.size(0) for s in batch_strokes])
    
    return padded_strokes, padded_chars, stroke_lengths

# ----------------- 3. Improved Model -------------------
class HandwritingRNN(nn.Module):
    def __init__(self, char_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.char_embedding = nn.Embedding(char_dim, hidden_dim//2)
        self.dropout = nn.Dropout(dropout)
        
        # Bidirectional LSTM for better context understanding
        self.lstm = nn.LSTM(
            hidden_dim//2 + output_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, char_input, stroke_input, stroke_lengths=None):
        char_embed = self.char_embedding(char_input)
        char_embed = self.dropout(char_embed)
        
        # Ensure stroke_input has the same batch and sequence dimensions as char_embed
        if stroke_input.size(1) != char_embed.size(1):
            stroke_input = stroke_input.repeat(1, char_embed.size(1), 1)
        
        combined = torch.cat([char_embed, stroke_input], dim=-1)
        
        if stroke_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                combined, stroke_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
            output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.lstm(combined)
        
        output = self.dropout(output)
        output = self.activation(self.fc1(output))
        output = self.fc2(output)
        
        return output

# ----------------- 4. Training with optimizations -------------------
def train_model(epochs=5, batch_size=16, learning_rate=0.001, max_samples=1000):
    # Load data with size limit for faster training
    strokes, char_labels, char_to_idx, alphabet = load_data(max_samples)
    train_dataset = HandwritingDataset(strokes, char_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,  # Reduced number of workers
        pin_memory=True
    )
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = HandwritingRNN(
        char_dim=len(alphabet),
        hidden_dim=512,
        output_dim=3,
        num_layers=3,
        dropout=0.3
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=5
    )
    
    loss_fn = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_strokes, batch_chars, stroke_lengths in pbar:
            batch_strokes = batch_strokes.to(device)
            batch_chars = batch_chars.to(device)
            
            optimizer.zero_grad()
            
            inputs = batch_strokes[:, :-1, :]
            targets = batch_strokes[:, 1:, :]
            char_inputs = batch_chars[:, :-1]
            
            predictions = model(char_inputs, inputs, stroke_lengths-1)
            loss = loss_fn(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss/(pbar.n+1))
        
        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_handwriting_model.pth')
    
    return model, char_to_idx

# ----------------- 5. Improved Generation -------------------
def generate_handwriting(model, char_to_idx, text, seq_len=100, temperature=0.5):
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Initialize sequence with proper dimensions
        current_stroke = torch.zeros(1, 1, 3, device=device)  
        
        # Convert text to character indices
        char_indices = [char_to_idx.get(c, 0) for c in text]
        char_seq = torch.tensor([char_indices], device=device) 
        
        # Generate strokes
        generated_strokes = []
        for i in range(seq_len):
            # Ensure current_stroke has the right dimensions
            if current_stroke.size(1) != char_seq.size(1):
                current_stroke = current_stroke.repeat(1, char_seq.size(1), 1)
            
            output = model(char_seq, current_stroke)
            
            # Add randomness
            output = output / temperature
            noise = torch.randn_like(output) * 0.1
            next_stroke = output + noise
            
            # Take the last stroke from the sequence
            next_stroke = next_stroke[:, -1, :]
            generated_strokes.append(next_stroke.cpu().numpy())
            current_stroke = next_stroke.unsqueeze(1)  # Add sequence dimension back
        
        # Stack all strokes into a single array
        return np.array(generated_strokes).squeeze()  # Remove any extra dimensions

# ----------------- Main Execution -------------------
if __name__ == "__main__":
    # Train model with reduced epochs for testing
    print("Training model...")
    model, char_to_idx = train_model(
        epochs=2,  
        batch_size=8,  
        learning_rate=0.001,
        max_samples=500  
    )
    
    # Generate samples
    print("\nGenerating samples...")
    text_samples = ["a", "b"]  # Shorter samples for testing
    
    plt.figure(figsize=(10, 4))
    for idx, text in enumerate(text_samples, 1):
        plt.subplot(1, 2, idx)
        generated_strokes = generate_handwriting(model, char_to_idx, text, seq_len=50)  # Reduced sequence length
        
        # Plot the generated handwriting
        x, y = 0, 0
        X, Y = [], []
        for stroke in generated_strokes:
            if len(stroke) == 3:  
                dx, dy, pen = stroke
                x += dx
                y += dy
                X.append(x)
                Y.append(y)
                if pen > 0.5:  
                    if len(X) > 1:  
                        plt.plot(X, Y, 'k-')
                    X, Y = [], []
        
        # Plot any remaining points
        if len(X) > 1:
            plt.plot(X, Y, 'k-')
                
        plt.title(f"Generated: '{text}'")
        plt.gca().invert_yaxis()
        plt.axis('equal')
    
    plt.tight_layout()
    plt.show()