import torch
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    model.to(device)
    epoch_avg_loss = 0.0  # Initialize epoch_avg_loss here
    for epoch in range(epochs):
        epoch_loss = 0.0  # Initialize epoch_loss here
        progress_bar = tqdm(enumerate(dataloader), 
                          total=len(dataloader),
                          desc=f'Epoch {epoch+1}/{epochs}',
                          leave=True)
        
        for batch_idx, (frames, text) in progress_bar:
            frames = frames.to(device)
            text = text.to(device)
            
            # Forward pass
            logits = model(frames, text)
            
            # Compute loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                text[:, 1:].reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update tracking
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'epoch_avg': f'{avg_loss:.4f}'
            })
        
        # Calculate final average for the epoch
        epoch_avg_loss = epoch_loss / len(dataloader)
        tqdm.write(f'Epoch {epoch+1} complete - Avg Loss: {epoch_avg_loss:.4f}')
    
    return epoch_avg_loss