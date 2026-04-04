import torch
import torch.nn as nn
import torch.optim as optim

class RobotPolicy(nn.Module):
    """
    Neural Policy Model for Robot Action Prediction.
    Learns from motion features (encoded human poses) to predict robot joint outputs.
    """

    def __init__(self, input_dim, output_dim):
        """
        Initialize the policy network.
        
        Args:
            input_dim (int): Dimension of encoded motion vector (Positions + Velocities).
            output_dim (int): Dimension of robot control outputs (e.g., target joint angles).
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout for regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        """Perform forward pass."""
        return self.net(x)

def train_policy(model, dataloader, epochs=10, lr=0.001):
    """Basic training loop for imitation learning."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            
            pred = model(x)
            loss = criterion(pred, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.6f}")

if __name__ == "__main__":
    # Test Policy Model
    input_size = 450 # (225 pos + 225 vel)
    output_size = 18 # (e.g., 6 joints * 3 coords)
    
    model = RobotPolicy(input_size, output_size)
    
    # Dummy input: (batch_size=8, input_size=450)
    test_input = torch.randn(8, input_size)
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Policy Model Output Shape: {output.shape}")
    print(f"Sample Output Row: {output[0][:5]}")
