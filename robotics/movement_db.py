import numpy as np
import os
import uuid

class MovementDatabase:
    """
    A persistent library for movement sequences stored as numpy files.
    Allows for training data collection and imitation learning datasets.
    """

    def __init__(self, path="movements"):
        """Initialize the database directory."""
        self.base_path = path
        os.makedirs(self.base_path, exist_ok=True)

    def save_movement(self, sequence, label):
        """
        Save a sequence of encoded motion vectors with a categorical label.
        
        Args:
            sequence (list or np.ndarray): Sequence of motion vectors.
            label (str): Movement type (e.g., "GRAB_OBJECT", "WAVE_HAND").
        """
        label_dir = os.path.join(self.base_path, label)
        os.makedirs(label_dir, exist_ok=True)

        # Generate a unique filename to avoid collisions
        filename = f"m_{uuid.uuid4().hex[:8]}.npy"
        file_path = os.path.join(label_dir, filename)

        np.save(file_path, np.array(sequence))
        print(f"Saved {label} sequence to {file_path}")

    def load_movement_by_label(self, label):
        """
        Load all stored movement sequences for a specific label.
        
        Returns:
            list: List of movement sequences (numpy arrays).
        """
        label_dir = os.path.join(self.base_path, label)
        if not os.path.exists(label_dir):
            return []

        movements = []
        for file in os.listdir(label_dir):
            if file.endswith(".npy"):
                data = np.load(os.path.join(label_dir, file))
                movements.append(data)
        
        return movements

    def get_labels(self):
        """List all available movement types in the database."""
        return [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]

if __name__ == "__main__":
    # Unit Test
    db = MovementDatabase(path="test_movements")
    mock_seq = np.random.rand(10, 12) # 10 frames, 12 features
    
    db.save_movement(mock_seq, "TEST_MOVE")
    loaded = db.load_movement_by_label("TEST_MOVE")
    
    print(f"Loaded {len(loaded)} sequences.")
    print(f"Shape of first sequence: {loaded[0].shape}")
    print(f"Available Labels: {db.get_labels()}")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_movements")
