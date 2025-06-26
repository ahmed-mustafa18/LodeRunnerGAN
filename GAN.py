import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random



class LodeRunnerDataset:
    def __init__(self, processed_path, json_path):
        self.processed_path = processed_path
        self.json_path = json_path
        self.levels = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.level_shape = None
        self.tile_mappings = {}

        self.load_tile_mappings()
        self.load_levels()
        self.create_char_mappings()

    def load_tile_mappings(self):
        """Load tile mappings from JSON file"""
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
                self.tile_mappings = data.get('tiles', {})
            print(f"Loaded tile mappings for {len(self.tile_mappings)} tile types")
        except FileNotFoundError:
            print(f"Warning: Could not find {self.json_path}, using default mappings")
            # Default mappings if JSON not found
            self.tile_mappings = {
                "B": ["solid", "ground"],
                "b": ["solid", "diggable", "ground"],
                ".": ["passable", "empty"],
                "-": ["passable", "climbable", "rope"],
                "#": ["passable", "climbable", "ladder"],
                "G": ["passable", "pickupable", "gold"],
                "E": ["damaging", "enemy"],
                "M": ["passable", "spawn"]
            }

    def get_tile_color_code(self, char):
        """Get ANSI color code for a tile character"""
        color_map = {
            'E': '\033[95m',  # Purple for enemies
            'M': '\033[92m',  # Green for spawn
            '-': '\033[97m',  # White for rope
            '#': '\033[97m',  # White for ladder
            'G': '\033[93m',  # Gold/Yellow for gold
            'b': '\033[33m',  # Brown for diggable ground
            'B': '\033[91m',  # Red for solid ground
            '.': '\033[90m'  # Dark gray for empty (close to black)
        }
        return color_map.get(char, '\033[0m')  # Default to reset

    def get_tile_html_color(self, char):
        """Get HTML color for a tile character"""
        color_map = {
            'E': '#800080',  # Purple for enemies
            'M': '#00FF00',  # Green for spawn
            '-': '#FFFFFF',  # White for rope
            '#': '#FFFFFF',  # White for ladder
            'G': '#FFD700',  # Gold for gold
            'b': '#8B4513',  # Brown for diggable ground
            'B': '#FF0000',  # Red for solid ground
            '.': '#000000'  # Black for empty
        }
        return color_map.get(char, '#FFFFFF')

    def load_levels(self):
        """Load all level files as ASCII strings"""
        txt_files = [f for f in os.listdir(self.processed_path) if f.endswith('.txt')]
        txt_files.sort()

        print(f"Loading {len(txt_files)} level files...")

        for filename in txt_files:
            filepath = os.path.join(self.processed_path, filename)
            with open(filepath, 'r') as f:
                lines = [line.rstrip('\n') for line in f.readlines()]

            # Store as list of strings (preserving ASCII)
            if lines:
                self.levels.append(lines)

                # Set level shape from first level
                if self.level_shape is None:
                    self.level_shape = (len(lines), len(lines[0]))
                    print(f"Level dimensions: {self.level_shape[0]} rows x {self.level_shape[1]} columns")

        print(f"Loaded {len(self.levels)} levels")

    def create_char_mappings(self):
        """Create mappings between characters and indices for neural network processing"""
        all_chars = set()
        for level in self.levels:
            for row in level:
                all_chars.update(row)

        # Sort characters for consistent mapping
        sorted_chars = sorted(list(all_chars))
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        print(f"\nFound {len(sorted_chars)} unique characters:")
        for char, idx in self.char_to_idx.items():
            tile_info = self.tile_mappings.get(char, ["unknown"])
            print(f"'{char}' -> {idx} ({', '.join(tile_info)})")

    def levels_to_tensors(self):
        """Convert ASCII levels to numerical tensors for training"""
        numerical_levels = []
        for level in self.levels:
            numerical_level = []
            for row in level:
                numerical_row = [self.char_to_idx[char] for char in row]
                numerical_level.append(numerical_row)
            numerical_levels.append(np.array(numerical_level))
        return np.array(numerical_levels)

    def tensor_to_level(self, tensor):
        """Convert numerical tensor back to ASCII level"""
        level_lines = []
        for row in tensor:
            line = ''.join([self.idx_to_char[int(idx)] for idx in row])
            level_lines.append(line)
        return level_lines

    def save_level_with_colors(self, level_lines, filename_base):
        """Save level in multiple formats with color information"""
        # Save plain ASCII
        ascii_filename = f"{filename_base}.txt"
        with open(ascii_filename, 'w') as f:
            for line in level_lines:
                f.write(line + '\n')

        # Save colored terminal version
        colored_filename = f"{filename_base}_colored.txt"
        with open(colored_filename, 'w') as f:
            f.write("Lode Runner Level - Terminal Colors\n")
            f.write("=" * 50 + "\n\n")
            for line in level_lines:
                colored_line = ""
                for char in line:
                    color_code = self.get_tile_color_code(char)
                    colored_line += f"{color_code}{char}\033[0m"
                f.write(colored_line + '\n')
            f.write("\n" + "=" * 50 + "\n")

        # Save HTML version
        html_filename = f"{filename_base}.html"
        with open(html_filename, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Lode Runner Level</title>
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background-color: #222; 
            color: white; 
            padding: 20px; 
        }
        .level { 
            font-size: 12px; 
            line-height: 1; 
            white-space: pre; 
        }
        .tile { display: inline-block; width: 1ch; }
        .legend { 
            margin-top: 20px; 
            border: 1px solid #666; 
            padding: 10px; 
        }
        .legend-item { 
            display: inline-block; 
            margin: 5px 10px; 
        }
    </style>
</head>
<body>
    <h1>Generated Lode Runner Level</h1>
    <div class="level">""")

            for line in level_lines:
                for char in line:
                    color = self.get_tile_html_color(char)
                    tile_info = self.tile_mappings.get(char, ["unknown"])
                    f.write(
                        f'<span class="tile" style="color: {color}; background-color: {color};" title="{char}: {", ".join(tile_info)}">&nbsp;</span>')
                f.write('\n')

            f.write("""    </div>

    <div class="legend">
        <h3>Tile Legend:</h3>""")

            for char, properties in self.tile_mappings.items():
                color = self.get_tile_html_color(char)
                f.write(f'''
        <div class="legend-item">
            <span style="color: {color}; background-color: {color};">&nbsp;&nbsp;</span>
            <strong>{char}</strong>: {", ".join(properties)}
        </div>''')

            f.write("""
    </div>
</body>
</html>""")

        return ascii_filename, colored_filename, html_filename


class LodeRunnerTorchDataset(Dataset):
    def __init__(self, levels):
        self.levels = torch.LongTensor(levels)

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, idx):
        return self.levels[idx]


class Generator(nn.Module):
    def __init__(self, latent_dim, level_shape, vocab_size):
        super().__init__()
        self.level_shape = level_shape
        self.vocab_size = vocab_size

        # Start with small feature maps
        self.init_h = 4
        self.init_w = 8

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_h * self.init_w),
            nn.ReLU(True)
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, vocab_size, 3, 1, 1)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(-1, 512, self.init_h, self.init_w)
        x = self.conv_layers(x)

        # Resize to exact target dimensions
        x = F.interpolate(x, size=self.level_shape, mode='bilinear', align_corners=False)
        return x


class Discriminator(nn.Module):
    def __init__(self, level_shape, vocab_size):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(vocab_size, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Calculate output size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, vocab_size, level_shape[0], level_shape[1])
            dummy_output = self.conv_layers(dummy_input)
            self.flattened_size = dummy_output.numel() // dummy_output.size(0)

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


class LodeRunnerGAN:
    def __init__(self, level_shape, vocab_size, latent_dim=100, checkpoint_dir='checkpoints'):
        self.level_shape = level_shape
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.generator = Generator(latent_dim, level_shape, vocab_size).to(self.device)
        self.discriminator = Discriminator(level_shape, vocab_size).to(self.device)

        # Training state tracking
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'g_losses': [],
            'd_losses': [],
            'epochs': []
        }

        print(f"Training on device: {self.device}")
        print(f"Checkpoints will be saved to: {checkpoint_dir}")

    def save_checkpoint(self, epoch, optimizer_G, optimizer_D, g_loss, d_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'level_shape': self.level_shape,
            'vocab_size': self.vocab_size,
            'latent_dim': self.latent_dim
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        # Save best model if this is the best so far
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved! (Combined loss: {g_loss + d_loss:.4f})")

        print(f"Checkpoint saved: {checkpoint_path}")

        # Keep only last 5 regular checkpoints to save space
        self.cleanup_old_checkpoints()

    def cleanup_old_checkpoints(self, keep_last=5):
        """Remove old checkpoint files, keeping only the most recent ones"""
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir)
                            if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]

        if len(checkpoint_files) > keep_last:
            # Sort by epoch number
            checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

            # Remove oldest checkpoints
            for old_file in checkpoint_files[:-keep_last]:
                old_path = os.path.join(self.checkpoint_dir, old_file)
                try:
                    os.remove(old_path)
                    print(f"Removed old checkpoint: {old_file}")
                except OSError:
                    pass

    def load_checkpoint(self, checkpoint_path=None, load_optimizers=True):
        """Load model from checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')

        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return None, None, 0

        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            # Try loading with weights_only=True first (safer)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=True. Error: {e}")
            print("Falling back to weights_only=False (use only with trusted checkpoints)")
            # Fallback to weights_only=False for compatibility
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model states
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {'g_losses': [], 'd_losses': [], 'epochs': []})

        # Create optimizers
        optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Load optimizer states if requested and available
        if load_optimizers and 'optimizer_G_state_dict' in checkpoint:
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

        print(f"Checkpoint loaded! Resuming from epoch {self.current_epoch}")
        print(f"Best loss so far: {self.best_loss:.4f}")

        return optimizer_G, optimizer_D, self.current_epoch

    def save_model_weights(self, filepath):
        """Save only model weights (for inference)"""
        weights = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'level_shape': self.level_shape,
            'vocab_size': self.vocab_size,
            'latent_dim': self.latent_dim
        }
        torch.save(weights, filepath)
        print(f"Model weights saved to: {filepath}")

    def load_model_weights(self, filepath):
        """Load only model weights (for inference)"""
        if not os.path.exists(filepath):
            print(f"No weights file found at {filepath}")
            return False

        print(f"Loading model weights from {filepath}")
        try:
            # Try loading with weights_only=True first (safer)
            weights = torch.load(filepath, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=True. Error: {e}")
            print("Falling back to weights_only=False (use only with trusted checkpoints)")
            # Fallback to weights_only=False for compatibility
            weights = torch.load(filepath, map_location=self.device, weights_only=False)

        self.generator.load_state_dict(weights['generator_state_dict'])
        self.discriminator.load_state_dict(weights['discriminator_state_dict'])

        print("Model weights loaded successfully!")
        return True

    def train(self, dataloader, epochs=30, resume_from_checkpoint=True, save_frequency=5,
              g_lr=0.0002, d_lr=0.0001, label_smoothing=0.1, d_steps=1, g_steps=1):
        """Train the GAN with checkpoint support and stability improvements"""

        # Try to resume from checkpoint
        start_epoch = 0
        if resume_from_checkpoint:
            optimizer_G, optimizer_D, start_epoch = self.load_checkpoint()
            if optimizer_G is None:
                # No checkpoint found, create new optimizers with different learning rates
                optimizer_G = optim.Adam(self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
                optimizer_D = optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
            else:
                # Update learning rates for existing optimizers
                for param_group in optimizer_G.param_groups:
                    param_group['lr'] = g_lr
                for param_group in optimizer_D.param_groups:
                    param_group['lr'] = d_lr
                print(f"Updated learning rates: G_lr={g_lr}, D_lr={d_lr}")
        else:
            optimizer_G = optim.Adam(self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
            optimizer_D = optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

        # Check if we've already reached the target epochs
        if start_epoch >= epochs:
            print(f"\nTraining already completed! Current epoch ({start_epoch}) >= target epochs ({epochs})")
            print("To continue training, increase the epochs parameter or set a higher target.")
            print(f"Example: gan.train(dataloader, epochs={start_epoch + 20})")
            return

        criterion = nn.BCELoss()

        print(f"\nStarting training from epoch {start_epoch + 1} to {epochs}")
        print(f"Generator LR: {g_lr}, Discriminator LR: {d_lr}")
        print(f"Label smoothing: {label_smoothing}")
        print(f"Training ratio - D steps: {d_steps}, G steps: {g_steps}")
        print(f"Checkpoints will be saved every {save_frequency} epochs")

        for epoch in range(start_epoch, epochs):
            epoch_g_losses = []
            epoch_d_losses = []

            for i, real_levels in enumerate(dataloader):
                batch_size = real_levels.size(0)

                # Convert to one-hot encoding for discriminator
                real_levels_onehot = F.one_hot(real_levels, num_classes=self.vocab_size).float()
                real_levels_onehot = real_levels_onehot.permute(0, 3, 1, 2).to(self.device)

                # Labels with smoothing for stability
                valid = torch.ones(batch_size, 1, device=self.device) * (1.0 - label_smoothing)
                fake = torch.zeros(batch_size, 1, device=self.device) + (label_smoothing / 2)

                # Train Discriminator (fewer steps to prevent overpowering)
                d_loss_total = 0
                for _ in range(d_steps):
                    optimizer_D.zero_grad()

                    # Real samples
                    real_loss = criterion(self.discriminator(real_levels_onehot), valid)

                    # Fake samples
                    z = torch.randn(batch_size, self.latent_dim, device=self.device)
                    fake_levels = self.generator(z)
                    fake_levels_softmax = F.softmax(fake_levels, dim=1)
                    fake_loss = criterion(self.discriminator(fake_levels_softmax.detach()), fake)

                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    optimizer_D.step()
                    d_loss_total += d_loss.item()

                # Train Generator (more steps to help it catch up)
                g_loss_total = 0
                for _ in range(g_steps):
                    optimizer_G.zero_grad()
                    z = torch.randn(batch_size, self.latent_dim, device=self.device)
                    fake_levels = self.generator(z)
                    fake_levels_softmax = F.softmax(fake_levels, dim=1)

                    # Use stronger "real" labels for generator training
                    strong_valid = torch.ones(batch_size, 1, device=self.device)
                    g_loss = criterion(self.discriminator(fake_levels_softmax), strong_valid)
                    g_loss.backward()
                    optimizer_G.step()
                    g_loss_total += g_loss.item()

                # Track average losses
                epoch_g_losses.append(g_loss_total / g_steps)
                epoch_d_losses.append(d_loss_total / d_steps)

                if i % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                          f"D_loss: {d_loss_total / d_steps:.4f} G_loss: {g_loss_total / g_steps:.4f}")

            # Calculate average losses for this epoch
            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            combined_loss = avg_g_loss + avg_d_loss

            # Update training history
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['g_losses'].append(avg_g_loss)
            self.training_history['d_losses'].append(avg_d_loss)

            # Check if this is the best model so far
            is_best = combined_loss < self.best_loss
            if is_best:
                self.best_loss = combined_loss

            print(f"\n--- Epoch {epoch + 1} Summary ---")
            print(f"Average G_loss: {avg_g_loss:.4f}")
            print(f"Average D_loss: {avg_d_loss:.4f}")
            print(f"Combined loss: {combined_loss:.4f}")
            print(f"Loss ratio (G/D): {avg_g_loss / max(avg_d_loss, 1e-8):.2f}")
            if is_best:
                print("New best model!")

            # Adaptive learning rate adjustment based on loss ratio
            loss_ratio = avg_g_loss / max(avg_d_loss, 1e-8)
            if loss_ratio > 100:  # Generator struggling too much
                for param_group in optimizer_D.param_groups:
                    param_group['lr'] *= 0.95  # Slow down discriminator
                print("Reduced discriminator learning rate")
            elif loss_ratio < 10:  # Discriminator struggling
                for param_group in optimizer_G.param_groups:
                    param_group['lr'] *= 1.05  # Speed up generator slightly
                print("Increased generator learning rate slightly")

            # Save checkpoint
            if (epoch + 1) % save_frequency == 0 or epoch == epochs - 1 or is_best:
                self.save_checkpoint(epoch + 1, optimizer_G, optimizer_D, avg_g_loss, avg_d_loss, is_best)

            self.current_epoch = epoch + 1

        # Save final model weights
        final_weights_path = os.path.join(self.checkpoint_dir, 'final_model_weights.pth')
        self.save_model_weights(final_weights_path)

        # Save training history
        self.save_training_history()

        print(f"\nTraining completed!")
        print(f"Best combined loss achieved: {self.best_loss:.4f}")
        print(f"Final model weights saved to: {final_weights_path}")

    def save_training_history(self):
        """Save training history as JSON and plot"""
        import matplotlib.pyplot as plt

        # Save as JSON
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        # Create training plots
        if len(self.training_history['epochs']) > 0:
            plt.figure(figsize=(12, 5))

            # Loss plot
            plt.subplot(1, 2, 1)
            plt.plot(self.training_history['epochs'], self.training_history['g_losses'],
                     label='Generator Loss', color='blue')
            plt.plot(self.training_history['epochs'], self.training_history['d_losses'],
                     label='Discriminator Loss', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend()
            plt.grid(True)

            # Combined loss plot
            plt.subplot(1, 2, 2)
            combined_losses = [g + d for g, d in zip(self.training_history['g_losses'],
                                                     self.training_history['d_losses'])]
            plt.plot(self.training_history['epochs'], combined_losses,
                     label='Combined Loss', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Combined Loss')
            plt.title('Combined Loss (G + D)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plot_path = os.path.join(self.checkpoint_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Training history saved to: {history_path}")
            print(f"Training curves saved to: {plot_path}")

    def generate_level(self, dataset):
        """Generate a new level"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(1, self.latent_dim, device=self.device)
            generated = self.generator(z)

            # Convert to character indices
            char_indices = torch.argmax(generated, dim=1).squeeze().cpu().numpy()

            # Convert back to ASCII level
            level_lines = dataset.tensor_to_level(char_indices)

        return level_lines


def validate_level(level_lines, dataset):
    """Enhanced validation and post-processing of generated level"""
    print("Validating and post-processing level...")

    # Count current tile types
    tile_counts = {}
    for line in level_lines:
        for char in line:
            tile_counts[char] = tile_counts.get(char, 0) + 1

    print("Current tile distribution:")
    for char, count in sorted(tile_counts.items()):
        tile_info = dataset.tile_mappings.get(char, ["unknown"])
        print(f"  '{char}' ({', '.join(tile_info)}): {count}")

    # Ensure we have a spawn point
    has_spawn = any('M' in line for line in level_lines)
    if not has_spawn:
        print("Adding spawn point...")
        # Add spawn point at bottom
        line_list = list(level_lines[-2])
        line_list[len(line_list) // 2] = 'M'
        level_lines[-2] = ''.join(line_list)

    # Ensure we have some gold
    gold_count = sum(line.count('G') for line in level_lines)
    if gold_count < 3:
        print(f"Adding gold (current: {gold_count}, target: 3)...")
        # Add some gold
        for _ in range(3 - gold_count):
            row_idx = random.randint(1, len(level_lines) - 2)
            line_list = list(level_lines[row_idx])
            for col_idx in range(len(line_list)):
                if line_list[col_idx] == '.':
                    line_list[col_idx] = 'G'
                    break
            level_lines[row_idx] = ''.join(line_list)

    # Ensure we have some enemies
    enemy_count = sum(line.count('E') for line in level_lines)
    if enemy_count < 2:
        print(f"Adding enemies (current: {enemy_count}, target: 2)...")
        for _ in range(2 - enemy_count):
            row_idx = random.randint(1, len(level_lines) - 3)
            line_list = list(level_lines[row_idx])
            for col_idx in range(len(line_list)):
                if line_list[col_idx] == '.':
                    line_list[col_idx] = 'E'
                    break
            level_lines[row_idx] = ''.join(line_list)

    return level_lines


def load_pretrained_model(dataset, checkpoint_dir='checkpoints', model_filename='best_model.pth'):
    """Load a pre-trained model for generation only"""
    gan = LodeRunnerGAN(
        level_shape=dataset.level_shape,
        vocab_size=len(dataset.char_to_idx),
        latent_dim=100,
        checkpoint_dir=checkpoint_dir
    )

    model_path = os.path.join(checkpoint_dir, model_filename)
    if gan.load_model_weights(model_path):
        return gan
    else:
        print("Failed to load pre-trained model. Please train a model first.")
        return None


def main():
    # Paths
    processed_path = ''
    json_path = ''
    checkpoint_dir = ''

    # Load dataset
    print("Loading Lode Runner dataset...")
    dataset = LodeRunnerDataset(processed_path, json_path)

    # Convert to numerical format for training
    numerical_levels = dataset.levels_to_tensors()
    torch_dataset = LodeRunnerTorchDataset(numerical_levels)
    dataloader = DataLoader(torch_dataset, batch_size=4, shuffle=True)

    # Initialize GAN with checkpoint support
    gan = LodeRunnerGAN(
        level_shape=dataset.level_shape,
        vocab_size=len(dataset.char_to_idx),
        latent_dim=100,
        checkpoint_dir=checkpoint_dir
    )

    # Option to skip training if you have a pre-trained model
    train_model = True  # Set to False to skip training and use existing model

    if train_model:
        # Train the model with checkpoint support and stability improvements
        print("\nTraining GAN...")
        print("Note: Training will resume from last checkpoint if available")
        gan.train(dataloader, epochs=1000, resume_from_checkpoint=True, save_frequency=10,
                  g_lr=0.0003, d_lr=0.0001, label_smoothing=0.1, d_steps=1, g_steps=2)
    else:
        # Load existing model
        print("\nLoading pre-trained model...")
        if not gan.load_model_weights(os.path.join(checkpoint_dir, 'best_model.pth')):
            print("No pre-trained model found. Please set train_model=True to train first.")
            return

    # Generate new levels
    print("\nGenerating new levels...")
    for i in range(5):
        print(f"\n--- Generating Level {i + 1} ---")
        level_lines = gan.generate_level(dataset)
        level_lines = validate_level(level_lines, dataset)

        # Save to multiple formats
        filename_base = f'generated_level_{i + 1}'
        ascii_file, colored_file, html_file = dataset.save_level_with_colors(level_lines, filename_base)

        print(f"Generated Level {i + 1} saved as:")
        print(f"  - ASCII: {ascii_file}")
        print(f"  - Colored Terminal: {colored_file}")
        print(f"  - HTML: {html_file}")

        print("\nPreview (first 8 lines):")
        for line in level_lines[:8]:
            colored_line = ""
            for char in line:
                color_code = dataset.get_tile_color_code(char)
                colored_line += f"{color_code}{char}\033[0m"
            print(colored_line)
        print("...")

    print("\nLevel generation complete!")
    print(f"\nModel files saved in: {checkpoint_dir}/")
    print("  - best_model.pth: Best performing model")
    print("  - final_model_weights.pth: Final model after training")
    print("  - latest_checkpoint.pth: Most recent checkpoint")
    print("  - training_history.json: Loss history data")
    print("  - training_curves.png: Training visualization")

    print("\nTile Color Mapping:")
    print("  Enemies (E): Purple")
    print("  Spawn (M): Green")
    print("  Rope (-): White")
    print("  Ladder (#): White")
    print("  Gold (G): Gold/Yellow")
    print("  Diggable Ground (b): Brown")
    print("  Solid Ground (B): Red")
    print("  Empty (.): Black")


def generate_only_mode():
    """Function to only generate levels from pre-trained model"""
    processed_path = ''
    json_path = ''
    checkpoint_dir = ''

    print("=== Generation Only Mode ===")
    dataset = LodeRunnerDataset(processed_path, json_path)

    gan = load_pretrained_model(dataset, checkpoint_dir)
    if gan is None:
        return

    num_levels = int(input("How many levels to generate? (default: 3): ") or "3")

    for i in range(num_levels):
        print(f"\n--- Generating Level {i + 1} ---")
        level_lines = gan.generate_level(dataset)
        level_lines = validate_level(level_lines, dataset)

        filename_base = f'generated_level_{i + 1}'
        ascii_file, colored_file, html_file = dataset.save_level_with_colors(level_lines, filename_base)

        print(f"Level {i + 1} saved as: {ascii_file}, {colored_file}, {html_file}")

    print(f"\nGenerated {num_levels} levels successfully!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--generate-only':
        generate_only_mode()
    else:
        main()
