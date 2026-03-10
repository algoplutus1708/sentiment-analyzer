import matplotlib.pyplot as plt
import numpy as np

# Set the academic font
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

def plot_loss_curve(epochs, train_loss, val_loss):
    plt.figure(figsize=(6, 5))
    
    # Plotting the lines using your actual terminal data
    plt.plot(epochs, train_loss, label='Training Loss', color='black', linestyle='-', marker='o', linewidth=1.5)
    plt.plot(epochs, val_loss, label='Validation Loss', color='gray', linestyle='--', marker='s', linewidth=1.5)
    
    # Formatting the graph for a research paper
    plt.title('Model Convergence: Training vs. Validation Loss', fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Cross-Entropy Loss', fontweight='bold')
    
    # Force x-axis to show the 8 integer epochs
    plt.xticks(epochs)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', frameon=True, edgecolor='black')
    
    # Save as high-res PNG for LaTeX/Word
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    print("Graph generated and saved as 'loss_curve.png'")
    plt.show()

if __name__ == "__main__":
    # Actual values extracted from your terminal output
    epochs = np.arange(1, 9)
    
    # Values from: Train Loss: [NUMBER]
    actual_train_loss = [0.4887, 0.2841, 0.2031, 0.1589, 0.1393, 0.1298, 0.1266, 0.1234]
    
    # Values from: Test Loss: [NUMBER]
    actual_val_loss =   [0.4270, 0.4575, 0.5556, 0.5870, 0.5905, 0.6101, 0.6096, 0.6159] 
    
    plot_loss_curve(epochs, actual_train_loss, actual_val_loss)