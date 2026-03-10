import matplotlib.pyplot as plt

# Set the academic font for IEEE/Springer standards
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

def plot_efficiency_tradeoff(tinyllm_latency):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    
    models = ['DistilBERT', 'TinyLLM\n(Ours)']
    # Academic grayscale color palette
    colors = ['#8C8C8C', '#1A1A1A'] 
    
    # --- Subplot 1: Storage Size (MB) ---
    # DistilBERT is ~260MB, your TinyLLM is ~7.2MB
    sizes = [260, 7.2] 
    bars1 = ax1.bar(models, sizes, color=colors, edgecolor='black', width=0.5)
    ax1.set_title('Storage Footprint', fontweight='bold', pad=10)
    ax1.set_ylabel('Size (MB)', fontweight='bold')
    ax1.set_ylim(0, 320)
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Add numerical labels on top of the bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval} MB", ha='center', va='bottom', fontweight='bold')

    # --- Subplot 2: Inference Latency (ms) ---
    # Standard DistilBERT CPU latency is ~120ms
    latencies = [120, tinyllm_latency] 
    bars2 = ax2.bar(models, latencies, color=colors, edgecolor='black', width=0.5)
    ax2.set_title('Inference Latency (CPU)', fontweight='bold', pad=10)
    ax2.set_ylabel('Latency (ms)', fontweight='bold')
    ax2.set_ylim(0, 150) 
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Add numerical labels on top of the bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{round(yval, 1)} ms", ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Efficiency Trade-off: Pre-trained SOTA vs. Custom TLM', fontweight='bold', fontsize=14, y=1.05)
    plt.tight_layout()
    
    # Save as high-res PNG for LaTeX
    plt.savefig('efficiency_chart.png', dpi=300, bbox_inches='tight')
    print("Saved high-resolution chart to 'efficiency_chart.png'")
    plt.show()

if __name__ == "__main__":
    # Measured latency for your specific 4-layer architecture on MacBook Air
    actual_tinyllm_latency_ms = 14.2  
    
    plot_efficiency_tradeoff(actual_tinyllm_latency_ms)