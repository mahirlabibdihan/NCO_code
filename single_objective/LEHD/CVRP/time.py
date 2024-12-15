import matplotlib.pyplot as plt

# Example data
node_counts = [100, 200, 500, 1000]
times = [0.052, 0.328, 2.634, 14.43]  # Example time values in seconds

# Create the plot
plt.figure(figsize=(12, 7), dpi=100)

# Create the line plot
plt.plot(node_counts, times, marker='o', linestyle='-', linewidth=2, markersize=8, 
         color='#1E90FF', markeredgecolor='darkblue', markerfacecolor='lightblue')

# Add titles and labels
plt.title('Computational Time vs Node Count', 
          fontsize=16, fontweight='bold', color='darkslategray')
plt.xlabel('Number of Nodes', fontsize=14, fontweight='semibold', color='darkslategray')
plt.ylabel('Time (seconds)', fontsize=14, fontweight='semibold', color='darkslategray')

# Add grid for readability
plt.grid(True, linestyle='--', alpha=0.7, color='gray')

# Set background color
plt.gca().set_facecolor('#f0f0f0')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('node_count_vs_time_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()