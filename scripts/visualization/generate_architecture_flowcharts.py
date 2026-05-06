import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def create_flowchart(filename, title, nodes, edges):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.axis('off')
    
    plt.text(0.5, 1.02, title, ha='center', va='center', fontsize=16, fontweight='bold')
    
    for name, (x, y, text, width, height, color) in nodes.items():
        ellipse = mpatches.Ellipse((x, y), width, height,
                                   edgecolor='#85A5D4', facecolor=color, lw=2)
        ax.add_patch(ellipse)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='#212121')
        
    for n1, n2 in edges:
        # Ellipse boundary approximation for arrows
        y_offset_1 = nodes[n1][4]/2 * 0.95
        y_offset_2 = nodes[n2][4]/2 * 0.95
        
        x1, y1 = nodes[n1][0], nodes[n1][1] - y_offset_1
        x2, y2 = nodes[n2][0], nodes[n2][1] + y_offset_2
        
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2, mutation_scale=20))

    output_path = os.path.join("../../results", filename)
    if not os.path.exists("../../results"):
        output_path = os.path.join("e:/FRP PROJECT/Aneamia Detection/results", filename)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# --- Reference Architecture Nodes ---
ref_nodes = {
    'train': (0.5, 0.92, "Original Training Set\n(X_train, y_train)", 0.40, 0.10, '#E3F2FD'),
    'rf': (0.25, 0.77, "Train Random Forest\non X_train", 0.45, 0.10, '#E3F2FD'),
    'ann': (0.75, 0.77, "Train Artificial Neural Network\non X_train", 0.45, 0.10, '#E3F2FD'),
    'p_rf': (0.25, 0.62, "RF Probabilities\nP_RF", 0.40, 0.10, '#E3F2FD'),
    'p_ann': (0.75, 0.62, "ANN Probabilities\nP_ANN", 0.40, 0.10, '#E3F2FD'),
    'meta': (0.5, 0.47, "Build Meta-Feature\nP_RF + P_ANN", 0.45, 0.10, '#E3F2FD'),
    'xgb': (0.5, 0.27, "Train XGBoost Meta-Model\non Meta Features", 0.45, 0.10, '#E3F2FD'),
    'pred': (0.5, 0.12, "Final Anemia\nPrediction", 0.40, 0.10, '#E3F2FD')
}

ref_edges = [
    ('train', 'rf'), ('train', 'ann'),
    ('rf', 'p_rf'), ('ann', 'p_ann'),
    ('p_rf', 'meta'), ('p_ann', 'meta'),
    ('meta', 'xgb'), ('xgb', 'pred')
]

create_flowchart("reference_architecture_flowchart.png", "Reference Hybrid Architecture (From Tanzania Study)", ref_nodes, ref_edges)

# --- Proposed Architecture Nodes ---
prop_nodes = {
    'train': (0.5, 0.95, "Original Training Set\n(X_train, y_train)", 0.35, 0.08, '#E8F5E9'),
    'smote': (0.5, 0.84, "Apply SMOTE\n(Synthesize Minority Class)", 0.35, 0.08, '#C8E6C9'),
    'rf': (0.2, 0.71, "Train Random Forest\non Balanced Data", 0.28, 0.08, '#E8F5E9'),
    'ann': (0.5, 0.71, "Train Neural Network\non Balanced Data", 0.28, 0.08, '#E8F5E9'),
    'lgbm': (0.8, 0.71, "Train LightGBM\non Balanced Data", 0.28, 0.08, '#E8F5E9'),
    'p_rf': (0.2, 0.58, "RF Probabilities\nP_RF", 0.28, 0.08, '#E8F5E9'),
    'p_ann': (0.5, 0.58, "ANN Probabilities\nP_ANN", 0.28, 0.08, '#E8F5E9'),
    'p_lgbm': (0.8, 0.58, "LGBM Probabilities\nP_LGBM", 0.28, 0.08, '#E8F5E9'),
    'meta': (0.5, 0.45, "Build Meta-Feature\nP_RF + P_ANN + P_LGBM", 0.45, 0.08, '#C8E6C9'),
    'cat': (0.5, 0.30, "Train CatBoost Meta-Model\non Meta Features", 0.45, 0.08, '#A5D6A7'),
    'pred': (0.5, 0.15, "Final Anemia Prediction\n(Balanced Detection)", 0.45, 0.08, '#E8F5E9')
}

prop_edges = [
    ('train', 'smote'),
    ('smote', 'rf'), ('smote', 'ann'), ('smote', 'lgbm'),
    ('rf', 'p_rf'), ('ann', 'p_ann'), ('lgbm', 'p_lgbm'),
    ('p_rf', 'meta'), ('p_ann', 'meta'), ('p_lgbm', 'meta'),
    ('meta', 'cat'), ('cat', 'pred')
]

create_flowchart("proposed_architecture_flowchart.png", "Proposed Enhanced Architecture (SMOTE + CatBoost)", prop_nodes, prop_edges)
