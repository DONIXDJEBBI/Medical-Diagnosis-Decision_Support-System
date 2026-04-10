from graphviz import Digraph

# Create a directed graph for workflow
dot = Digraph(comment='System Architecture Workflow', graph_attr={'rankdir': 'TB'})
dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Arial', fontsize='10')

# Step 1: Patient Input
dot.node('1_patient', 'PATIENT INPUT\n(Symptoms & Lab Tests)\n\nFever, Cough, WBC,\nCRP, SpO2, X-ray, etc.', fillcolor='#FFE4B5')

# Step 2: Data Distribution
dot.node('2_data', 'COLLECT PATIENT DATA', fillcolor='#FFE4B5')

# Step 3: Parallel Processing
dot.node('3_fuzzy', 'FUZZY LOGIC MODEL\n\nApplies medical expert rules\nfor symptom analysis', fillcolor='#87CEEB')
dot.node('3_tree', 'DECISION TREE MODEL\n\nLearns patterns from data\nfor classification', fillcolor='#98FB98')

# Step 4: Model Outputs
dot.node('4_fuzzy_out', 'FUZZY DIAGNOSIS\n\nDisease scores & confidence\nBased on fuzzy rules', fillcolor='#87CEEB')
dot.node('4_tree_out', 'ML DIAGNOSIS\n\nDisease prediction & probability\nBased on learned patterns', fillcolor='#98FB98')

# Step 5: Comparison
dot.node('5_compare', 'COMPARE RESULTS\n\nAnalyze model outputs\nCheck agreement level', fillcolor='#DDA0DD')

# Step 6: Decision Support
dot.node('6_agreement', 'AGREEMENT CHECK\n\nModels agree?\n✓ High confidence\n✗ Investigate difference', fillcolor='#F0E68C')

# Step 7: Final Output
dot.node('7_output', 'FINAL RESULTS & METRICS\n\nDiagnosis recommendation\nConfidence scores\nPerformance metrics', fillcolor='#FFB6C1')

# Edges - Define the workflow
dot.edge('1_patient', '2_data', label='Enter Data')
dot.edge('2_data', '3_fuzzy', label='Pass to Model')
dot.edge('2_data', '3_tree', label='Pass to Model')

dot.edge('3_fuzzy', '4_fuzzy_out', label='Inference')
dot.edge('3_tree', '4_tree_out', label='Prediction')

dot.edge('4_fuzzy_out', '5_compare', label='Result 1')
dot.edge('4_tree_out', '5_compare', label='Result 2')

dot.edge('5_compare', '6_agreement', label='Analysis')
dot.edge('6_agreement', '7_output', label='Decision')

# Save and render as PNG
dot.render('system_architecture', view=True, format='png')
print('System architecture workflow diagram generated as system_architecture.png')
