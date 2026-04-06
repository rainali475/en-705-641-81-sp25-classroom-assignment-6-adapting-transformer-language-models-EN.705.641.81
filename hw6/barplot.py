import matplotlib.pyplot as plt
import numpy as np

# Data 
models = ['distilbert-base-uncased', 'roberta-base']
dev_acc = [0.705, 0.6217125382262997]
test_acc = [0.706, 0.6257883672039243]

# X locations
x = np.arange(len(models))
width = 0.35  # bar width

# Create plot
fig, ax = plt.subplots()

# Bars
ax.bar(x - width/2, dev_acc, width, label='Dev')
ax.bar(x + width/2, test_acc, width, label='Test')

# Labels and formatting
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Dev vs Test Accuracy by Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Show plot
plt.tight_layout()
plt.savefig("barplot.png")