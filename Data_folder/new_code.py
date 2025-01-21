import tensorflow as tf
from dafne_dl.DynamicDLModel import source_to_fn, DynamicDLModel

# Load the pre-trained model
with open('/Users/dibya/dafne/MyThesisDatasets/CHAOS_Dataset_for_Dibya/chaos.model', 'rb') as f:
    old_dafne_model = DynamicDLModel.Load(f)

# Access the Keras model from the loaded DynamicDLModel
old_model = old_dafne_model.model

# Print the total number of layers
print('Total layers:', len(old_model.layers))
# Print the names of all layers
print("Layer Names:")
for i, layer in enumerate(old_model.layers):
    print(f"Layer {i}: {layer.name}")
