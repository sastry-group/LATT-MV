import numpy as np
import pickle 

targets = pickle.load(open('targets.pkl', 'rb'))

# Extract the target values
print(targets)