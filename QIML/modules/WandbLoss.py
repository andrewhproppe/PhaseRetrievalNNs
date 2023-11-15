import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class WandbLoss:
    def __init__(self, filename, root_path= "../../data/training_losses/"):
        # Construct the full file path
        csv_file_path = os.path.join(root_path, filename)

        # Read the CSV file into a pandas DataFrame
        self.data = pd.read_csv(csv_file_path)

        # Extract relevant part of column names and set attributes
        for col in self.data.columns:
            if '__MAX' not in col and '__MIN' not in col:
                # Extract the part after the last '-' and remove leading/trailing spaces
                attribute_name = col.split('-')[-1].strip()
                setattr(self, attribute_name, np.array(self.data[col]))

    def plot(self, attribute_name, fig, label=None):
        # Check if the specified attribute exists
        if hasattr(self, attribute_name):
            # Get the attribute values
            attribute_values = getattr(self, attribute_name)

            # Find non-NaN entries
            non_nan_indices = ~np.isnan(attribute_values)

            # Plot the attribute versus number of steps for non-NaN entries
            plt.plot(self.data['Step'][non_nan_indices], attribute_values[non_nan_indices], label=label)
        else:
            print(f'Attribute {attribute_name} does not exist.')