import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

decoder = load_model('../Model/decoder.h5', compile=False)

def generate_synthetic_data(num_samples, latent_dim):
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
    generated_data = decoder.predict(random_latent_vectors)
    return generated_data

synthetic_data = generate_synthetic_data(100, latent_dim=2)
synthetic_df = pd.DataFrame(synthetic_data, columns=[f'feature_{i}' for i in range(synthetic_data.shape[1])])
synthetic_df.to_csv('../Results/synthetic_data.csv', index=False)
