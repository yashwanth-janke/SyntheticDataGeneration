import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_and_plot_synthetic_data(decoder, num_samples, save_path=None):
    latent_dim = decoder.input_shape[1]
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
    generated_data = decoder.predict(random_latent_vectors)
    
    # Converting the generated data to a DataFrame
    generated_df = pd.DataFrame(generated_data, columns=[f'feature_{i}' for i in range(generated_data.shape[1])])
    
    # Plotting the generated data
    plt.figure(figsize=(10, 6))
    for i in range(min(num_samples, 10)):  
        plt.subplot(2, 5, i + 1)
        plt.plot(generated_data[i])  
        plt.title(f'Sample {i + 1}')
        plt.axis('off')
    plt.show()
    
    
    if save_path:
        generated_df.to_csv(save_path, index=False)
        print(f"Generated data saved to {save_path}")
    
    return generated_data