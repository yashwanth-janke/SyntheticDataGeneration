from src.data_preprocessing import load_and_preprocess_data
from src.vae_model import build_vae
from src.utils import generate_and_plot_synthetic_data
import pandas as pd

def main():
    # Load and preprocess the dataset
    df = pd.read_csv('Data/Raw/ckd.csv')
    X_train, X_test = load_and_preprocess_data(df)

    # Build and train the VAE
    encoder, decoder, vae = build_vae(X_train.shape[1], latent_dim=2)
    vae.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

    # Save the models
    encoder.save('Model/encoder.h5')
    decoder.save('Model/decoder.h5')
    vae.save('Model/vae.h5')

    # Generate synthetic data and save it to a CSV file
    synthetic_data_path = 'Results/synthetic_data.csv'
    generate_and_plot_synthetic_data(decoder, num_samples=100, save_path=synthetic_data_path)

if __name__ == "__main__":
    main()