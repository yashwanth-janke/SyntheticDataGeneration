from data_preprocessing import load_and_preprocess_data
from vae_model import build_vae
import pandas as pd

df = pd.read_csv('../Data/Raw/ckd.csv')
X_train, X_test = load_and_preprocess_data(df)

encoder, decoder, vae = build_vae(X_train.shape[1], latent_dim=2)
vae.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

encoder.save('../Model/encoder.h5')
decoder.save('../Model/decoder.h5')
vae.save('../Model/vae.h5')
