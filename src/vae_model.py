import tensorflow as tf
from tensorflow.keras import layers

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, latent_dim):
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(encoder_inputs)
    x = layers.Dense(64, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation='relu')(latent_inputs)
    x = layers.Dense(128, activation='relu')(x)
    decoder_outputs = layers.Dense(input_dim, activation='sigmoid')(x)

    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name='decoder')

    outputs = decoder(encoder(encoder_inputs)[2])
    vae = tf.keras.Model(encoder_inputs, outputs, name='vae')

    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(encoder_inputs, outputs)
    )
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return encoder, decoder, vae
