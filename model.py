import tensorflow as tf

# Network
latent_dim = 128
dropout = 0.2
leaky_relu_alpha = 0.2

# CNN
channels = 1
filters = 32
kernel_size = 3
strides = 2
padding = 'same'

# Loss Coefficients
kl_loss_coeff = 1
perc_loss_coeff = 1

class VAEWGANGP(tf.keras.Model):
    def __init__(self, encoder, decoder, discriminator, gp_weight=10.0):
        super(VAEWGANGP, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.gp_weight = gp_weight

    def compile(self, e_optimizer, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, kl_loss_fn, perc_loss_fn, rec_loss_fn):
        super(VAEWGANGP, self).compile()
        self.e_optimizer = e_optimizer
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.kl_loss_fn = kl_loss_fn
        self.perc_loss_fn = perc_loss_fn
        self.rec_loss_fn = rec_loss_fn
        
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.perc_loss_metric = tf.keras.metrics.Mean(name = 'perc_loss')
        self.kl_loss_metric = tf.keras.metrics.Mean(name = 'kl_loss')
        self.rec_loss_metric = tf.keras.metrics.Mean(name = 'rec_loss')
        
    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            _, pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    
    def call(self, data, training=False):
    # This method exists only because Keras expects it to be able to use data_generator()
    # your custom code when you call the model
    # or just pass, you don't need this method
    # for training
        pass

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric, self.perc_loss_metric, self.kl_loss_metric, self.rec_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        real_images = tf.cast(real_images, tf.float32)
        # ===========================================================================
        # Train Discriminator
        # For WGAN, it is advised to train this multiple times before training generator
        for _ in range(5):
            _, _, z_encoder_output = self.encoder(real_images)
            with tf.GradientTape() as tape:
                fake_images = self.decoder(z_encoder_output, training = True)
                _, logits_fake_images = self.discriminator(fake_images, training = True)
                _, logits_real_images = self.discriminator(real_images, training = True)

                d_cost = self.d_loss_fn(logits_real_images, logits_fake_images)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            
        # ===========================================================================
        # Training Encoder
        with tf.GradientTape() as tape:
            z_mean, z_log_sigma, z_encoder_output = self.encoder(real_images, training = True)
            kl_loss = self.kl_loss_fn(z_mean, z_log_sigma) * kl_loss_coeff
            
            fake_images = self.decoder(z_encoder_output, training = True)
            fake_inter_activations, logits_fake = self.discriminator(fake_images, training = True)
            real_inter_activations, logits_real = self.discriminator(real_images, training = True)
            
            perc_loss = self.perc_loss_fn(fake_inter_activations, real_inter_activations) * perc_loss_coeff

            total_encoder_loss = kl_loss + perc_loss
            
        grads = tape.gradient(total_encoder_loss, self.encoder.trainable_weights)
        self.e_optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))
        
        # ===========================================================================
        # Train Decoder
        _, _, z_encoder_output = self.encoder(real_images)
        with tf.GradientTape() as tape:
            fake_images = self.decoder(z_encoder_output)
            fake_inter_activations, logits_fake = self.discriminator(fake_images, training = True)
            real_inter_activations, _ = self.discriminator(real_images, training = True)
            
            g_loss = self.g_loss_fn(logits_fake, 0)
            perc_loss = self.perc_loss_fn(fake_inter_activations, real_inter_activations)
            
            total_decoder_loss = g_loss + perc_loss * perc_loss_coeff

        grads = tape.gradient(total_decoder_loss, self.decoder.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.decoder.trainable_weights))
        
        # Lasty, compute reconstruction loss for reporting purposes
        rec_loss = self.rec_loss_fn(real_images, fake_images)
        
        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.perc_loss_metric.update_state(perc_loss)
        self.kl_loss_metric.update_state(kl_loss)
        self.rec_loss_metric.update_state(rec_loss)
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "perc_loss": self.perc_loss_metric.result(),
            "kl_loss": self.kl_loss_metric.result(),
            "rec_loss": self.rec_loss_metric.result()
        }

class CustomKLLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_kl_loss"):
        super().__init__(name=name)

    def call(self, z_mean, z_log_sigma):
        kl_loss = - 0.5 * tf.keras.backend.mean(1 + z_log_sigma - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_sigma))
        return kl_loss
    
# L1 Norm loss
class CustomL1NormLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_L1_loss"):
        super().__init__(name=name)

    def call(self, z1, z2):
        diff = z1 - z2
        abs_ = tf.keras.backend.abs(diff)
        return tf.keras.backend.sum(abs_)
    
class CustomDWLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_d_wasserstein_loss"):
        super().__init__(name=name)

    def call(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss
    
class CustomGWLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_g_wasserstein_loss"):
        super().__init__(name=name)

    def call(self, fake_img, a):
        return -tf.reduce_mean(fake_img)


def vae_sampling(args):
    z_mean, z_log_sigma = args
    batch_size = tf.shape(z_mean)[0]
    latent_dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape = (batch_size, latent_dim), mean = 0, stddev = 0.1)
    
    return z_mean + tf.keras.backend.exp(z_log_sigma / 2) * epsilon

def create_vaegan_networks(window_size, channels, latent_dim,
                            filters, kernel_size, strides, padding, 
                            leaky_relu_alpha, dropout):

    # Discriminator
    # =========================================================================================
    disc_input = tf.keras.layers.Input(shape=(window_size, 128, channels))

    disc_conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)(disc_input)
    disc_conv = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(disc_conv)

    disc_conv = tf.keras.layers.Conv2D(filters * 2, kernel_size, strides, padding)(disc_conv)
    disc_conv = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(disc_conv)

    disc_conv = tf.keras.layers.Conv2D(filters * 4, kernel_size, strides, padding)(disc_conv)
    disc_conv = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(disc_conv)

    disc_conv = tf.keras.layers.Conv2D(filters * 8, kernel_size, strides, padding)(disc_conv)
    disc_conv = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(disc_conv)

    disc_conv = tf.keras.layers.Conv2D(filters * 16, kernel_size, strides, padding)(disc_conv)
    disc_conv = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(disc_conv)

    fc = tf.keras.layers.Flatten()(disc_conv)
    fc = tf.keras.layers.Dense(filters * 2)(fc)
    fc = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(fc)
    fc = tf.keras.layers.Dropout(dropout)(fc)

    disc_output = tf.keras.layers.Dense(1)(fc)

    discriminator = tf.keras.models.Model(inputs = [disc_input], outputs = [disc_conv, disc_output])

    # Encoder
    # =========================================================================================
    enc_filters = filters // 1
    encoder_input = tf.keras.layers.Input(shape=(window_size, 128, channels))
    enc_conv = tf.keras.layers.Conv2D(enc_filters, kernel_size, strides, padding)(encoder_input)
    enc_conv = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(enc_conv)

    enc_conv = tf.keras.layers.Conv2D(enc_filters * 2, kernel_size, strides, padding)(enc_conv)
    enc_conv = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(enc_conv)

    enc_conv = tf.keras.layers.Conv2D(enc_filters * 4, kernel_size, strides, padding)(enc_conv)
    enc_conv = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(enc_conv)

    enc_conv = tf.keras.layers.Conv2D(enc_filters * 8, kernel_size, strides, padding)(enc_conv)
    enc_conv = tf.keras.layers.LeakyReLU(leaky_relu_alpha)(enc_conv)

    enc_conv = tf.keras.layers.Conv2D(enc_filters * 16, kernel_size, strides, padding)(enc_conv)
    enc_conv =  tf.keras.layers.LeakyReLU(leaky_relu_alpha)(enc_conv)

    enc_conv = tf.keras.layers.AveragePooling2D()(enc_conv) # this reduces the num params by 2 to 3x
    enc_conv = tf.keras.layers.Flatten()(enc_conv)
    enc_conv = tf.keras.layers.Dropout(dropout)(enc_conv)

    # Latent Space
    z_mean = tf.keras.layers.Dense(latent_dim, activation = 'tanh')(enc_conv)
    z_log_sigma = tf.keras.layers.Dense(latent_dim, activation = 'tanh')(enc_conv)
    z_encoder_output = tf.keras.layers.Lambda(vae_sampling, output_shape = (latent_dim))([z_mean, z_log_sigma])
    encoder = tf.keras.models.Model(inputs = encoder_input, outputs = [z_mean, z_log_sigma, z_encoder_output])

    # Decoder
    # =========================================================================================
    dec_input = tf.keras.layers.Input(shape=(latent_dim))

    dec_conv = tf.keras.layers.Dense(15 * 8 * 128)(dec_input)
    dec_conv = tf.keras.layers.Reshape((15, 8, 128))(dec_conv)

    dec_conv = tf.keras.layers.UpSampling2D()(dec_conv)
    dec_conv = tf.keras.layers.Conv2D(filters * 4, kernel_size, 1, padding)(dec_conv)
    dec_conv = tf.keras.layers.LayerNormalization()(dec_conv)
    dec_conv = tf.keras.layers.ReLU()(dec_conv)

    dec_conv = tf.keras.layers.UpSampling2D()(dec_conv)
    dec_conv = tf.keras.layers.Conv2D(filters * 2, kernel_size, 1, padding)(dec_conv)
    dec_conv = tf.keras.layers.LayerNormalization()(dec_conv)
    dec_conv = tf.keras.layers.ReLU()(dec_conv)

    dec_conv = tf.keras.layers.UpSampling2D()(dec_conv)
    dec_conv = tf.keras.layers.Conv2D(filters * 1, kernel_size, 1, padding)(dec_conv)
    dec_conv = tf.keras.layers.LayerNormalization()(dec_conv)
    dec_conv = tf.keras.layers.ReLU()(dec_conv)

    dec_conv = tf.keras.layers.UpSampling2D()(dec_conv)
    dec_conv = tf.keras.layers.Conv2D(channels, kernel_size, 1, padding)(dec_conv)
    dec_conv = tf.keras.layers.Activation('tanh')(dec_conv)

    decoder = tf.keras.models.Model(inputs = [dec_input], outputs = [dec_conv])


    return discriminator, encoder, decoder