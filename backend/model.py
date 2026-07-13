"""
CNN Segmentation Model
========================
Arsitektur U-Net (encoder-decoder) yang direplikasi persis sesuai
Fig. 3 dan Fig. 6 pada paper:

Encoder:
  Conv2D(32, 3x3, padding=same) -> 256x256x32
  MaxPooling2D(2x2)             -> 128x128x32
  Conv2D(64, 3x3, padding=same) -> 128x128x64
  MaxPooling2D(2x2)             -> 64x64x64
  Conv2D(128,3x3, padding=same) -> 64x64x128
  MaxPooling2D(2x2)             -> 32x32x128
  Dense(bottleneck)             -> 32x32x128

Decoder:
  UpSampling2D(2x2)             -> 64x64x128
  Conv2D(128, 3x3)              -> 64x64x128
  UpSampling2D(2x2)             -> 128x128x128
  Conv2D(64, 3x3)               -> 128x128x64
  UpSampling2D(2x2)             -> 256x256x64
  Conv2D(1, 1x1, sigmoid)       -> 256x256x1  (output mask)

Total parameter sesuai paper: 331,137
"""

from tensorflow.keras import layers, models


def build_unet(input_shape=(256, 256, 1)) -> "models.Model":
    inputs = layers.Input(shape=input_shape)

    # ---- Encoder ----
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)   # 256x256x32
    x = layers.MaxPooling2D(2)(x)                                        # 128x128x32

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)        # 128x128x64
    x = layers.MaxPooling2D(2)(x)                                        # 64x64x64

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)       # 64x64x128
    x = layers.MaxPooling2D(2)(x)                                        # 32x32x128

    # ---- Bottleneck (Dense, sesuai Fig.6) ----
    x = layers.Dense(128, activation="relu")(x)                          # 32x32x128

    # ---- Decoder ----
    x = layers.UpSampling2D(2)(x)                                        # 64x64x128
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)       # 64x64x128

    x = layers.UpSampling2D(2)(x)                                        # 128x128x128
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)        # 128x128x64

    x = layers.UpSampling2D(2)(x)                                        # 256x256x64
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)  # 256x256x1

    model = models.Model(inputs, outputs, name="CNN_MIS_UNet_CLAHE_HE")
    return model


def compile_model(model, learning_rate=1e-5):
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "mse"],
    )
    return model


if __name__ == "__main__":
    m = build_unet()
    m.summary()
    print("Total params:", m.count_params())
