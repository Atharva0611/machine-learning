import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(num_classes: int, input_shape=(224, 224, 3)) -> keras.Model:
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_datasets(data_dir: Path, image_size=(224, 224), batch_size=32):
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for plant disease classification.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset directory. One folder per class.")
    parser.add_argument("--model-out", type=Path, default=Path("models/plant_disease_mobilenetv2.keras"))
    parser.add_argument("--labels-out", type=Path, default=Path("models/class_names.json"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    train_ds, val_ds = load_datasets(
        args.data_dir,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    model = build_model(num_classes=len(train_ds.class_names), input_shape=(args.img_size, args.img_size, 3))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.labels_out.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.model_out, monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    model.save(args.model_out)

    args.labels_out.write_text(json.dumps(train_ds.class_names, indent=2), encoding="utf-8")

    print(f"Model saved to {args.model_out}")
    print(f"Class labels saved to {args.labels_out}")


if __name__ == "__main__":
    main()
