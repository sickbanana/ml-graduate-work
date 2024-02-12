import collections
import pathlib
import pretty_midi
import glob
import matplotlib

import create_models

matplotlib.use('TkAgg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import tensorflow as tf
import pandas as pd

from matplotlib import pyplot as plt


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def create_sequences(
        dataset: tf.data.Dataset,
        seq_length: int,
        vocab_size=128,
) -> tf.data.Dataset:
    seq_length = seq_length + 1

    windows = dataset.window(seq_length, shift=1, stride=1,
                             drop_remainder=True)

    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

if __name__ == '__main__':
    data_dir = pathlib.Path('data/maestro-v2.0.0')
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )

    filenames = glob.glob(str(data_dir / '**/*.mid*'))
    print('Number of files:', len(filenames))

    num_files = 5
    all_notes = []
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)
        all_notes.append(notes)
    all_notes = pd.concat(all_notes)

    n_notes = len(all_notes)

    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
    X_train = tf.data.Dataset.from_tensor_slices(train_notes)
    print(X_train.element_spec)

    seq_length = create_models.seq_length
    vocab_size = 128
    X_train_seq = create_sequences(X_train, seq_length, vocab_size)
    batch_size = 64
    buffer_size = n_notes - seq_length  # the number of items in the dataset
    X_train_seq = (X_train_seq
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    print(X_train_seq.element_spec)

    model = create_models.create_model_lstm_v1()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints_test/cp.ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]

    epochs = 50
    history = model.fit(
        X_train_seq,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save('saved_models/lstm_v1')

    plt.plot(history.epoch, history.history['loss'], label='total loss')
    plt.show()

