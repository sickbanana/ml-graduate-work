import collections
import pathlib
import glob
import pretty_midi

import matplotlib
matplotlib.use('TkAgg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import numpy as np
import tensorflow as tf
import pandas as pd


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

    num_files = 50
    all_notes = []
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)
        all_notes.append(notes)


    seq_length = 32
    step = 10
    vocab_size = 128
    embedding_dim = 256 * 2
    rnn_units = 1024 * 2

    x = tf.keras.Input(shape=(seq_length, 3))
    e = tf.keras.layers.Embedding(vocab_size, embedding_dim)(x)
    l = tf.keras.layers.LSTM(rnn_units, return_sequences=True)(e)
    d = tf.keras.layers.Dense(vocab_size, activation='softmax')(l)
    model = tf.keras.Model(inputs=x, outputs=d)
    model.summary()