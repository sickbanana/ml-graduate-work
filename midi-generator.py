import pathlib
import glob
import tensorflow as tf

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