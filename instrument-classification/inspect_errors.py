from argparse import ArgumentParser
import os
import pandas as pd
import shutil
import subprocess

from prepare_training_data import load_transformers


def inspect_errors(model_id):
    model_dir = 'data/working/single-notes-2000/models/' + model_id
    input_dir = 'data/prepared/single-notes-2000'
    output_dir = model_dir + '/evaluation/errors'

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(model_dir + '/output-data/predictions.csv')
    errors = df.query("split == 'valid' & ~accurate")[:]

    instr_family_le = load_transformers(model_dir)[0]

    errors['sample_id'] = errors.index.map(lambda i: '%06d' % i)
    errors['label_true'] = errors['y_true'].apply(lambda y_true: instr_family_le.inverse_transform(y_true))
    errors['label_pred'] = errors['y_pred'].apply(lambda y_pred: instr_family_le.inverse_transform(y_pred))
    errors['input_file'] = errors['sample_id'].apply(lambda sample_id: input_dir + '/sample_%s.flac' % sample_id)
    errors['output_file'] = errors.apply(lambda row: output_dir + '/sample_%s_%s_%s_%s.flac'
        % (row['split'], row['sample_id'], row['label_true'], row['label_pred']), axis=1)

    print(errors[['sample_id', 'label_true', 'label_pred']])

    # rename the files so that they are more legible
    for i, row in errors.iterrows():
        shutil.copy(row['input_file'], row['output_file'])

    # play the files (this is probably limited so some max command length)
    subprocess.call(['open'] + list(errors['output_file']))

def parse_args():
    parser = ArgumentParser('Inspect errors - listen to misclassified audio files.')
    parser.add_argument('model_dir', metavar='MODEL_DIR')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    inspect_errors(args.model_dir)
