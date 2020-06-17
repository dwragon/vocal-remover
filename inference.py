import argparse
import os

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from lib import dataset
from lib import nets
from lib import spec_utils


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default='models/baseline.pth')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-l', type=int, default=1024)
    p.add_argument('--window_size', '-w', type=int, default=512)
    p.add_argument('--output_image', '-I', action='store_true')
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedASPPNet(8, args.n_fft)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
    print('done')

    print('stft of wave source...', end=' ')
    X = spec_utils.get_spectrogram(X, args.hop_length, args.n_fft)
    X_mag, X_phase = np.abs(X), np.angle(X)
    X_in = np.concatenate([X_mag, X_phase])
    print('done')

    offset = model.offset
    l, r, roi_size = dataset.make_padding(X.shape[2], args.window_size, offset)
    X_pad = np.pad(X_in, ((0, 0), (0, 0), (l, r)), mode='constant')[None]

    model.eval()
    with torch.no_grad():
        mag_preds = []
        phase_preds = []
        for i in tqdm(range(int(np.ceil(X.shape[2] / roi_size)))):
            start = i * roi_size
            X_window = X_pad[:, :, :, start:start + args.window_size]
            X_window = torch.from_numpy(X_window).to(device)

            X_window_mag = X_window[:, :2]
            X_window_phase = X_window[:, 2:4]
            y_mag, y_phase_left, y_phase_right = model.predict(X_window_mag, X_window_phase)

            y_mag = y_mag.detach().cpu().numpy()
            mag_preds.append(y_mag[0])

            y_phase_left = y_phase_left.detach().cpu().numpy()
            y_phase_right = y_phase_right.detach().cpu().numpy()
            phase_preds.append(np.stack([
                np.argmax(y_phase_left[0], axis=0),
                np.argmax(y_phase_right[0], axis=0),
            ]))

        mag_pred = np.concatenate(mag_preds, axis=2)[:, :, :X.shape[2]]
        phase_pred_label = np.concatenate(phase_preds, axis=2)[:, :, :X.shape[2]]

        del mag_preds, phase_preds

    basename = os.path.splitext(os.path.basename(args.input))[0]

    label_bins = np.linspace(-np.pi, np.pi, num=16)
    phase_pred = np.zeros_like(phase_pred_label, dtype=np.float32)
    for i in range(16):
        phase_pred[phase_pred_label == i] = label_bins[i]

    print('inverse stft of instruments...', end=' ')
    spec = mag_pred * np.exp(1.j * phase_pred)
    wave = spec_utils.spectrogram_to_wave(spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}_Instruments.wav'.format(basename), wave.T, sr)

    print('inverse stft of vocals...', end=' ')
    spec = (X_mag - mag_pred) * np.exp(1.j * np.angle(X - spec))
    wave = spec_utils.spectrogram_to_wave(spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}_Vocals.wav'.format(basename), wave.T, sr)

    if args.output_image:
        with open('{}_Instruments.jpg'.format(basename), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(mag_pred)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_Vocals.jpg'.format(basename), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(X_mag - mag_pred)
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)


if __name__ == '__main__':
    main()
