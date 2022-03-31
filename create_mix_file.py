import os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from constants import SAMPLERATE, MAX_SAMPLE_AMP
import argparse
from utils import read_scaled_wav, quantize, fix_length, create_overlap_mixes, find_cospro_path, find_tat_path


FILELIST_STUB = os.path.join('data', '{}', 'mix_{}_spk_{}.txt')

MIX_100_DIR = 'mix_100'
MIX_80_DIR = 'mix_80'
MIX_60_DIR = 'mix_60'
MIX_40_DIR = 'mix_40'
MIX_20_DIR = 'mix_20'
MIX_0_DIR = 'mix_0'
S1_DIR = 's1'
S2_DIR = 's2'
S3_DIR = 's3'


def create_cospro_mix(cospro_root, tat_root, wsj0_root, output_root, spk_num, data, full_overlap):
    assert spk_num == 2 or spk_num == 3, f"Spk num should be 2 or 3, but we got {spk_num}"
    if '3' in data:
        spk_num = 3

    for splt in ['tr', 'cv', 'tt']:
        mix_data_path = FILELIST_STUB.format(data, spk_num, splt)
        mix_data_df = []
        with open(mix_data_path, 'r') as f:
            for line in f.readlines():
                mix_detail = line.split()
                s1_path = mix_detail[0]
                s2_path = mix_detail[2]
                mix_ratio = float(mix_detail[1])
                if spk_num == 3:
                    s3_path = mix_detail[4]

                if spk_num == 2:
                    mix_data = [s1_path, s2_path, mix_ratio]
                else:
                    mix_data = [s1_path, s2_path, s3_path, mix_ratio]
                mix_data_df.append(mix_data)

        for sr_dir in ['8k', '16k']:
            wav_dir = 'wav' + sr_dir
            if sr_dir == '8k':
                sr = 8000
                downsample = True
                loudness_meter = pyln.Meter(8000)
            else:
                sr = SAMPLERATE
                downsample = False
                loudness_meter = pyln.Meter(SAMPLERATE)

            for datalen_dir in ['min', 'max']:
                
                print('{} {} dataset, {} split'.format(sr_dir, datalen_dir, splt))
                
                output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)
                os.makedirs(os.path.join(output_path, MIX_100_DIR), exist_ok=True)
                if not full_overlap:
                    os.makedirs(os.path.join(output_path, MIX_80_DIR), exist_ok=True)
                    os.makedirs(os.path.join(output_path, MIX_60_DIR), exist_ok=True)
                    os.makedirs(os.path.join(output_path, MIX_40_DIR), exist_ok=True)
                    os.makedirs(os.path.join(output_path, MIX_20_DIR), exist_ok=True)
                    os.makedirs(os.path.join(output_path, MIX_0_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, S1_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, S2_DIR), exist_ok=True)
                if spk_num == 3:
                    os.makedirs(os.path.join(output_path, S3_DIR), exist_ok=True)

                for i_utt, mix_data in enumerate(mix_data_df):
                    if spk_num == 2:
                        s1_path, s2_path, mix_ratio = mix_data
                        s1_path_w = s1_path[:-4].split('/')[-1]
                        s2_path_w = s2_path[:-4].split('/')[-1]
                        output_name = s1_path_w + '_' + str(mix_ratio) + '_' + s2_path_w + '_' + str(-1*mix_ratio) + '.wav'
                    else:
                        s1_path, s2_path, s3_path, mix_ratio = mix_data
                        s1_path_w = s1_path[:-4].split('/')[-1]
                        s2_path_w = s2_path[:-4].split('/')[-1]
                        s3_path_w = s3_path[:-4].split('/')[-1]
                        output_name = s1_path_w + '_' + str(mix_ratio) + '_' + s2_path_w + '_' + str(-1*mix_ratio) + '_' + s3_path_w + '_0.wav'

                    if s1_path.startswith('si'):
                        s1_path = os.path.join(wsj0_root, s1_path)
                        s1_start = 0
                        s1_end = None
                    elif 'train' in s1_path or 'eval' in s1_path:
                        s1_path, s1_start, s1_end = find_tat_path(tat_root, s1_path)
                        s1_start *= SAMPLERATE
                        s1_end *= SAMPLERATE
                    else:
                        s1_path, s1_start, s1_end = find_cospro_path(cospro_root, s1_path)
                        s1_start *= SAMPLERATE
                        s1_end *= SAMPLERATE

                    if s2_path.startswith('si'):
                        s2_path = os.path.join(wsj0_root, s2_path)
                        s2_start = 0
                        s2_end = None
                    elif 'train' in s2_path or 'eval' in s2_path:
                        s2_path, s2_start, s2_end = find_tat_path(tat_root, s2_path)
                        s2_start *= SAMPLERATE
                        s2_end *= SAMPLERATE
                    else:
                        s2_path, s2_start, s2_end = find_cospro_path(cospro_root, s2_path)
                        s2_start *= SAMPLERATE
                        s2_end *= SAMPLERATE

                    s1 = read_scaled_wav(s1_path, int(s1_start), int(s1_end), scaling_factor=1.0, downsample_8K=downsample)
                    s1_speech_level = loudness_meter.integrated_loudness(s1)
                    gain_db = mix_ratio - s1_speech_level
                    s1_g = 10 ** ( gain_db / 20.)
                    s1 = quantize(s1) * s1_g

                    s2 = read_scaled_wav(s2_path, int(s2_start), int(s2_end), scaling_factor=1.0, downsample_8K=downsample)
                    s2_speech_level = loudness_meter.integrated_loudness(s2)
                    gain_db = - mix_ratio - s2_speech_level
                    s2_g = 10 ** ( gain_db / 20.)
                    s2 = quantize(s2) * s2_g


                    if spk_num == 2:
                        s1_samples, s2_samples = fix_length(s1, s2, min_or_max=datalen_dir)
                        mix_samples_list = create_overlap_mixes(s1_samples, s2_samples, full_overlap=full_overlap)
                        samps = mix_samples_list + [s1_samples, s2_samples]

                    else:
                        if s3_path.startswith('si'):
                            s3_path = os.path.join(wsj0_root, s3_path)
                            s3_start = 0
                            s3_end = None
                        elif 'train' in s3_path or 'eval' in s3_path:
                            s3_path, s3_start, s3_end = find_tat_path(tat_root, s3_path)
                            s3_start *= SAMPLERATE
                            s3_end *= SAMPLERATE
                        else:
                            s3_path, s3_start, s3_end = find_cospro_path(cospro_root, s3_path)
                            s3_start *= SAMPLERATE
                            s3_end *= SAMPLERATE
                        s3 = read_scaled_wav(s3_path, int(s3_start), int(s3_end), scaling_factor=1.0, downsample_8K=downsample)
                        s3_speech_level = loudness_meter.integrated_loudness(s3)
                        gain_db = 0 - s3_speech_level
                        s3_g = 10 ** ( gain_db / 20.)
                        s3 = quantize(s3) * s3_g

                        s1_samples, s2_samples, s3_samples = fix_length(s1, s2, s3, min_or_max=datalen_dir)
                        mix_samples_list = create_overlap_mixes(s1_samples, s2_samples, s3_samples, full_overlap=full_overlap)
                        samps = mix_samples_list + [s1_samples, s2_samples, s3_samples]

                    # check for clipping and fix gains
                    max_amp = 0
                    for samp in samps:
                        max_value = np.max(np.abs(samp))
                        if max_value > max_amp:
                            max_amp = max_value
                    if max_amp > MAX_SAMPLE_AMP:
                        lin_gain = MAX_SAMPLE_AMP / max_amp
                    else:
                        lin_gain = 1.0
                    samps_after_g = [samp * lin_gain for samp in samps]

                    # write audio
                    if full_overlap:
                        dirs = [MIX_100_DIR, S1_DIR, S2_DIR]
                    else:
                        dirs = [MIX_100_DIR, MIX_80_DIR, MIX_60_DIR, MIX_40_DIR, MIX_20_DIR, MIX_0_DIR, S1_DIR, S2_DIR]
                    if spk_num == 3: dirs.append(S3_DIR)
                    for dir, samp in zip(dirs, samps_after_g):
                        sf.write(os.path.join(output_path, dir, output_name), samp,
                                sr, subtype='FLOAT')

                    if (i_utt + 1) % 500 == 0:
                        print('Completed {} of {} utterances'.format(i_utt + 1, len(mix_data_df)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Output directory for writing wsj0-2mix 8 kHz and 16 kHz datasets.')
    parser.add_argument('--cospro-root', '-c', type=str, default=None,
                        help='Path to the folder containing COSPRO wavs')
    parser.add_argument('--tat-root', '-t', type=str, default=None,
                        help='Path to the folder containing TAT wavs')
    parser.add_argument('--wsj0-root', '-w', type=str, default=None,
                        help='Path to the folder containing WSJ0 dirs')
    parser.add_argument('--spk-num', '-n', type=int, default=2,
                        help='mix 2 or 3 spk')
    parser.add_argument('--data', '-d', type=str, default='data_COSPRO-2mix',
                        help='the data you want to mix')
    parser.add_argument('--full_overlap', '-f', type=bool, default=True,
                        help='whether make different overlap ratios ,True for only fully overlapped')
    args = parser.parse_args()
    create_cospro_mix(args.cospro_root, args.tat_root, args.wsj0_root, args.output_dir, args.spk_num, args.data, args.full_overlap)
