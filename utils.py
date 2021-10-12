import os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_scaled_wav(path, start, end, scaling_factor, downsample_8K=False):
    samples, sr_orig = sf.read(path, start=start, stop=end)

    if len(samples.shape) > 1:
        samples = samples[:, 0]

    if downsample_8K:
        samples = resample_poly(samples, 8000, sr_orig)
    samples *= scaling_factor
    return samples


def wavwrite_quantize(samples):
    return np.int16(np.round((2 ** 15) * samples))


def quantize(samples):
    int_samples = wavwrite_quantize(samples)
    return np.float64(int_samples) / (2 ** 15)


def wavwrite(file, samples, sr):
    """This is how the old Matlab function wavwrite() quantized to 16 bit.
    We match it here to maintain parity with the original dataset"""
    int_samples = wavwrite_quantize(samples)
    sf.write(file, int_samples, sr, subtype='PCM_16')


def append_or_truncate(s1_samples, s2_samples, noise_samples, min_or_max='max', start_samp_16k=0, downsample=False):
    if downsample:
        speech_start_sample = start_samp_16k // 2
    else:
        speech_start_sample = start_samp_16k

    speech_end_sample = speech_start_sample + len(s1_samples)

    if min_or_max == 'min':
        noise_samples = noise_samples[speech_start_sample:speech_end_sample]
    else:
        s1_append = np.zeros_like(noise_samples)
        s2_append = np.zeros_like(noise_samples)
        s1_append[speech_start_sample:speech_end_sample] = s1_samples
        s2_append[speech_start_sample:speech_end_sample] = s2_samples
        s1_samples = s1_append
        s2_samples = s2_append

    return s1_samples, s2_samples, noise_samples


def fix_length(s1, s2, s3=[0], min_or_max='max'):
    if len(s3) > 1:
        # Fix length
        if min_or_max == 'min':
            utt_len = min(len(s1), len(s2), len(s3))
            s1 = s1[:utt_len]
            s2 = s2[:utt_len]
            s3 = s3[:utt_len]
        else:  # max
            utt_len = min(len(s1), len(s2), len(s3))
            s1 = np.append(s1, np.zeros(utt_len - len(s1)))
            s2 = np.append(s2, np.zeros(utt_len - len(s2)))
            s3 = np.append(s3, np.zeros(utt_len - len(s3)))
        return s1, s2, s3
    else:
        # Fix length
        if min_or_max == 'min':
            utt_len = np.minimum(len(s1), len(s2))
            s1 = s1[:utt_len]
            s2 = s2[:utt_len]
        else:  # max
            utt_len = np.maximum(len(s1), len(s2))
            s1 = np.append(s1, np.zeros(utt_len - len(s1)))
            s2 = np.append(s2, np.zeros(utt_len - len(s2)))
        return s1, s2
    


def create_wham_mixes(s1_samples, s2_samples, noise_samples):
    mix_clean = s1_samples + s2_samples
    mix_single = noise_samples + s1_samples
    mix_both = noise_samples + s1_samples + s2_samples
    return mix_clean, mix_single, mix_both


def create_overlap_mixes(s1_samples, s2_samples, s3_samples=[0]):
    utt_overlaps = []
    utt_len = len(s1_samples)
    for overlap_ratio in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        append_len = int(utt_len * (0.5 - overlap_ratio / 2) / (0.5 + overlap_ratio / 2))
        if len(s3_samples) == 1:
            zero_append = np.zeros(append_len)
            s1_samples = np.append(s1_samples, zero_append)
            s2_samples = np.append(zero_append, s2_samples)
            mix_samples = s1_samples + s2_samples
            utt_overlaps.append(mix_samples)
        else:
            zero_append = np.zeros(append_len)
            half_append = np.zeros(append_len // 2)
            s1_samples = np.append(s1_samples, zero_append)
            s2_samples = np.append(s2_samples, half_append)
            s2_samples = np.append(half_append, s2_samples)
            s3_samples = np.append(zero_append, s3_samples)
            if len(s1_samples) != len(s2_samples): s2_samples = np.append(s2_samples, [0])
            mix_samples = s1_samples + s2_samples + s3_samples
            utt_overlaps.append(mix_samples)
    return utt_overlaps


def find_cospro_path(cospro_root, s_path):
    # find details of wav
    # ex:  s_path: 03-M002_phrase_i_440_000000-000744.wav
    #     set_dir: COSPRO_03
    #         spk: M002
    #     utt_dir: phrase_i
    #     utt_num: 440
    #       start: 000000
    #         end: 000744
    details = s_path.split('.')[0].split('_')
    set_dir = f'COSPRO_{details[0][:2]}'
    spk = details[0].split('-')[-1]
    utt_dir = details[1]
    if set_dir == 'COSPRO_03':
        sub_utt_dir = details[2]
    elif set_dir == 'COSPRO_05':
        utt_dir = utt_dir + '_' + details[2]
    elif set_dir == 'COSPRO_08' and utt_dir == 'phrase':
        sub_utt_dir = details[2]
    elif set_dir == 'COSPRO_09' and len(details) == 5:
        utt_dir = utt_dir + '_' + details[2]
    utt_num = details[-2]
    start = details[-1].split('-')[0]
    start = float(start) / 100
    end = details[-1].split('-')[1]
    end = float(end) / 100
    
    # find wav dir
    if set_dir == 'COSPRO_02':
        if spk[0] == 'F':
            gender = 'Female'
        else:
            gender = 'Male'
        wav_dir = os.path.join(cospro_root, set_dir, gender, spk, utt_dir, 'wav')
    elif set_dir == 'COSPRO_03':
        wav_dir = os.path.join(cospro_root, set_dir, spk, utt_dir, sub_utt_dir, 'wav')
    elif set_dir == 'COSPRO_08' and utt_dir == 'phrase':
        wav_dir = os.path.join(cospro_root, set_dir, spk, utt_dir, sub_utt_dir, 'wav')
    else:
        wav_dir = os.path.join(cospro_root, set_dir, spk, utt_dir, 'wav')
    
    # find wav
    wav_name = None
    for wav in os.listdir(wav_dir):
        if utt_num in wav[14:]:
            if not wav.endswith('_f.wav') and not wav.endswith('_s.wav'):
                wav_name = wav
                break
    assert wav_name != None, f'We didn\'t find {utt_num} in {wav_dir}'
    wav_path = os.path.join(wav_dir, wav_name)

    return wav_path, start, end