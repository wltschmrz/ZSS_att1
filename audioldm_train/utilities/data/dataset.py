import sys

sys.path.append("src")
import os
import pandas as pd
import yaml
import audioldm_train.utilities.audio as Audio
from audioldm_train.utilities.tools import load_json
from audioldm_train.dataset_plugin import *
from librosa.filters import mel as librosa_mel_fn

import random
from torch.utils.data import Dataset
import torch.nn.functional
import torch
import numpy as np
import torchaudio
import json


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

class AudioDataset(Dataset):
    def __init__(
        self,
        config=None,
        split="train",
        waveform_only=False,
        add_ons=[],
        dataset_json=None,
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.config = config
        self.split = split
        self.pad_wav_start_sample = 0  # If none, random choose
        self.trim_wav = False
        self.waveform_only = waveform_only
        self.add_ons = [eval(x) for x in add_ons]
        print("Add-ons:", self.add_ons)

        self.build_setting_parameters()  # 초기 prms 추가 세팅

        # For an external dataset
        if dataset_json is not None:
            self.data = dataset_json["data"]
            self.id2label, self.index_dict, self.num2label = {}, {}, {}
        else:
            self.metadata_root = load_json(self.config["metadata_root"])
            self.dataset_name = self.config["data"][self.split]
            keys = list(self.config["data"].keys())
            assert split in self.config["data"].keys(), f"The dataset split {split} you specified is not present in the config. You can choose from {keys}"  # 이거 걍 split이 train, test 아닌값 받는거 거름용
            self.build_dataset()
            self.build_id_to_label()
            
        self.build_dsp()  # Digital Signal Processing 관련 prms 세팅
        self.label_num = len(self.index_dict)
        print("Dataset initialize finished")

    def __len__(self):  # data length 반환
        return len(self.data)

    def __getitem__(self, index):  # data를 3+3+3가지 key에 대한 dictionary 형태로 묶어 반환
        (
            fname,
            waveform,
            stft,
            log_mel_spec,
            label_vector,  # the one-hot representation of the audio class
            (datum, mix_datum),  # the metadata of the sampled audio file and the mixup audio file (if exist)
            random_start,
        ) = self.feature_extraction(index)
        
        text = self.get_sample_text_caption(datum, mix_datum, label_vector)
        if text is None:
            print("Warning: The model return None on key text", fname); text = ""     

        data = {
            "text": text,  # list
            "fname": self.text_to_filename(text) if (not fname) else fname,  # list
            "label_vector": "" if (label_vector is None) else label_vector.float(),  # tensor, [B, class_num]
            "waveform": "" if (waveform is None) else waveform.float(),  # tensor, [B, 1, samples_num]
            "stft": "" if (stft is None) else stft.float(),  # tensor, [B, t-steps, f-bins]
            "log_mel_spec": "" if (log_mel_spec is None) else log_mel_spec.float(),  # tensor, [B, t-steps, mel-bins]
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "random_start_sample_in_original_audio_file": random_start,}

        for add_on in self.add_ons:
            data.update(add_on(self.config, data, self.data[index]))

        return data

    def text_to_filename(self, text:str):  # text(str) -> fname(str) 반환
        return text.replace(" ", "_").replace("'", "_").replace('"', "_")

    def get_dataset_root_path(self, dataset):  # 이해안됨
        assert dataset in self.metadata_root.keys()
        return self.metadata_root[dataset]

    def get_dataset_metadata_path(self, dataset, key):  # 이해안됨
        # key: train, test, val, class_label_indices
        try:
            if dataset in self.metadata_root["metadata"]["path"].keys():
                return self.metadata_root["metadata"]["path"][dataset][key]
        except:
            raise ValueError(f'Dataset {dataset} does not metadata "{key}" specified')

    def feature_extraction(self, index):
        if index > len(self.data) - 1:
            print(f"The index of the dataloader is out of range: {index}/{len(self.data)}")
            index = random.randint(0, len(self.data) - 1)

        # Read wave file and extract feature
        while True:
            try:
                label_indices = np.zeros(self.label_num, dtype=np.float32)
                datum = self.data[index]
                (log_mel_spec, stft, waveform, random_start) = self.read_audio_file(datum["wav"])
                mix_datum = None
                if self.label_num > 0 and "labels" in datum.keys():
                    for label_str in datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] = 1.0

                # If the key "label" is not in the metadata, return all zero vector
                label_indices = torch.FloatTensor(label_indices)
                break
            except Exception as error:
                index = (index + 1) % len(self.data)
                print(f"Error encounter during audio feature extraction: {error}, path: {datum['wav']}")
                continue

        # The filename of the wav file
        fname = datum["wav"]
        # t_step = log_mel_spec.size(0)
        # waveform = torch.FloatTensor(waveform[..., : int(self.hopsize * t_step)])
        waveform = torch.FloatTensor(waveform)

        return (fname,
                waveform,
                stft,
                log_mel_spec,
                label_indices,
                (datum, mix_datum),
                random_start,)

    '''
    def augmentation(self, log_mel_spec): # 원래 주석이었던거 닫아놓았음
        assert torch.min(log_mel_spec) < 0
        log_mel_spec = log_mel_spec.exp()

        log_mel_spec = torch.transpose(log_mel_spec, 0, 1)
        # this is just to satisfy new torchaudio version.
        log_mel_spec = log_mel_spec.unsqueeze(0)
        if self.freqm != 0:
            log_mel_spec = self.frequency_masking(log_mel_spec, self.freqm)
        if self.timem != 0:
            log_mel_spec = self.time_masking(log_mel_spec, self.timem)  # self.timem=0

        log_mel_spec = (log_mel_spec + 1e-7).log()
        # squeeze back
        log_mel_spec = log_mel_spec.squeeze(0)
        log_mel_spec = torch.transpose(log_mel_spec, 0, 1)
        return log_mel_spec
    '''

    def build_setting_parameters(self):
        # Read from the json config                                                   # s-full 기준 (audioldm_original.yaml),
        self.melbins = self.config["preprocessing"]["mel"]["n_mel_channels"]          # melbins       = 64     # 중복
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]   # sampling_rate = 16000  # 중복
        self.hopsize = self.config["preprocessing"]["stft"]["hop_length"]             # hopsize       = 160    # 중복
        self.duration = self.config["preprocessing"]["audio"]["duration"]             # duration      = 10.24
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)   # target_length = 1024
        self.mixup = self.config["augmentation"]["mixup"]                             # mixup         = 0.0
        
        if "train" not in self.split:
            self.mixup = 0.0
        '''
        # augmentation 있었으면 썼을 setting
            self.freqm = 0
            self.timem = 0
        self.freqm = self.config["preprocessing"]["mel"]["freqm"]
        self.timem = self.config["preprocessing"]["mel"]["timem"]
        Calculate parameter derivations
        self.waveform_sample_length = int(self.target_length * self.hopsize)
        if (self.config["balance_sampling_weight"]):
            self.samples_weight = np.loadtxt(self.config["balance_sampling_weight"], delimiter=",")
        '''

    def _relative_path_to_absolute_path(self, metadata, dataset_name):  # 메타데이터에 있는 오디오 파일의 경로를 전체(절대) 경로로 업데이트
        root_path = self.get_dataset_root_path(dataset_name)

        for item in metadata["data"]:
            if "wav" not in item:
                raise KeyError(f"Missing 'wav' key in metadata item: {item}")
            if item["wav"].startswith("/"):
                raise ValueError(f"Path must be relative: {item['wav']}")
            
            item["wav"] = os.path.join(root_path, item["wav"])
        return metadata

    def build_dataset(self):
        print(f"Build dataset split {self.split} from {self.dataset_name}")
        # str이나 list 여부 체크 & iterable하게 [] 감싸기
        self.data = []
        datasets = [self.dataset_name] if isinstance(self.dataset_name, str) else self.dataset_name
        if not isinstance(datasets, list):
            raise ValueError("dataset_name must be either string or list")
        
        for dataset in datasets:
            data_json = load_json(self.get_dataset_metadata_path(dataset, key=self.split))
            data_json = self._relative_path_to_absolute_path(data_json, dataset)
            self.data.extend(data_json["data"])
        print(f"Data size: {len(self.data)}")

    def build_dsp(self):  # Digital Signal Processing을 build함 
        self.mel_basis = {}
        self.hann_window = {}
                                                                                      # s-full 기준 (audioldm_original.yaml),
        self.filter_length = self.config["preprocessing"]["stft"]["filter_length"]    # filter_length = 1024
        self.hop_length = self.config["preprocessing"]["stft"]["hop_length"]          # hop_length    = 160
        self.win_length = self.config["preprocessing"]["stft"]["win_length"]          # win_length    = 1024
        self.n_mel = self.config["preprocessing"]["mel"]["n_mel_channels"]            # n_mel         = 64
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]   # sampling_rate = 16000
        self.mel_fmin = self.config["preprocessing"]["mel"]["mel_fmin"]               # mel_fmin      = 0
        self.mel_fmax = self.config["preprocessing"]["mel"]["mel_fmax"]               # mel_fmax      = 8000

        self.STFT = Audio.stft.TacotronSTFT(
            self.filter_length,
            self.hop_length,
            self.win_length,
            self.n_mel,
            self.sampling_rate,
            self.mel_fmin,
            self.mel_fmax,)

    def build_id_to_label(self):  # CSV 파일을 읽어서 class label과 관련된 세 가지 매핑 dict를 생성
        self.id2label = {}
        self.index_dict = {}
        self.num2label = {}
        
        path = self.get_dataset_metadata_path(
            dataset=self.config["data"]["class_label_indices"],
            key="class_label_indices")
        
        if path:
            df = pd.read_csv(path)
            for index, mid, display_name in df[["index", "mid", "display_name"]].values:
                self.id2label[mid] = display_name
                self.index_dict[mid] = index
                self.num2label[index] = display_name

    def resample(self, waveform, sr):  # waveform의 sampling rate를 변환하는 함수
        waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        return waveform

    def normalize_wav(self, waveform):  # 오디오 파형을 정규화
        MAX_AMPLITUDE = 0.5
        EPSILON = 1e-8
        
        centered = waveform - np.mean(waveform)
        normalized = centered / (np.max(np.abs(centered)) + EPSILON)
        return normalized * MAX_AMPLITUDE  # Manually limit the maximum amplitude into 0.5

    def random_segment_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, f"Waveform is too short, {waveform_length}"

        # Too short
        if waveform_length <= target_length:
            return waveform, 0

        for _ in range(10):
            random_start = int(self.random_uniform(0, waveform_length - target_length))
            segment = waveform[:, random_start:random_start + target_length]
            if torch.max(torch.abs(segment)) > 1e-4:
                return segment, random_start
        # 10번 시도에도 적절한 세그먼트 못찾은 경우, 마지막 시도 반환
        return segment, random_start

    def pad_wav(self, waveform, target_length):  # wav를 목표 길이로 padding -> padded_wav
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, f"Waveform is too short, {waveform_length}"

        if waveform_length == target_length:
            return waveform

        # Pad
        padded_wav = np.zeros((1, target_length), dtype=np.float32)
        start_pos = 0 if self.pad_wav_start_sample else int(self.random_uniform(0, target_length - waveform_length))
        
        padded_wav[:, start_pos:start_pos + waveform_length] = waveform
        return padded_wav

    def trim_wav(self, waveform, threshold=0.0001, chunk_size=1000):  # wav의 시작&끝에 있는 무음 구간을 제거하는(trim) 함수
        """      
        Args:
            waveform: 입력 오디오 파형
            threshold: 무음으로 간주할 진폭 임계값
            chunk_size: 한 번에 처리할 샘플 수
        """
        if np.max(np.abs(waveform)) < threshold:
            return waveform
        
        def find_sound_boundary(samples, reverse=False):
            length = samples.shape[0]
            pos = length if reverse else 0
            limit = 0 if reverse else length
            step = -chunk_size if reverse else chunk_size
            
            while (pos - step if reverse else pos + chunk_size) > limit:
                chunk_start = pos - chunk_size if reverse else pos
                chunk_end = pos if reverse else pos + chunk_size
                if np.max(np.abs(samples[chunk_start:chunk_end])) < threshold:
                    pos += step
                else:
                    break
                    
            return pos + (chunk_size if reverse else 0)
        
        start = find_sound_boundary(waveform, reverse=False)
        end = find_sound_boundary(waveform, reverse=True)
        
        return waveform[start:end]

    def read_wav_file(self, filename):  # 오디오 파일을 읽고 전처리
        '''
        Args:
            filename: 오디오 파일 경로
        Returns:
            waveform: 처리된 오디오 파형
            random_start: 랜덤 세그먼트의 시작 위치
        '''
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        waveform, original_sr = torchaudio.load (filename)                            # 1. 파일 로드

        target_samples = int(original_sr * self.duration)                             # 2. 랜덤 세그먼트 추출
        waveform, random_start = self.random_segment_wav(waveform, target_samples)

        waveform = self.resample(waveform, original_sr)                               # 3. 리샘플링
        # random_start = int(random_start * (self.sampling_rate / sr))
                                                                                      # 4. 전처리 단계
        waveform = waveform.numpy()[0, ...]                                           #     numpy 변환 및 첫 번째 채널 선택
        waveform = self.normalize_wav(waveform)                                       #     정규화
        if self.trim_wav:
            waveform = self.trim_wav(waveform)                                        #     무음 구간 제거
                                                                                      # 5. 최종 형태로 변환
        waveform = waveform[None, ...]                                                #     채널 차원 추가
        target_length = int(self.sampling_rate * self.duration)
        waveform = self.pad_wav(waveform, target_length)                              #     패딩
        
        return waveform, random_start

    def read_audio_file(self, filename):  # 오디오 파일을 읽고 필요한 경우 특성을 추출
        """
        Args:
            filename: 오디오 파일 경로
        
        Returns:
            tuple: (log_mel_spec, stft, waveform, random_start)
        """
        # 1. 오디오 파일 로드 또는 빈 파형 생성
        if os.path.exists(filename):
            waveform, random_start = self.read_wav_file(filename)
        else:
            target_length = int(self.sampling_rate * self.duration)
            waveform = torch.zeros((1, target_length))
            random_start = 0
            print(f'Non-fatal Warning [dataset.py]: The wav path "{filename}" not found. Using empty waveform.')
        
        # 2. 특성 추출 (필요한 경우)
        log_mel_spec, stft = (None, None) if self.waveform_only else self.wav_feature_extraction(waveform)
    
        return log_mel_spec, stft, waveform, random_start

    def get_sample_text_caption(self, datum, mix_datum, label_indices):
        text = self.label_indices_to_text(datum, label_indices)
        if mix_datum is not None:
            text += " " + self.label_indices_to_text(mix_datum, label_indices)
        return text

    def mel_spectrogram_train(self, y):
        if torch.min(y) < -1.0:
            print("train min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("train max value is ", torch.max(y))

        if self.mel_fmax not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.n_mel,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,)
            
            self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)] = (torch.from_numpy(mel).float().to(y.device))
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((self.filter_length - self.hop_length) / 2),
             int((self.filter_length - self.hop_length) / 2),),
            mode="reflect",
        )

        y = y.squeeze(1)

        stft_spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(y.device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        stft_spec = torch.abs(stft_spec)

        mel = spectral_normalize_torch(torch.matmul(self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], stft_spec))

        return mel[0], stft_spec[0]

    # This one is significantly slower than "wav_feature_extraction_torchaudio" if num_worker > 1
    def wav_feature_extraction(self, waveform):  # (1, 163840)
        waveform = waveform[0, ...]  # (163840,)

        waveform = torch.FloatTensor(waveform)

        # log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)[0]
        log_mel_spec, stft = self.mel_spectrogram_train(waveform.unsqueeze(0))  # input: torch.Size([1, 163840])

        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)

        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
        return log_mel_spec, stft

    '''
    @profile
    def wav_feature_extraction_torchaudio(self, waveform):
        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        stft = self.stft_transform(waveform)
        mel_spec = self.melscale_transform(stft)
        log_mel_spec = torch.log(mel_spec + 1e-7)

        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)

        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
        return log_mel_spec, stft'''

    def pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : self.target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec

    def _read_datum_caption(self, datum):
        caption_keys = [x for x in datum.keys() if ("caption" in x)]
        random_index = torch.randint(0, len(caption_keys), (1,))[0].item()
        return datum[caption_keys[random_index]]

    def _is_contain_caption(self, datum):
        caption_keys = [x for x in datum.keys() if ("caption" in x)]
        return len(caption_keys) > 0

    def label_indices_to_text(self, datum, label_indices):
        if self._is_contain_caption(datum):
            return self._read_datum_caption(datum)
        elif "label" in datum.keys():
            name_indices = torch.where(label_indices > 0.1)[0]
            # description_header = "This audio contains the sound of "
            description_header = ""
            labels = ""
            for id, each in enumerate(name_indices):
                if id == len(name_indices) - 1:
                    labels += "%s." % self.num2label[int(each)]
                else:
                    labels += "%s, " % self.num2label[int(each)]
            return description_header + labels
        else:
            return ""  # TODO, if both label and caption are not provided, return empty string

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def frequency_masking(self, log_mel_spec, freqm):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq - mask_len))
        log_mel_spec[:, mask_start : mask_start + mask_len, :] *= 0.0
        return log_mel_spec

    def time_masking(self, log_mel_spec, timem):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps - mask_len))
        log_mel_spec[:, :, mask_start : mask_start + mask_len] *= 0.0
        return log_mel_spec


if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader

    seed_everything(0)

    def write_json(my_dict, fname):
        # print("Save json file at "+fname)
        json_str = json.dumps(my_dict)
        with open(fname, "w") as json_file:
            json_file.write(json_str)

    def load_json(fname):
        with open(fname, "r") as f:
            data = json.load(f)
            return data

    config = yaml.load(
        open(
            "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/config/vae_48k_256/ds_8_kl_1.0_ch_16.yaml",
            "r",
        ),
        Loader=yaml.FullLoader,
    )

    add_ons = config["data"]["dataloader_add_ons"]

    # load_json(data)
    dataset = AudioDataset(
        config=config, split="train", waveform_only=False, add_ons=add_ons
    )

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    for cnt, each in tqdm(enumerate(loader)):
        # print(each["waveform"].size(), each["log_mel_spec"].size())
        # print(each['freq_energy_percentile'])
        import ipdb

        ipdb.set_trace()
        # pass