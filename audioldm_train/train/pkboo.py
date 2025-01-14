import os
import sys
import logging

# 모듈 검색 경로 추가
sys.path.append("/media/wlts/Elements/dev/projects/ZSS_att1/")

# 토크나이저 병렬 처리 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything
from tqdm import tqdm

# 프로젝트 내부 모듈
from audioldm_train.utilities.data.dataset import AudioDataset, spectral_normalize_torch
from audioldm_train.utilities.model_util import instantiate_from_config
from audioldm_train.utilities.tools import get_restore_step, copy_test_subset_data
from peekaboo.mask_generation import LearnableImageFourier, LearnableImageRaster


# 로깅 레벨을 WARNING으로 설정
logging.basicConfig(level=logging.WARNING)

def make_learnable_image(height, width, num_channels, representation='fourier'):
    "이미지의 파라미터화 방식을 결정하여 학습 가능한 이미지를 생성."
    if representation == 'fourier':
        return LearnableImageFourier(height, width, num_channels)
    elif representation == 'raster':
        return LearnableImageRaster(height, width, num_channels)
    else:
        raise ValueError(f'Invalid method: {representation}')
    
def masking_torch_image(foreground, alpha):
    assert foreground.shape == alpha.shape, 'foreground shape != alpha shape'
    assert alpha.min().item() >= 0 and alpha.max().item() <= 1, f'alpha range error {alpha.min().item()}, {alpha.max().item()}'
    min_val = foreground.min()
    blended = (foreground - min_val) * alpha + min_val
    # print('max 변화량', foreground.max().item(), blended.max().item())
    return blended

class AudioPeekabooSeparator(nn.Module):    
    def __init__(self, 
                 data,
                 dataset,
                 device,
                 num_label=1,
                 representation='raster',):

        super().__init__()
        self.data = data  # data['log_mel_spec']: [B,1024,64] / data['stft']: [B,t-steps,f-bins]
        self.height = data['stft'].shape[1]  # 1024 -> 513
        self.width = data['stft'].shape[2]  # 64 X -> 1024 -> 512 -> 1024
        self.dataset = dataset
        self.device = device
        self.num_label = num_label  # 1
        self.representation = representation
        
        # self.foreground = data['log_mel_spec'].to(device)  # [1, t-steps, mel-bins]
        # self.alpha = make_learnable_image(self.height, self.width, num_channels=self.num_label, representation=self.representation)  # [num_label, H, W]
        self.foreground = data['stft'].to(device)  # tensor, [B, t-steps, f-bins]
        self.alpha = make_learnable_image(self.height, self.width, self.num_label, representation)

    @property
    def num_labels(self):
        return self.num_label
        
    def forward(self, alpha=None, return_alpha=False):        
        alpha = alpha if alpha is not None else self.alpha()
        masked_stft = masking_torch_image(self.foreground, alpha)

        self.data['stft'] = masked_stft.float()
        # print(self.dataset.mel_basis[str(self.dataset.mel_fmax) + "_" + str('cpu')].shape)
        # print(self.dataset.mel_basis[str(self.dataset.mel_fmax) + "_" + str('cpu')].to(self.device).shape)
        # print(masked_stft.shape)

        mel = spectral_normalize_torch(torch.matmul(self.dataset.mel_basis[str(self.dataset.mel_fmax) + "_" + str('cpu')].to(self.device), masked_stft))
        masked_log_mel_spec = self.dataset.pad_spec(mel[0].T).unsqueeze(0).float()
        original_shape = self.data['log_mel_spec'].shape
        assert self.data['log_mel_spec'].shape == masked_log_mel_spec.shape, f'{original_shape} != {masked_log_mel_spec.shape}'
        self.data['log_mel_spec'] = masked_log_mel_spec

        assert not torch.isnan(alpha).any() or not torch.isinf(alpha).any(), "alpha contains NaN or Inf values"  # NaN이나 Inf 값 체크

        return (self.data, alpha) if return_alpha else self.data
    
def get_mixed_batches(batch1, batch2, dataset, snr_db=0, device=None):
    wav1, wav2 = batch1["waveform"], batch2["waveform"]  # [B=1, 1, samples_num]
    assert wav1.shape == wav2.shape, "두 WAV 텐서의 shape이 같아야 합니다"
    power1, power2 = torch.mean(wav1 ** 2), torch.mean(wav2 ** 2)  # 각 신호의 파워 계산
    scaling_factor = torch.sqrt(power1 / power2 * 10 ** (-snr_db/10))  # SNR에 따른 스케일링 계수 계산
    mixed = wav1 + wav2 * scaling_factor  # 두 신호 믹스
    max_abs = mixed.abs().max()
    if max_abs > 1:
        mixed /= max_abs  # 클리핑 방지를 위한 정규화 (선택사항)
    mixed_wav = (mixed * 0.5).float()
    log_mel_spec, stft = dataset.mel_spectrogram_train(mixed_wav[0, ...])
    log_mel_spec = dataset.pad_spec(torch.FloatTensor(log_mel_spec.T)).unsqueeze(0).float()
    # stft = dataset.pad_spec(torch.FloatTensor(stft.T)).unsqueeze(0).float()  #####
    if device:
        mixed_wav, log_mel_spec, stft = mixed_wav.to(device), log_mel_spec.to(device), stft.unsqueeze(0).float().to(device)  #####
    def set_dict(batch):
        label_vector = batch["label_vector"].float()
        if device:
            label_vector = label_vector.to(device)
        return {
            "text": batch["text"],
            "fname": batch["fname"],
            "label_vector": label_vector,
            "waveform": mixed_wav,
            "stft": stft,
            "log_mel_spec": log_mel_spec,
            "duration": batch["duration"],
            "sampling_rate": batch["sampling_rate"],
            "random_start_sample_in_original_audio_file":
                batch["random_start_sample_in_original_audio_file"],
        }
    return set_dict(batch1), set_dict(batch2)

def setting_result_folder(fname: str):
    '''
    project/result/ 폴더 생성.
    result/ 안에 text 폴더 생성 (중복 시 숫자 붙임) (참조: composite_batch['text'][0]: str).
    result/[text]/ 안에 alpha_process, separation_process 폴더 생성.
    '''
    base_result_dir = "result"
    os.makedirs(base_result_dir, exist_ok=True)

    text_dir_name = fname.replace(" ", "_")[:30]
    counter = 0
    while True:
        text_dir = os.path.join(base_result_dir, f"{text_dir_name}_{counter:03d}")
        if not os.path.exists(text_dir):
            break
        counter += 1
    os.makedirs(text_dir)
    alphas_dir = os.path.join(text_dir, "alphas")
    separations_dir = os.path.join(text_dir, "separations")
    os.makedirs(alphas_dir)
    os.makedirs(separations_dir)

    return (text_dir, alphas_dir, separations_dir)

def save_melspec_as_img(mel_tensor, save_path):
    mel = mel_tensor.detach().cpu().numpy()
    if mel.shape[0] > mel.shape[1]:
        mel = mel.T  # (64, 1024)로 전치
    height, width = mel.shape
    aspect_ratio = width / height  # 1024/64 = 16
    fig_width = 20  # 기준 가로 길이
    fig_height = fig_width / aspect_ratio  # 20/16 = 1.25
    if mel.min() < 0:
        min_, max_ = -11.5129, 3.4657
    else:
        min_, max_ = 0, 1
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='magma',
               vmin=min_, vmax=max_)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_initial_state(log_mel_spec, fname, latent_diffusion):
    """초기 상태 저장"""
    log_mel_spec = log_mel_spec.to(latent_diffusion.device)

    # 초기 멜스펙트로그램 저장
    save_melspec_as_img(
        log_mel_spec[0, ...],
        os.path.join(fname, "mixed_mel.png"))
    
    # 초기 wav 파일 저장
    with torch.no_grad():
        latent_diffusion.mel_spectrogram_to_waveform(
            log_mel_spec,
            savepath=fname,
            name="mixed",
            save=True)

def save_intermediate_results(iter_idx, alpha, composite_batch, f_alpha, f_sep_mel):
    """중간 결과 저장"""
    with torch.no_grad():
        alpha_cpu = alpha[0, ...].detach().cpu()
        mel_cpu = composite_batch['log_mel_spec'][0, ...].detach().cpu()
        save_melspec_as_img(alpha_cpu, os.path.join(f_alpha, f"{iter_idx:04d}.png"))
        save_melspec_as_img(mel_cpu, os.path.join(f_sep_mel, f"{iter_idx:04d}.png"))

def save_final_results(composite_batch, fname, losses1, losses2, p, n, p2, n2, latent_diffusion):
    """최종 결과 저장"""
    try:
        mel = composite_batch['log_mel_spec'].unsqueeze(0).to(latent_diffusion.device)
        mel_cpu = composite_batch['log_mel_spec'][0, ...].detach().cpu()

        save_melspec_as_img(mel_cpu, os.path.join(fname, "seped_mel.png"))

        # 최종 wav 파일 저장
        with torch.no_grad():
            latent_diffusion.mel_spectrogram_to_waveform(
                    mel,
                    savepath=fname,
                    name="seped",
                    save=True)

        # Loss plot 저장
        save_loss_plot(losses1, losses2, os.path.join(fname, "loss.png"))
        save_loss_plot_together(p, n, os.path.join(fname, "noises_mean.png"))
        save_loss_plot_together(p2, n2, os.path.join(fname, "noises_abs_mean.png"))
    except Exception as e:
        print(f"결과 저장 중 에러 발생: {e}")

def save_loss_plot_together(p, n, save_path):
    plt.figure(figsize=(10, 6))
    try:
        pred = np.array(p)
        noise = np.array(n)

        plt.plot(pred, label="Predicted Noise")
        plt.plot(noise, label="Gaussian Noise", color='red')
        plt.title("Noises")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
    except Exception as e:
        print(f"Loss plot 저장 중 에러 발생: {e}")
    finally:
        plt.close()

def save_loss_plot(losses1, losses2, save_path):
    plt.figure(figsize=(10, 6))
    try:
        losses1_np = np.array(losses1)
        losses2_np = np.array(losses2)

        plt.subplot(2, 1, 1)
        plt.plot(losses1_np, label="Noise MSE Loss")
        plt.title("Noise MSE Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(losses2_np, label="Alpha Regularization Loss", color='red')
        plt.title("Alpha Regularization Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
    except Exception as e:
        print(f"Loss plot 저장 중 에러 발생: {e}")
    finally:
        plt.close()

def save_audio_metadata(batch1_data, batch2_data, hyperparams, save_path):
    def convert_tensor_to_native(value):
        if torch.is_tensor(value):
            return value.item() if value.numel() == 1 else value.tolist()
        elif isinstance(value, (list, tuple)):
            return [convert_tensor_to_native(item) for item in value]
        return value
    
    def process_batch_data(batch):
        return {
            'text': convert_tensor_to_native(batch['text']),
            'fname': convert_tensor_to_native(batch['fname']),
            'duration': convert_tensor_to_native(batch['duration']),
            'sampling_rate': convert_tensor_to_native(batch['sampling_rate']),
            'random_start': convert_tensor_to_native(batch['random_start_sample_in_original_audio_file'])
        }

    metadata = {
        'hyperparameters': {
            'GRAVITY': hyperparams['GRAVITY'],
            'NUM_ITER': hyperparams['NUM_ITER'],
            'LEARNING_RATE': hyperparams['LEARNING_RATE'],
            'BATCH_SIZE': hyperparams['BATCH_SIZE'],
            'GUIDANCE_SCALE': hyperparams['GUIDANCE_SCALE']
        },
        'audio1': process_batch_data(batch1_data),
        'audio2': process_batch_data(batch2_data)
    }
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(metadata, f, default_flow_style=False, allow_unicode=True)
        print(f"메타데이터가 성공적으로 저장되었습니다: {save_path}")
    except Exception as e:
        print(f"메타데이터 저장 중 에러 발생: {e}")


def run_pkboo(configs, config_yaml_path, exp_group_name, exp_name, perform_validation, **kwargs):

    # 기본 시드 설정
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        seed_everything(0)

    # precision 설정 
    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(configs["precision"])  # highest, high, medium

    # 기본 설정값
    batch_size = configs["model"]["params"]["batchsize"] -1
    dataloader_add_ons = configs["data"].get("dataloader_add_ons", [])

    # 데이터 로더 설정
    dataset = AudioDataset(configs, split="train", add_ons=dataloader_add_ons)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=True)
    
    # 데이터셋 길이와 배치 사이즈 출력
    print(f"The length of the dataset is {len(dataset)}, the length of the dataloader is {len(loader)}, the batchsize is {batch_size}")
    
    # 체크포인트 로드 설정
    resume_from_checkpoint = None
    if "reload_from_ckpt" in configs:
        resume_from_checkpoint = configs["reload_from_ckpt"]
        ckpt = torch.load(resume_from_checkpoint)["state_dict"]
    
    # 모델 초기화
    latent_diffusion = instantiate_from_config(configs["model"])

    # 외부 체크포인트 로드
    if resume_from_checkpoint:
        state_dict = latent_diffusion.state_dict()
        for key in list(ckpt.keys()):
            if key not in state_dict.keys() or state_dict[key].size() != ckpt[key].size():
                del ckpt[key]
        latent_diffusion.load_state_dict(ckpt, strict=False)

    ##### 여기까지는 latent_diffusion 모델을 불러오는 과정 #####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # latent diffusion 파라미터 freezing
    for param in latent_diffusion.parameters():
        param.requires_grad = False
    # latent_diffusion.eval().to(device)          ########## eval일때 Warning 발생. text encoder 뭘 쓰고 있는지 확인해야함.
    latent_diffusion.to(device)

    # mixed and sep data 제작
    batch1 = next(iter(loader))
    for _ in range(2):
        __ = next(iter(loader))
    batch2 = next(iter(loader))

    sep_data1, sep_data2 = get_mixed_batches(batch1, batch2, dataset, snr_db=0, device=device)    

    global same_count
    same_count = 0
    # hyper parameter setting
    GRAVITY=kwargs['GRAVITY']
    NUM_ITER=kwargs['NUM_ITER']
    LEARNING_RATE=kwargs['LEARNING_RATE']
    BATCH_SIZE=kwargs['BATCH_SIZE']
    GUIDANCE_SCALE=kwargs['GUIDANCE_SCALE']
    REPRESENTATION=kwargs['REPRESENTATION']

    fname, f_alpha, f_sep_mel = setting_result_folder(sep_data1['text'][0])

    pkboo = AudioPeekabooSeparator(sep_data1, dataset, device, representation=REPRESENTATION).to(device)
    params = list(pkboo.parameters())
    for param in pkboo.parameters():
        param.requires_grad = True
    optim = torch.optim.SGD(params, lr=LEARNING_RATE)
    
    losses1 = np.zeros(NUM_ITER)
    losses2 = np.zeros(NUM_ITER)
    preds_mean = np.zeros(NUM_ITER)
    noises_mean = np.zeros(NUM_ITER)
    preds_absmean = np.zeros(NUM_ITER)
    noises_absmean = np.zeros(NUM_ITER)

    NUM_PREVIEWS = 10
    preview_interval = max(1, NUM_ITER // NUM_PREVIEWS)  # 10번의 미리보기를 표시

    save_initial_state(sep_data1['log_mel_spec'], fname, latent_diffusion)
    trigger = 0

    try:
        for iter_idx in tqdm(range(NUM_ITER)):
            alpha = pkboo.alpha()  # [C,H:1024,w:64]
            
            composite_batch = None
            noise_mse_loss = 0
            # for __ in range(BATCH_SIZE):
            if BATCH_SIZE == 1:
                composite_batch = pkboo()
                noise_mse_loss, pred, noise, pred_abs, noise_abs = training_step(composite_batch, latent_diffusion, guidance_scale=GUIDANCE_SCALE)  # 가이던스 스케일 조정하는거 확인 부탁
            else:
                raise ValueError

            alpha_regularization_loss = alpha.sum()
            (alpha_regularization_loss * GRAVITY).backward()
            optim.step()
            optim.zero_grad()
            alpha_reg_loss = alpha_regularization_loss.item()

            losses1[iter_idx] = noise_mse_loss
            # print(noise_mse_loss)
            if noise_mse_loss > 1 and trigger == 0:
                whenstartdiverge = iter_idx
                trigger = 1
            losses2[iter_idx] = alpha_reg_loss
            preds_mean[iter_idx] = pred
            noises_mean[iter_idx] = noise
            preds_absmean[iter_idx] = pred_abs
            noises_absmean[iter_idx] = noise_abs
        
            with torch.no_grad():
                if not iter_idx % preview_interval:
                    save_intermediate_results(
                        iter_idx,
                        alpha,
                        composite_batch,
                        f_alpha,
                        f_sep_mel
                    )

        # 최종 결과 저장
        save_final_results(
            composite_batch,
            fname,
            losses1,
            losses2,
            preds_mean,
            noises_mean,
            preds_absmean,
            noises_absmean,
            latent_diffusion
        )
        print(whenstartdiverge, same_count)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current results...")
        save_final_results(
            composite_batch,
            fname,
            losses1[:iter_idx],
            losses2[:iter_idx],
            preds_mean[:iter_idx],
            noises_mean[:iter_idx],
            preds_absmean[:iter_idx],
            noises_absmean[:iter_idx],
            latent_diffusion
        )

    finally:
        kwargs['GRAVITY'] = GRAVITY
        kwargs['LEARNING_RATE'] = LEARNING_RATE
        save_audio_metadata(sep_data1, sep_data2, kwargs, os.path.join(fname, "metadata.yaml"))
        torch.cuda.empty_cache()

    '''
    def mel_spectrogram_to_waveform(self, mel, savepath=".", bs=None, name="outwav", save=True):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.first_stage_model.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        if save:
            self.save_waveform(waveform, savepath, name)
        return waveform
    
    # # 아 그리고 cond1은 dict로  결과가 나오더라
    # audio1, cond1 = latent_diffusion.get_input(batch1, latent_diffusion.first_stage_key)
    # audio2, cond2 = latent_diffusion.get_input(batch2, latent_diffusion.first_stage_key)
    # print(cond1,cond1.shape,cond1.max())
    # composite_audio = audio1 + audio2  # BCHW 1,8,256,16

    #########################################
    # 남은 해야할 것들:
    # 1. 일단 sep class 완성하기. (O)
    # 1-1. 이때 로그 스펙을 전처리하고 돌려놓는거 (-11.5129) 해야함.
    # 2. 가이던스 수정
    # 3. 이미지 파일로 저장하고 로스 기록하는거 구현
    #########################################
    '''
'''
    def train_step(self, text_embeddings: torch.Tensor, pred_rgb: torch.Tensor, guidance_scale: float = 100, t: Optional[int] = None):

        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False).to(torch.float16) ##

        if t is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # VAE를 사용하여 이미지를 latents로 인코딩 / grad 필요.
        latents = self.encode_imgs(pred_rgb_512)

        # unet으로 noise residual 예측 / NO grad.
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t.cpu())
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

        # guidance 수행 (high scale from paper)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2   
        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # grad에서 item을 생략하고 자동 미분 불가능하므로, 수동 backward 수행.
        latents.backward(gradient=grad, retain_graph=True)
        return 0  # dummy loss value
'''
def training_step(composite_batch, latent_diffusion, guidance_scale=100):
    global same_count

    x, cond = latent_diffusion.get_input(composite_batch, latent_diffusion.first_stage_key, unconditional_prob_cfg=0.0)
    # print(cond.keys())  # dict_keys(['film_clap_cond1'])
    # print(cond['film_clap_cond1'])
    _, uncond = latent_diffusion.get_input(composite_batch, latent_diffusion.first_stage_key, unconditional_prob_cfg=1.0)
    # print(uncond.keys())  # dict_keys(['film_clap_cond1'])
    # print(uncond['film_clap_cond1'])

    co, unco = cond['film_clap_cond1'], uncond['film_clap_cond1']
    # assert not torch.equal(co, unco), f'Cond:\n{co}\n\nUncond\n{unco}'
    if torch.equal(co, unco):
        same_count += 1

    with torch.no_grad():
        t = torch.randint(int(latent_diffusion.num_timesteps * 0.02), int(latent_diffusion.num_timesteps * 0.98) + 1 , (x.shape[0],), device=latent_diffusion.device).long()         ############   t 조정
        noise = torch.randn_like(x)
        x_noisy = latent_diffusion.q_sample(x_start=x, t=t, noise=noise)
        # print(x_noisy.shape)  # torch.Size([1, 8, 256, 16])
        pred_noise_text = latent_diffusion.apply_model(x_noisy, t, cond)
        pred_noise_uncond = latent_diffusion.apply_model(x_noisy, t, uncond)
        pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_text - pred_noise_uncond)

    # w(t), sigma_t^2 
    w = (1 - latent_diffusion.alphas_cumprod[t])
    grad = w * (pred_noise - noise)
    x.backward(gradient=grad, retain_graph=True)

    custom_mse_loss = ((pred_noise - noise) ** 2).mean().item()

    # grad에서 item을 생략하고 자동 미분 불가능하므로, 수동 backward 수행.
    
    return custom_mse_loss, pred_noise.mean().item(), noise.mean().item(), pred_noise.abs().mean().item(), noise.abs().mean().item()  # dummy loss value


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    
    config_yaml_path = os.path.join('audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml')
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)
    exp_group_name = os.path.basename(os.path.dirname(config_yaml_path))
    exp_name = os.path.basename(config_yaml_path.split(".")[0])
    perform_validation = False
    
    reload_from_ckpt = 'data/checkpoints/audioldm-s-full.ckpt'
    if reload_from_ckpt is not None:
        config_yaml["reload_from_ckpt"] = reload_from_ckpt

    if perform_validation:
        config_yaml["model"]["params"]["cond_stage_config"]["crossattn_audiomae_generated"]["params"]["use_gt_mae_output"] = False
        config_yaml["step"]["limit_val_batches"] = None

    pkboo_h_prms = {
        'GRAVITY': 1e-1/2,  # 1e-1/2,
        'NUM_ITER': 300,
        'LEARNING_RATE': 1e-6,  # 1e-5, 
        'BATCH_SIZE': 1,
        'GUIDANCE_SCALE': 100,
        'REPRESENTATION': 'fourier',
    }

    pkboo_h_prms = {
        'GRAVITY': 6e-3,  # 1e-1/2,
        'NUM_ITER': 300,
        'LEARNING_RATE': 0.01,  # 1e-5, 
        'BATCH_SIZE': 1,
        'GUIDANCE_SCALE': 100,
        'REPRESENTATION': 'raster',
    }
    ## raster 기준 (GRAVITY: 0.1 / LEARNING_RATE: 0.01) 까지는 e가 1에서 진동

    run_pkboo(config_yaml, config_yaml_path, exp_group_name, exp_name, perform_validation, **pkboo_h_prms)
