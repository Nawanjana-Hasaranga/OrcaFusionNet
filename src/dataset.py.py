import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class DolphinFusionDataset25(Dataset):
    def __init__(self, folder_path: str, target_sr: int = 192000, mode: str = 'val'):
        self.folder_path = folder_path
        self.target_sr = target_sr
        self.mode = mode
        self.file_paths = []
        self.labels = []
        self.metadata_list = []
        
        # We are sticking to your Champion 3-Class System! (Kept as MS)
        self.class_map = {"Click": 0, "Echo": 1, "MS": 2}
        
        self.spec_freq = T.MelSpectrogram(sample_rate=self.target_sr, n_fft=2048, hop_length=64, n_mels=128)
        self.spec_time = T.MelSpectrogram(sample_rate=self.target_sr, n_fft=256, hop_length=8, n_mels=64)
        self.amplitude_to_db = T.AmplitudeToDB()
        self.compute_deltas = T.ComputeDeltas()
        
        self.time_masking = T.TimeMasking(time_mask_param=15)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=15)

        print(f"Hunting for Dataset files across: {self.folder_path} (Mode: {self.mode}, 25-FEATURE SYSTEM)...")
        
        tapes = {}
        
        # 1. PARSE THE FILENAMES
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav') and '__' in file:
                    name_no_ext = file[:-4]
                    tape_part, metadata_part = name_no_ext.split('__', 1)
                    tape_id = tape_part.replace('.wav', '')
                    
                    parts = metadata_part.split('_')
                    if len(parts) >= 6: 
                        try:
                            label_str = parts[0]
                            start_time = float(parts[1])
                            end_time = float(parts[2])
                            conf = float(parts[3])
                            num_fod = float(parts[4])
                            fod_score = float(parts[5])
                            
                            if tape_id not in tapes: tapes[tape_id] = []
                            
                            tapes[tape_id].append({
                                "path": os.path.join(root, file),
                                "start": start_time, "end": end_time,
                                "duration": end_time - start_time,
                                "conf": conf, "num_fod": num_fod, "fod_score": fod_score,
                                "label_str": label_str
                            })
                        except ValueError:
                            continue

        # 2. CALCULATE THE BASE 20 FEATURES
        for tape_id, items in tapes.items():
            items.sort(key=lambda x: x["start"])
            N = len(items)
            
            confs = np.array([e["conf"] for e in items], dtype=np.float32)
            fods = np.array([e["fod_score"] for e in items], dtype=np.float32)
            
            for i in range(N):
                ev = items[i]
                duration, conf, num_fod, fod_score = ev["duration"], ev["conf"], ev["num_fod"], ev["fod_score"]
                center_time = (ev["start"] + ev["end"]) * 0.5

                if i > 0:
                    prev = items[i - 1]
                    dt_prev = ev["start"] - prev["end"]
                    duration_ratio_prev = duration / prev["duration"] if prev["duration"] > 0 else -1
                    conf_delta_prev = conf - prev["conf"]
                    fod_delta_prev = fod_score - prev["fod_score"]
                else:
                    dt_prev = dt_next = duration_ratio_prev = conf_delta_prev = fod_delta_prev = -1.0

                if i < N - 1:
                    nxt = items[i + 1]
                    dt_next = nxt["start"] - ev["end"]
                    duration_ratio_next = duration / nxt["duration"] if nxt["duration"] > 0 else -1
                    conf_delta_next = conf - nxt["conf"]
                    fod_delta_next = fod_score - nxt["fod_score"]
                else:
                    dt_next = duration_ratio_next = conf_delta_next = fod_delta_next = -1.0

                local_start, local_end = max(0, i - 5), min(N, i + 6)
                local_confs = confs[local_start:local_end]
                local_fods = fods[local_start:local_end]
                
                local_mean_conf = float(np.mean(local_confs)) if len(local_confs) > 0 else 0.0
                local_std_conf = float(np.std(local_confs)) if len(local_confs) > 0 else 0.0

                local_density_10ms = sum(1 for e in items if abs(((e["start"]+e["end"])*0.5) - center_time) <= 0.010) - 1
                local_density_20ms = sum(1 for e in items if abs(((e["start"]+e["end"])*0.5) - center_time) <= 0.020) - 1
                
                position_in_file = i / (N - 1) if N > 1 else 0.0
                
                conf_rank_local = int(np.sum(local_confs > conf))
                fod_rank_local = int(np.sum(local_fods > fod_score))
                
                prev_is_strong = 1 if (i > 0 and items[i - 1]["conf"] > local_mean_conf) else (-1 if i == 0 else 0)

                # 3-Class Label Routing
                lbl = ev["label_str"].lower()
                if "click" in lbl or "lf" in lbl or "hf" in lbl or "us" in lbl: 
                    final_label = 0 # Click
                elif "echo" in lbl: 
                    final_label = 1 # Echo
                else: 
                    final_label = 2 # MS Noise
                
                # The Base 20-Feature Tensor
                meta_20 = torch.tensor([
                    duration, conf, num_fod, fod_score,
                    dt_prev, dt_next, duration_ratio_prev, duration_ratio_next,
                    conf_delta_prev, conf_delta_next, fod_delta_prev, fod_delta_next,
                    local_mean_conf, local_std_conf, float(local_density_10ms), float(local_density_20ms),
                    position_in_file, float(conf_rank_local), float(fod_rank_local), float(prev_is_strong)
                ], dtype=torch.float32)

                self.file_paths.append(ev["path"])
                self.metadata_list.append(meta_20) # We store the 20 here
                self.labels.append(final_label)

    def __len__(self): return len(self.file_paths)

    def pad_or_crop_to_224(self, tensor):
        tensor = tensor[:224, :224]
        pad_bottom, pad_right = max(0, 224 - tensor.shape[0]), max(0, 224 - tensor.shape[1])
        # UPDATED: value is now 0.0 to ensure true black zero-padding
        return F.pad(tensor, (0, pad_right, 0, pad_bottom), value=0.0)

    def __getitem__(self, idx):
        raw_audio, _ = torchaudio.load(self.file_paths[idx])
        raw_audio = raw_audio.view(-1)
        if raw_audio.shape[0] < 2048:
            raw_audio = F.pad(raw_audio, (0, 2048 - raw_audio.shape[0]))
            
        # =========================================================
        # NEW: DYNAMIC AUDIO PHYSICS (THE SPECTRAL 5)
        # =========================================================
        # 1. RMS Energy
        rms = torch.sqrt(torch.mean(raw_audio**2) + 1e-8)
        # 2. Zero Crossing Rate (ZCR)
        zcr = (raw_audio[1:] * raw_audio[:-1] < 0).float().mean()
        # 3. Peak Amplitude
        peak = torch.max(torch.abs(raw_audio))
        # 4. Crest Factor
        crest_factor = peak / rms
        # 5. Signal Entropy (Complexity)
        power = raw_audio**2 + 1e-8
        prob = power / torch.sum(power)
        entropy = -torch.sum(prob * torch.log2(prob))

        spectral_5 = torch.tensor([rms, zcr, peak, crest_factor, entropy], dtype=torch.float32)
        
        # Combine the old 20 features with the new 5 features to make exactly 25!
        full_25_meta = torch.cat((self.metadata_list[idx], spectral_5), dim=0)
        # =========================================================

        # Image Generation (RGB Spec)
        blue = self.pad_or_crop_to_224(self.amplitude_to_db(self.spec_freq(raw_audio)))
        green = self.pad_or_crop_to_224(self.amplitude_to_db(self.spec_time(raw_audio)))
        red = self.compute_deltas(blue)

        if self.mode == 'train':
            blue = self.time_masking(self.freq_masking(blue))
            green = self.time_masking(self.freq_masking(green))
            red = self.time_masking(self.freq_masking(red))

        return torch.stack([red, green, blue], dim=0), full_25_meta, torch.tensor(self.labels[idx], dtype=torch.long)