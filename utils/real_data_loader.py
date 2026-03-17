#!/usr/bin/env python3
"""
Real Data Loader for AETHER Training
真实数据加载器 - 参考FarGanSOTA设计，组件化数据加载
使用FeatureSpec配置对象，不再依赖硬编码特征切片
"""

import os
import json
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Optional, Dict, List
from pathlib import Path
from .feature_spec import get_default_feature_spec, FeatureSpec
try:
    from ..models.feature_adapter import get_fargan_feature_spec
except ImportError:  # pragma: no cover
    from models.feature_adapter import get_fargan_feature_spec


class AETHERRealDataset(Dataset):
    """
    AETHER真实数据集加载器
    支持大规模数据加载和能量选择
    使用FeatureSpec配置，不依赖硬编码特征索引
    """

    def __init__(self,
                 data_dir: str,
                 sequence_length: int = 800,  # 8秒音频
                 frame_size: int = 160,
                 max_samples: Optional[int] = None,
                 stride: int = 400,
                 energy_selection: bool = True,
                 feature_spec: Optional[FeatureSpec] = None,
                 features_file: Optional[str] = None,
                 audio_file: Optional[str] = None,
                 validation_split: float = 0.0,
                 split_mode: str = "train",
                 teacher_latent_file: Optional[str] = None):
        """
        Args:
            data_dir: 数据目录
            sequence_length: 序列长度(帧)
            frame_size: 每帧音频样本数
            max_samples: 最大样本数
            stride: 序列间隔
            energy_selection: 是否基于能量选择
            feature_spec: 特征规范配置，默认使用48维标准配置
        """

        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.stride = stride
        self.energy_selection = energy_selection
        self.feature_spec = feature_spec or get_default_feature_spec()
        self.validation_split = validation_split
        self.split_mode = split_mode

        # 文件路径
        self.data_dir = Path(data_dir)

        # 使用直接指定的文件路径，否则回退到默认路径策略
        if features_file is not None:
            features_path = Path(features_file)
        else:
            # 根据特征规范选择对应的特征文件
            if self.feature_spec.total_dim == 36:
                features_path = self.data_dir / "lmr_export" / "features_36_fargan_baseline.f32"
            else:
                features_path = self.data_dir / "lmr_export" / "features_48_complete.f32"

        if audio_file is not None:
            audio_path = Path(audio_file)
        else:
            audio_path = self.data_dir / "out_speech.pcm"

        # 检查文件存在
        for path, name in [(features_path, "features"), (audio_path, "audio")]:
            if not path.exists():
                raise FileNotFoundError(f"{name} file not found: {path}")

        print(f"Loading real data from {data_dir}...")

        # 加载特征 - 使用内存映射，避免将全量数据载入内存
        feature_dim = self.feature_spec.total_dim
        self._features_mmap = np.memmap(str(features_path), dtype=np.float32, mode='r')
        total_features = self._features_mmap.size // feature_dim
        self.features = self._features_mmap.reshape(-1, feature_dim)
        print(f"  Loaded features: {self.features.shape} (使用{feature_dim}维特征规范)")

        # 加载音频（int16）- 使用内存映射，按需转换为float
        self._audio_mmap = np.memmap(str(audio_path), dtype=np.int16, mode='r')
        self.audio = self._audio_mmap  # 延迟转换为float32（在__getitem__时）
        print(f"  Loaded audio: {self.audio.shape} ({len(self.audio)/16000/3600:.1f}h)")

        # 严格验证10ms帧率对齐 (16kHz采样率下，160样本=10ms)
        audio_frames = len(self.audio) // self.frame_size
        feature_frames = len(self.features)

        print(f"  音频帧数: {audio_frames:,} (基于{self.frame_size}样本/帧)")
        print(f"  特征帧数: {feature_frames:,}")

        # 确保帧率对齐
        if abs(audio_frames - feature_frames) > 1:
            print(f"  ⚠️ 帧数不匹配较大: 音频{audio_frames} vs 特征{feature_frames}")

        min_frames = min(audio_frames, feature_frames)
        print(f"  对齐到: {min_frames:,} 帧 ({min_frames*10:.1f}ms)")

        # 严格裁剪确保对齐
        self.features = self.features[:min_frames]
        audio_samples = min_frames * self.frame_size
        self.audio = self.audio[:audio_samples]

        # 验证最终对齐
        final_audio_frames = len(self.audio) // self.frame_size
        final_feature_frames = len(self.features)
        assert final_audio_frames == final_feature_frames, \
            f"对齐后仍不匹配: 音频{final_audio_frames} vs 特征{final_feature_frames}"

        # 可选：加载 StableCodec teacher latent，并在帧级对齐
        self.teacher_latent: Optional[torch.Tensor] = None
        if teacher_latent_file is not None:
            latent_path = Path(teacher_latent_file)
            if latent_path.exists():
                try:
                    latent = torch.load(latent_path, map_location="cpu")
                    if isinstance(latent, torch.Tensor) and latent.dim() == 2:
                        t_latent = latent.size(0)
                        t_min = min(t_latent, final_feature_frames)
                        if t_latent != final_feature_frames:
                            print(
                                f"  [StableCodecTeacher] latent frames ({t_latent}) "
                                f"!= feature frames ({final_feature_frames}); aligning to {t_min}"
                            )
                        self.teacher_latent = latent[:t_min].contiguous()
                        if t_min < final_feature_frames:
                            # 保守处理：进一步裁剪特征和音频以匹配teacher长度
                            self.features = self.features[:t_min]
                            audio_samples = t_min * self.frame_size
                            self.audio = self.audio[:audio_samples]
                    else:
                        print(f"  [StableCodecTeacher] Ignoring latent {latent_path} with shape {getattr(latent, 'shape', None)}")
                except Exception as _e:
                    print(f"  [StableCodecTeacher] Failed to load latent from {latent_path}: {_e}")
            else:
                print(f"  [StableCodecTeacher] Latent file not found: {latent_path}")

        # 计算有效序列位置
        valid_start = 2  # 跳过前2帧
        valid_end = min_frames - sequence_length - 2

        if valid_end <= valid_start:
            raise ValueError(f"数据不足，无法创建长度为{sequence_length}的序列")

        # 生成序列起始位置
        if energy_selection and sequence_length >= 400:  # 长序列使用能量选择
            all_positions = self._select_high_energy_positions(
                valid_start, valid_end, stride, max_samples
            )
        else:
            # 短序列使用均匀步长
            all_positions = list(range(valid_start, valid_end, stride))
            if max_samples is not None and len(all_positions) > max_samples:
                # 均匀采样而不是截断
                indices = np.linspace(0, len(all_positions)-1, max_samples, dtype=int)
                all_positions = [all_positions[i] for i in indices]

        # 应用训练/验证分割（非重叠位置）
        if validation_split > 0.0:
            total_positions = len(all_positions)
            val_size = int(total_positions * validation_split)

            # 使用非重叠策略：验证集使用数据的最后部分
            if split_mode == "val":
                self.valid_positions = all_positions[-val_size:] if val_size > 0 else []
                split_info = f"validation ({val_size}/{total_positions})"
            else:
                self.valid_positions = all_positions[:-val_size] if val_size > 0 else all_positions
                split_info = f"training ({len(self.valid_positions)}/{total_positions})"

            print(f"  Applied {validation_split:.1%} split -> {split_info}")
        else:
            self.valid_positions = all_positions
            split_info = "no split"

        print(f"  Valid sequences: {len(self.valid_positions):,} (stride={stride}, energy_based={energy_selection and sequence_length>=400}, {split_info})")

        # 预计算CSI缓存以优化运行时性能（仅对中小数据集启用）
        if len(self.valid_positions) > 0 and len(self.valid_positions) <= 50000:
            self._precompute_csi_cache()
        else:
            self.csi_cache = {}
            if len(self.valid_positions) > 50000:
                print(f"  ⚡ Skipping CSI precomputation for large dataset ({len(self.valid_positions):,} samples)")
                print(f"  📋 Using on-demand CSI generation for better startup time")

    def _select_high_energy_positions(self, valid_start: int, valid_end: int, stride: int, max_samples: Optional[int]):
        """基于音频能量密度和多样性选择序列位置"""
        candidate_positions = list(range(valid_start, valid_end, stride))

        if len(candidate_positions) == 0:
            return candidate_positions

        # 大数据集优化：当候选数非常大时，跳过逐段能量计算，直接返回均匀步长位置
        if len(candidate_positions) > 200_000:
            print(f"    ⚡ Large dataset detected (N={len(candidate_positions):,}), skipping energy scan")
            return candidate_positions

        # 计算每个候选位置的能量和方差（按需将音频段转为float32并归一化）
        energies = []
        variances = []
        for pos in candidate_positions:
            audio_start = pos * self.frame_size
            audio_end = audio_start + self.sequence_length * self.frame_size
            seg_i16 = self.audio[audio_start:audio_end]
            seg = seg_i16.astype(np.float32) / 32768.0 if len(seg_i16) > 0 else seg_i16
            energy = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 0.0
            variance = np.var(seg) if len(seg) > 0 else 0.0
            energies.append(energy)
            variances.append(variance)

        energies = np.asarray(energies, dtype=np.float32)
        variances = np.asarray(variances, dtype=np.float32)

        # 改进的采样策略：结合能量和多样性
        if max_samples is not None and max_samples < len(candidate_positions):
            # 70%高能量样本 + 30%随机样本保证多样性
            high_energy_count = int(max_samples * 0.7)
            diverse_count = max_samples - high_energy_count

            # 选择高能量样本
            energy_sorted_indices = np.argsort(energies)[::-1]
            high_energy_indices = energy_sorted_indices[:high_energy_count]

            # 从剩余位置中随机选择，偏向高方差的位置
            remaining_indices = energy_sorted_indices[high_energy_count:]
            if len(remaining_indices) > 0:
                # 使用加权随机采样，方差越大权重越大
                remaining_variances = variances[remaining_indices]
                if np.sum(remaining_variances) > 0:
                    weights = remaining_variances / np.sum(remaining_variances)
                    diverse_indices = np.random.choice(
                        remaining_indices,
                        size=min(diverse_count, len(remaining_indices)),
                        replace=False,
                        p=weights
                    )
                else:
                    # 如果方差都为0，则均匀随机采样
                    diverse_indices = np.random.choice(
                        remaining_indices,
                        size=min(diverse_count, len(remaining_indices)),
                        replace=False
                    )
            else:
                diverse_indices = []

            selected_indices = np.concatenate([high_energy_indices, diverse_indices])
        else:
            # 如果没有样本限制，按能量排序返回所有位置
            energy_sorted_indices = np.argsort(energies)[::-1]
            selected_indices = energy_sorted_indices

        # 返回按帧位置排序的选中位置
        selected_positions = [candidate_positions[i] for i in selected_indices]
        selected_positions.sort()

        print(f"    Selected {len(selected_positions):,}/{len(candidate_positions):,} samples")
        if len(energies) > 0:
            print(f"    Energy range: [{np.min(energies):.4f}, {np.max(energies):.4f}]")
            print(f"    Variance range: [{np.min(variances):.6f}, {np.max(variances):.6f}]")
            if max_samples is not None and max_samples < len(candidate_positions):
                print(f"    Strategy: {int(max_samples * 0.7)} high-energy + {max_samples - int(max_samples * 0.7)} diverse")

        return selected_positions

    def __len__(self):
        return len(self.valid_positions)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # 低RMS片段过滤：若片段能量过低，尝试向后寻找可用片段（最多尝试5次）
        attempt = 0
        pos_idx = int(idx)
        features = None
        audio = None
        while attempt < 5:
            start_frame = self.valid_positions[pos_idx]
            end_frame = start_frame + self.sequence_length

            # 获取特征序列
            features = torch.tensor(self.features[start_frame:end_frame], dtype=torch.float32)

            # 获取对应音频（按需将int16转换为float32并归一化）
            audio_start = start_frame * self.frame_size
            audio_end = audio_start + self.sequence_length * self.frame_size
            seg_i16 = self.audio[audio_start:audio_end]
            seg = seg_i16.astype(np.float32) / 32768.0 if len(seg_i16) > 0 else seg_i16
            audio = torch.from_numpy(seg.copy()).to(dtype=torch.float32)

            # 计算RMS，放宽阈值以获得更多训练样本
            rms = float(torch.sqrt(torch.mean(audio.pow(2)) + 1e-12).item())
            if rms >= 5e-5:  # 降低阈值从1e-4到5e-5
                break
            attempt += 1
            pos_idx = (pos_idx + 1) % len(self.valid_positions)

        # StableCodec teacher latent 片段（若可用）
        if self.teacher_latent is not None and end_frame <= self.teacher_latent.size(0):
            teacher_seg = self.teacher_latent[start_frame:end_frame, :]
            teacher_seg = teacher_seg.to(dtype=torch.float32)
        else:
            # 使用空张量占位，便于 collate；未启用teacher时不会参与loss
            teacher_seg = torch.empty(0, 0, dtype=torch.float32)

        # 使用预计算的CSI缓存（高性能版本）
        if idx in self.csi_cache:
            csi_cached = self.csi_cache[idx]
            # 创建fading_onehot
            fading_onehot = torch.zeros(8)
            fading_onehot[csi_cached['fading_type']] = 1.0

            csi_dict = {
                'snr_db': torch.tensor(csi_cached['snr_db'], dtype=torch.float32),
                'ber': torch.tensor(csi_cached['ber'], dtype=torch.float32),
                'fading_onehot': fading_onehot,
            }
        else:
            # 回退到实时生成（用于向后兼容）
            csi_dict = self._generate_csi(audio.numpy(), features, pos_idx)

        return {
            'x': features,
            'y': features.clone(),  # 自编码任务
            'audio': audio,
            'csi': csi_dict,
            'seq_idx': idx,
            'teacher_latent': teacher_seg,
        }

    def _generate_csi(self, audio: np.ndarray, features: torch.Tensor, idx: int) -> Dict[str, torch.Tensor]:
        """生成统一的10维信道状态信息（snr_db(1) + ber(1) + fading_onehot(8)）。"""
        # 基于音频真实能量计算SNR
        # 数值稳健：能量下限裁剪，避免 log10 产生 -inf/NaN
        energy = float(np.mean(audio ** 2)) if len(audio) > 0 else 0.0
        energy = max(energy, 1e-12)
        snr_base = 10 * np.log10(energy) + 35
        snr_db = np.clip(snr_base + np.random.normal(0, 3), 0, 30)

        # 基于SNR计算BER
        ber = np.clip(10 ** (-snr_db / 15), 1e-5, 0.05)

        # 衰落类型循环 - 保持8类以匹配原始配置
        fading_onehot = np.zeros(8)
        fading_type = (idx // 2000) % 8
        fading_onehot[fading_type] = 1.0

        return {
            'snr_db': torch.tensor(snr_db, dtype=torch.float32),  # [1] 维
            'ber': torch.tensor(ber, dtype=torch.float32),        # [1] 维
            'fading_onehot': torch.from_numpy(fading_onehot).float(),  # [8] 维
        }

    def _precompute_csi_cache(self):
        """预计算CSI缓存以优化性能"""
        print(f"  Precomputing CSI cache for {len(self.valid_positions):,} samples...")

        # 预计算所有样本的CSI信息
        self.csi_cache = {}

        # 批量预计算以提高效率
        batch_size = 1000
        for i in range(0, len(self.valid_positions), batch_size):
            end_idx = min(i + batch_size, len(self.valid_positions))

            for j in range(i, end_idx):
                pos_idx = j
                start_frame = self.valid_positions[pos_idx]

                # 获取音频片段用于CSI计算
                audio_start = start_frame * self.frame_size
                audio_end = audio_start + self.sequence_length * self.frame_size
                audio_segment = self.audio[audio_start:audio_end]

                # 基于音频能量计算SNR（无随机噪声，保证可重现）
                energy = float(np.mean(audio_segment ** 2)) if len(audio_segment) > 0 else 0.0
                energy = max(energy, 1e-12)
                snr_base = 10 * np.log10(energy) + 35
                snr_db = np.clip(snr_base, 0, 30)  # 移除随机噪声提高性能

                # 基于SNR计算BER
                ber = np.clip(10 ** (-snr_db / 15), 1e-5, 0.05)

                # 衰落类型循环
                fading_type = (pos_idx // 2000) % 8

                # 直接存储numpy数组，避免重复tensor创建
                self.csi_cache[pos_idx] = {
                    'snr_db': float(snr_db),
                    'ber': float(ber),
                    'fading_type': int(fading_type)
                }

            if (i + batch_size) % 10000 == 0:
                print(f"    Processed {i + batch_size:,}/{len(self.valid_positions):,} samples")

        print(f"  CSI cache precomputed: {len(self.csi_cache):,} entries")

    def get_info(self) -> dict:
        """获取数据集信息"""
        return {
            'total_frames': len(self.features),
            'total_samples': len(self.valid_positions),
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_spec.total_dim,
            'audio_duration_hours': len(self.audio) / 16000 / 3600,
            'frame_size': self.frame_size,
            'sequence_duration_seconds': self.sequence_length * self.frame_size / 16000,
            'feature_spec_info': self.feature_spec.get_feature_info(),
            'validation_split': getattr(self, 'validation_split', 0.0),
            'split_mode': getattr(self, 'split_mode', 'train')
        }


def create_aether_data_loader(
    data_dir: str = "/home/bluestar/FARGAN/opus/data_cn",
    sequence_length: int = 200,  # 修复默认值
    batch_size: int = 32,        # 修复默认值
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    energy_selection: bool = True,
    test_mode: bool = False,
    feature_spec_type: str = "aether",
    features_file: Optional[str] = None,
    audio_file: Optional[str] = None,
    validation_split: float = 0.0,
    split_mode: str = "train",
    stride_frames: Optional[int] = None,  # 新增可配置步幅参数
    # Optional DataLoader performance knobs
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    distributed: bool = False,
) -> Tuple[DataLoader, AETHERRealDataset]:
    """
    创建AETHER数据加载器

    Args:
        data_dir: 数据目录
        sequence_length: 序列长度(帧)
        batch_size: 批大小
        max_samples: 最大样本数
        num_workers: 工作进程数
        energy_selection: 是否使用能量选择
        test_mode: 测试模式
        feature_spec_type: 特征规范类型
        features_file: 特征文件路径
        audio_file: 音频文件路径
        validation_split: 验证集比例 (0.0-1.0)
        split_mode: 分割模式 ("train" 或 "val")
        stride_frames: 步幅大小(帧)，None则使用自动策略

    Returns:
        dataloader, dataset
    """

    print(f"Creating AETHER data loader (test_mode={test_mode}, distributed={distributed})...")

    # 步幅策略：可配置或智能自适应
    if stride_frames is not None:
        stride = stride_frames
        print(f"Using user-specified stride: {stride} frames")
    else:
        # 默认策略：根据序列长度智能选择步幅
        if sequence_length >= 800:  # 8秒以上：使用更大步幅减少重叠
            stride = sequence_length // 8  # 87.5%重叠 (改进：从//16到//8)
        elif sequence_length >= 400:  # 4-8秒：中等步幅
            stride = sequence_length // 6  # 83.3%重叠
        else:  # 短序列：较小步幅但仍比原来大
            stride = max(sequence_length // 4, 60)  # 75%重叠或至少60帧
        print(f"Using adaptive stride: {stride} frames for sequence_length={sequence_length}")

    # 根据特征规范类型选择对应的配置
    if feature_spec_type == "fargan":
        feature_spec = get_fargan_feature_spec()
        print(f"Using FARGAN feature spec: 36 dimensions")
    else:
        feature_spec = get_default_feature_spec()
        print(f"Using AETHER feature spec: 48 dimensions")

    # 创建数据集
    dataset = AETHERRealDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        frame_size=160,
        max_samples=max_samples,
        stride=stride,
        energy_selection=energy_selection,
        feature_spec=feature_spec,
        features_file=features_file,
        audio_file=audio_file,
        validation_split=validation_split,
        split_mode=split_mode
    )

    # 打印数据集信息
    info = dataset.get_info()
    print(f"Dataset info:")
    print(f"  Total frames: {info['total_frames']:,}")
    print(f"  Training samples: {info['total_samples']:,}")
    print(f"  Audio duration: {info['audio_duration_hours']:.1f}h")
    print(f"  Sequence length: {info['sequence_length']} frames ({info['sequence_duration_seconds']:.1f}s)")

    # 🔥 CPU优化：基于分析优化DataLoader配置
    # 推荐配置：worker=4, prefetch=2, pin_memory=True, persistent_workers=True
    dl_pin_memory = True if pin_memory is None else bool(pin_memory)
    dl_persistent_workers = (num_workers > 0) if persistent_workers is None else bool(persistent_workers)

    # 优化prefetch_factor：从4降至2以减少CPU-GPU队列竞争
    if prefetch_factor is None:
        dl_prefetch_factor = 2 if num_workers > 0 else None
    else:
        dl_prefetch_factor = int(prefetch_factor)

    # 如果workers过多，自动调整以优化性能
    optimized_workers = num_workers
    if num_workers > 6:
        optimized_workers = 4
        print(f"  ⚡ Auto-optimized workers: {num_workers} → {optimized_workers} (reduce CPU-GPU contention)")

    # 分布式采样器：在 DDP 下确保每个 rank 看到不同子集
    sampler = None
    if distributed and dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=not test_mode)
        print(
            f"Using DistributedSampler: world_size={dist.get_world_size()}, "
            f"rank={dist.get_rank()}, shuffle={not test_mode}"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and not test_mode),
        sampler=sampler,
        num_workers=optimized_workers,
        pin_memory=dl_pin_memory,
        drop_last=True,
        persistent_workers=dl_persistent_workers,
        prefetch_factor=dl_prefetch_factor,
    )

    print(f"Data loader created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {len(dataloader):,}")
    print(f"  Workers: {optimized_workers} (pin_memory={dl_pin_memory}, persistent_workers={dl_persistent_workers}, prefetch={dl_prefetch_factor})")

    return dataloader, dataset


# ---- Combined multi-expert dataset (混合批：按配比从四类数据集中采样) ----

EXPERT_KEYS = ['harmonic', 'transient', 'burst_inpaint', 'low_snr']


class CombinedExpertDataset(Dataset):
    """将四个专家数据集合并为一个数据集，按配比在 __getitem__ 中随机选择子集并返回样本。

    - 使用内部的 AETHERRealDataset 实例（每类一个）
    - 每个样本附带 'expert_class' 标签：0=harmonic, 1=transient, 2=burst_inpaint, 3=low_snr
    - __len__ 返回四个子集长度之和的近似值（用于进度估算）；采样随机，不使用 idx 索引真实位置
    """
    def __init__(
        self,
        datasets: Dict[str, AETHERRealDataset],
        mix_ratio: List[float],
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.mix_ratio = np.array(mix_ratio, dtype=np.float64)
        self.mix_ratio = self.mix_ratio / max(1e-8, self.mix_ratio.sum())

        # 构建类别到索引的映射
        self.key_to_id = {k: i for i, k in enumerate(EXPERT_KEYS)}
        self.id_to_key = {i: k for k, i in self.key_to_id.items()}

        # 记录每个子集长度用于采样
        self.lengths = {k: len(ds) for k, ds in datasets.items()}
        self.total_len = int(sum(self.lengths.values()))

        print("CombinedExpertDataset:")
        for k in EXPERT_KEYS:
            if k in self.lengths:
                print(f"  - {k}: {self.lengths[k]:,} samples")
        print(f"  Mix ratio: {self.mix_ratio.tolist()}")

        # 为类别抽样构建累积分布
        self.cumprob = np.cumsum(self.mix_ratio)

    def __len__(self) -> int:
        # 近似总长度（不严格使用），仅用于DataLoader进度与epoch大小控制
        return self.total_len

    def _sample_class_id(self) -> int:
        r = np.random.rand()
        for i, cp in enumerate(self.cumprob):
            if r <= cp:
                return i
        return len(self.cumprob) - 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 忽略 idx，按配比随机选择类别
        cid = self._sample_class_id()
        key = self.id_to_key[cid]
        ds = self.datasets[key]

        # 在该子集中随机取一个样本
        ridx = np.random.randint(0, len(ds))
        item = ds[ridx]

        # 附加类别标签
        item['expert_class'] = torch.tensor(cid, dtype=torch.long)
        return item


def create_combined_data_loader(
    data_root: str,
    sequence_length: int,
    batch_size: int,
    frame_size: int = 160,
    stride_frames: Optional[int] = None,
    energy_selection: bool = True,
    feature_dims: int = 36,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> Tuple[DataLoader, CombinedExpertDataset]:
    """创建混合专家数据加载器。

    采用固定命名规则在 data_root 下寻找四类数据；若存在 small-200k 版本优先使用，否则回退到 *_enhanced 命名。
    """
    print("Creating CombinedExpertDataset (multi-expert mixed batches)...")
    # 根据feature_dims选择规范
    assert feature_dims in (36, 48), "Only 36 or 48 dims supported"
    if feature_dims == 36:
        feature_spec = get_fargan_feature_spec()
    else:
        feature_spec = get_default_feature_spec()
    # 文件名模板（优先small-200k）
    root = Path(data_root)
    small_ok = (root / 'harmonic_200k_36.f32').exists()

    def paths_for(name: str):
        if small_ok:
            features_path = root / f"{name}_200k_36.f32"
            audio_path = root / f"{name}_200k.pcm"
            latent_path = root / f"{name}_200k.pt"
        else:
            features_path = root / f"{name}_enhanced_36.f32"
            audio_path = root / f"{name}_enhanced.pcm"
            latent_path = root / f"{name}_enhanced.pt"
        return str(features_path), str(audio_path), str(latent_path)

    datasets: Dict[str, AETHERRealDataset] = {}
    for k in EXPERT_KEYS:
        fpath, apath, lpath = paths_for(k)
        try:
            # 若不存在对应的 StableCodec latent，则对该子集禁用teacher
            latent_file = lpath if Path(lpath).exists() else None
            if latent_file is None:
                print(f"  [WARN] StableCodec latent for expert '{k}' not found at {lpath}; teacher disabled for this subset")

            ds = AETHERRealDataset(
                data_dir=data_root,
                sequence_length=sequence_length,
                frame_size=frame_size,
                max_samples=max_samples,
                stride=stride_frames if stride_frames is not None else max(100, sequence_length // 8),
                energy_selection=energy_selection,
                feature_spec=feature_spec,
                features_file=fpath,
                audio_file=apath,
                teacher_latent_file=latent_file,
            )
            datasets[k] = ds
        except FileNotFoundError as e:
            print(f"  [WARN] Missing subset for {k}: {e}")

    if not datasets:
        raise RuntimeError("No subsets found for CombinedExpertDataset")

    # 默认等比混合（均衡）
    mix_ratio = [1.0 if k in datasets else 0.0 for k in EXPERT_KEYS]
    s = sum(mix_ratio)
    mix_ratio = [x / s for x in mix_ratio]

    combined = CombinedExpertDataset(datasets=datasets, mix_ratio=mix_ratio)

    # DataLoader参数与单集一致
    dl_pin_memory = True if pin_memory is None else bool(pin_memory)
    dl_persistent_workers = (num_workers > 0) if persistent_workers is None else bool(persistent_workers)
    dl_prefetch_factor = 2 if (prefetch_factor is None and num_workers > 0) else prefetch_factor

    loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=dl_pin_memory,
        drop_last=True,
        persistent_workers=dl_persistent_workers,
        prefetch_factor=dl_prefetch_factor,
    )

    print("Data loader (combined) created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {len(loader):,}")
    print(f"  Workers: {num_workers} (pin_memory={dl_pin_memory}, persistent_workers={dl_persistent_workers}, prefetch={dl_prefetch_factor})")
    return loader, combined


def create_train_val_loaders(
    validation_split: float = 0.15,
    **kwargs
) -> Tuple[DataLoader, DataLoader, AETHERRealDataset, AETHERRealDataset]:
    """
    创建训练和验证数据加载器（非重叠分割）

    Args:
        validation_split: 验证集比例 (0.0-1.0)
        **kwargs: 传递给create_aether_data_loader的其他参数

    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    if validation_split <= 0.0:
        raise ValueError("validation_split must be > 0.0 for creating train/val split")

    # 移除split相关参数（如果存在）
    kwargs.pop('validation_split', None)
    kwargs.pop('split_mode', None)

    print(f"Creating train/val split with {validation_split:.1%} validation...")

    # 创建训练集
    train_loader, train_dataset = create_aether_data_loader(
        validation_split=validation_split,
        split_mode="train",
        **kwargs
    )

    # 创建验证集
    # 验证集使用较小的batch_size以节省内存
    val_batch_size = min(kwargs.get('batch_size', 4), 32)
    val_kwargs = kwargs.copy()
    val_kwargs['batch_size'] = val_batch_size
    val_kwargs['num_workers'] = min(val_kwargs.get('num_workers', 4), 2)  # 验证时减少workers

    val_loader, val_dataset = create_aether_data_loader(
        validation_split=validation_split,
        split_mode="val",
        **val_kwargs
    )

    print(f"Train/Val split completed:")
    print(f"  Training: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print(f"  Validation: {len(val_dataset):,} samples, {len(val_loader):,} batches")

    return train_loader, val_loader, train_dataset, val_dataset


def test_aether_data_loader():
    """测试AETHER数据加载器"""
    print("🧪 测试AETHER数据加载器")
    print("=" * 50)

    try:
        dataloader, dataset = create_aether_data_loader(
            sequence_length=800,  # 8秒
            batch_size=2,
            max_samples=100,  # 测试用小数据
            num_workers=0,  # 避免多进程问题
            test_mode=True
        )

        # 测试一个批次
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  Features: {batch['x'].shape}")
            print(f"  Audio: {batch['audio'].shape}")
            print(f"  CSI keys: {list(batch['csi'].keys())}")

            print(f"  Feature range: [{batch['x'].min():.3f}, {batch['x'].max():.3f}]")
            print(f"  Audio range: [{batch['audio'].min():.3f}, {batch['audio'].max():.3f}]")
            print(f"  Audio duration: {batch['audio'].shape[-1] / 16000:.1f}s")
            break

        print("✓ AETHER数据加载器测试通过")
        return True

    except Exception as e:
        print(f"✗ AETHER数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    pass
