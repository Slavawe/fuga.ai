"""Экспериментальные модули (песочница) — единый публичный API.

Реестр технологий:
  NEAT/HyperNEAT   — генетическое программирование (эволюция топологии)
  SNN/нейроморфные — LIF-нейроны + STDP
  HTM              — Hierarchical Temporal Memory (SDR, последовательности)
  BLT              — Byte Latent Transformer (энтропийное патчевание байт)
  Mini Cognitive   — мини H-JEPA + VL-JEPA + VSA
  NonGradient      — полностью безградиентное обучение (HTM+VSA+SNN+NEAT)
  Brain Transplant — перенос hidden states реальной модели в VSA
  Math-Space       — математика + 3D-пространство + учитель-цикл
  VSA-Math         — VSA-предикаты математики + фазовые позиции (FPE)
  HardTeacher      — жёсткий учитель (VSA-верификация гипотез)
  VSADecoder       — генерация через VSA+BLT память + учитель

Использование:
  from astral.experiments import registry
  for name, (module, desc) in registry.items(): ...
"""

registry = {
    "neat_hyperneat": ("astral.experiments.neat_hyperneat", "NEAT + HyperNEAT (эволюция топологии, CPPN)"),
    "snn_neuromorphic": ("astral.experiments.snn_neuromorphic", "SNN: LIF-нейроны + STDP пластичность"),
    "htm_bridge": ("astral.experiments.htm_bridge", "HTM: SDR-кодирование, последовательности"),
    "blt_patcher": ("astral.experiments.blt_patcher", "BLT: энтропийное патчевание байт"),
    "mini_cognitive": ("astral.experiments.mini_cognitive", "Mini H-JEPA + VL-JEPA + VSA"),
    "nongradient_engine": ("astral.experiments.nongradient_engine", "Backprop-Free: HTM+VSA+SNN+NEAT"),
    "brain_transplant": ("astral.experiments.brain_transplant", "Трансплантация hidden states → VSA"),
    "gemma_transplant": ("astral.experiments.gemma_transplant", "Реальная трансплантация HF-модели"),
    "math_teacher": ("astral.experiments.math_teacher", "Математика + пространство + учитель"),
    "vsa_math_link": ("astral.experiments.vsa_math_link", "VSA-предикаты + фазовые позиции"),
    "hard_teacher_vsa_blt": ("astral.experiments.hard_teacher_vsa_blt", "Жёсткий учитель + VSA+BLT"),
    "vsa_decoder": ("astral.experiments.vsa_decoder", "VSA+BLT декодер с учителем"),
}

__all__ = ["registry"]
