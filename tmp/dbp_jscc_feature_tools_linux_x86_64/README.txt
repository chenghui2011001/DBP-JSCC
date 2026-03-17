DBP-JSCC feature extraction tools
=================================

This directory contains tools for feature extraction of DBP-JSCC.

Files:
------
- fargan_demo: Tool for extracting features from PCM audio files
- dump_data: Tool for training data preparation

Usage:
------
1. Extract features from PCM file (output in float32 format):
   ./fargan_demo -features input.pcm output_features.f32

2. Prepare training data (convert 16-bit signed PCM to features and raw PCM):
   ./dump_data -train input.s16 output_features.f32 output_pcm.pcm

Notes:
------
- input.pcm: Raw PCM audio file (format depends on input)
- input.s16: 16-bit signed integer PCM audio file
- output_features.f32: Output feature file in float32 format
- output_pcm.pcm: Extracted raw PCM audio file
