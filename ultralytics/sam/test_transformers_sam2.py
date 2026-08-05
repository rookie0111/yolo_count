#!/usr/bin/env python3
"""测试 transformers SAM2 是否可用"""
try:
    from transformers import Sam2Model
    print("Transformers SAM2 可用")
except ImportError as e:
    print(f"Transformers SAM2 不可用: {e}")
except Exception as e:
    print(f"错误: {e}")
