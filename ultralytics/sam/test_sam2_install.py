#!/usr/bin/env python3
"""测试 SAM2 是否可用"""
try:
    from sam2.build_sam import build_sam2
    print("SAM2 可用")
except ImportError as e:
    print(f"SAM2 不可用: {e}")
except Exception as e:
    print(f"错误: {e}")
