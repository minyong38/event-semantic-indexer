#!/usr/bin/env python3
"""CLIP(ViT-B-32)을 이미지/텍스트 인코더 ONNX 두 개로 변환. 오프라인에서 한 번만 실행.

  python export_clip.py            → clip_image.onnx, clip_text.onnx, bpe_simple_vocab_16e6.txt.gz

C++ 인덱서는 clip_image.onnx만 쓴다 (질문 쪽 clip_text.onnx는 검색 단계에서 사용).
마지막에 torch 결과와 ONNX Runtime 결과가 같은지 코사인 유사도로 확인한다.
"""
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import open_clip
import torch

ARCH, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
out = Path(__file__).parent

model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
model.eval()
tokenizer = open_clip.get_tokenizer(ARCH)


class Img(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.encode_image(x)


class Txt(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, tokens):
        return self.m.encode_text(tokens)


img = torch.randn(1, 3, 224, 224)
tok = tokenizer(["a person wearing an orange shirt"])
with torch.no_grad():
    torch.onnx.export(Img(model).eval(), img, out / "clip_image.onnx", input_names=["image"],
                      output_names=["embedding"], dynamic_axes={"image": {0: "n"}, "embedding": {0: "n"}},
                      opset_version=18)
    torch.onnx.export(Txt(model).eval(), tok, out / "clip_text.onnx", input_names=["tokens"],
                      output_names=["text_emb"], dynamic_axes={"tokens": {0: "n"}, "text_emb": {0: "n"}},
                      opset_version=18)

# 토크나이저 어휘 파일 (C++ 토크나이저용으로 같이 보관)
vocab = Path(open_clip.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
if vocab.exists():
    shutil.copyfile(vocab, out / vocab.name)


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


with torch.no_grad():
    ti = model.encode_image(img).numpy()
    tt = model.encode_text(tok).numpy()
oi = ort.InferenceSession(str(out / "clip_image.onnx")).run(None, {"image": img.numpy()})[0]
ot = ort.InferenceSession(str(out / "clip_text.onnx")).run(None, {"tokens": tok.numpy()})[0]
print(f"[export] dim={oi.shape[-1]}  image cos={cos(ti, oi):.5f}  text cos={cos(tt, ot):.5f}  (1.0에 가까우면 OK)")
