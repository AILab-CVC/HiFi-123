# HiFi-123: Towards High-fidelity One Image to 3D Content Generation
<div align="center">
 <a href='https://arxiv.org/abs/2310.06744'><img src='https://img.shields.io/badge/arXiv-2310.06744-b31b1b.svg'></a> &nbsp;
 <a href='https://drexubery.github.io/HiFi-123/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;

<strong>ECCV 2024</strong>
</div>

## 📝 Changelog
- __[2024.7.12]__: Release the code for reference-guided novel view enhancement (RGNV), we will soon add Zero-1-to-3 support for the RGNV pipeline and release the code for Image-to-3D generation.
<br>

## 🔆 Introduction
Official implementation of HiFi-123: Towards High-fidelity One Image to 3D Content Generation, we are working hard on cleaning the code, please stay tuned.

## ⚙️ Setup for Reference-guided novel view enhancement (RGNV)
### Install Environment via Anaconda (Recommended)
```bash
cd ./HiFi-NVS
conda create -n rgnv python=3.9.7
conda activate rgnv

pip install -r requirements_rgnv.txt
```
Note that the diffusers version should be exactly the same with our requirements.

## 💫 Inference for Reference-guided novel view enhancement (RGNV)
### 1. Command line
1) Download the pre-trained depth estimation and matting model from [here](https://drive.google.com/file/d/1LEOmXAeylde0DSvUmfKeEt9_H1ENcdwD/view?usp=sharing), and put them in `./ptms`.
2) Download pretrained stable-diffusion-2-depth model via [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-depth).
2) Download pretrained stable-diffusion-x4-upscaler model via [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler).
3) Input the following commands in terminal, you can upload your source image (the high-quality reference image) and coarse image (the generated coarse novel view) then specify their path in the script. We use a specified background `./load/bg2.png` during processing, since the stable-diffusion-2-depth model is sensitive to pure background.
```bash
  sh run.sh
```

## ⚙️ Setup for Image-to-3D generation
Our code will be integrated into threestudio to combine with a variety of models implemented by threestudio, making our method a generalized tool for enhancing texture quality in Image-to-3D generation.

## 🤗 Acknowledgements
Many thanks to the projects [threestudio](https://github.com/threestudio-project/threestudio), [MasaCtrl](https://github.com/TencentARC/MasaCtrl).

## 🤝 Citation
```bib
  @article{yu2023hifi,
      title={Hifi-123: Towards high-fidelity one image to 3d content generation},
      author={Yu, Wangbo and Yuan, Li and Cao, Yan-Pei and Gao, Xiangjun and Li, Xiaoyu 
          and Hu, Wenbo and Quan, Long and Shan, Ying and Tian, Yonghong},
      journal={arXiv preprint arXiv:2310.06744},
      year={2023}
      }
  }
```