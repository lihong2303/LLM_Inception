
<h2 align="center">
 The Labyrinth of Links: Navigating the Associative Maze of Multi-modal LLMs
</h2>

<h3 align="center">
 ICLR 2025
</h3>
&nbsp;

<div align="center" margin-bottom="6em">
<a target="_blank" href="https://github.com/lihong2303">Hong Li</a>,
<a target="_blank" href="https://github.com/andylinx">Nanxi Li</a>,
<a target="_blank" href="https://github.com/ccmoony">Yuanjie Chen</a>,
<a target="_blank" href="https://github.com/Peebinens">Jianbin Zhu</a>,
<a target="_blank" href="https://github.com/ggsdeath">Qinlu Guo</a>,
<a target="_blank" href="https://www.mvig.org/">Cewu Lu</a>,
<a target="_blank" href="https://dirtyharrylyl.github.io/">Yong-Lu Li</a>
</div>
&nbsp;


<div align="center">
    <a href="https://arxiv.org/abs/2410.01417" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://mvig-rhos.com/llm_inception" target="_blank">
    <img src="https://img.shields.io/badge/Page-RHOS-green" alt="Project Page"></a>
    <!-- <a href="https://youtu.be/mlnjz4eSjB4?si=NN9z7TpkTPgBAzBw" target="_blank">
    <img src="https://img.shields.io/badge/Video-YouTube-9966ff" alt="Video"></a> -->
    <a href="https://drive.google.com/file/d/1OuYnLXYoRp6i6jdID3ntdNg9A0rgFcoc/view?usp=drive_link" target="_blank">
    <img src="https://img.shields.io/badge/Data-Google_Drive-green" alt="Data"></a>
    <a href="https://huggingface.co/spaces/LIHONG2303/LLM_Inception" target="_blank">
    <img src="https://img.shields.io/badge/Demo-Huggingface-orange" alt="Model"></a>
</div>
&nbsp;

<div align="left">
<img src="./Images/teaser_figure.png" width="99%" alt="LLM_Inception Teaser">
</div>


## Get Started

1. Clone the repository
```shell
git clone https://github.com/lihong2303/LLM_Inception.git
cd LLM_Inception
```

2. Create `conda` environment and install dependencies.
```shell
conda create -n llm_inception python=3.10
conda activate llm_inception

# install PyTorch, take our version for example
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 cudatoolkit=11.8 -c pytorch

pip install -r requirements.txt
```

3. Eval Single-step Association with:
```shell
python eval_singlestep.py \
    --data_root Data \
    --data_type pangea_data \
    --model_type "mplug3" \
    --prompt_type "task_instruction_nomem" \
    --attr_constraint "cut" \
    --expt_dir "logs" \
    --few_shot_num 3
```

4. Eval Multi-step Association with:
```shell
python eval_multistep.py \
    --data_root Data \
    --data_type ocl_attr_data \
    --model_type "llava-onevision" \
    --prompt_type "task_instruction" \
    --attr_constraint "furry,metal" \
    --expt_dir "logs" \
    --few_shot_num 3
```



## Dataset

We reconstructed two association datasets based on adjective and verb concepts, for details on how to download the dataset and the structure please refer to [Data](./data/Data.md).


## Reference
```bibtex
@article{li2024labyrinth,
  title={The Labyrinth of Links: Navigating the Associative Maze of Multi-modal LLMs},
  author={Li, Hong and Li, Nanxi and Chen, Yuanjie and Zhu, Jianbin and Guo, Qinlu and Lu, Cewu and Li, Yong-Lu},
  journal={arXiv preprint arXiv:2410.01417},
  year={2024}
}
```

## Acknowledgement

We extend our gratitude to the prior outstanding work in object concept learning, particularly [OCL](https://github.com/silicx/ObjectConceptLearning) and [Pangea](https://github.com/DirtyHarryLYL/Sandwich), which serve as the foundation for our research.

