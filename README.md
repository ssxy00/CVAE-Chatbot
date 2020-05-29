# CVAE-Chatbot
This project combines [NeuralDialog-CVAE](https://github.com/snakeztc/NeuralDialog-CVAE) proposed in [(Zhao et al., 2017)](https://arxiv.org/abs/1703.10960) and GPT2 pretrained model released by [Hugginface](https://huggingface.co/transformers/pretrained_models.html) to implement an open-domain chatbot. 

## References
+ [(Zhao et al., 2017)](https://arxiv.org/abs/1703.10960) for cvae architecture
+ [(Li et al., 2020)](https://arxiv.org/abs/2004.04092) for combination method
+ [transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai) for baseline
+ [(Ippolito et al., 2019)](https://arxiv.org/abs/1906.06362) for diversity evaluation

## Environment
+ python == 3.6.8
+ pytorch==1.2.0
+ transformers==2.5.1
+ jsonlines
+ tqdm

To install the requried packages with `conda`, you can run the following script:
1. clone the repo
```
https://github.com/ssxy00/CVAE-Chatbot
cd CVAE-Chatbot
```
2. create virtual environment(optional)
```
conda create -n cvae_chatbot python==3.6.8
conda activate cvae_chatbot
```
3. install packages
```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0
python -m pip install transformers==2.5.1
python -m pip install -r requirements.txt
```

## Usage
### prepare data
The dataset used in this project is [PersonaChat Dataset](https://arxiv.org/abs/1801.07243) provided in [Convai2](http://convai.io/). `train_self_original_no_cands` is used to train the model and `valid_self_original_no_cands.txt` is used to evaluate.

The easiest way to prepare data is to download my processed dataset [here](https://drive.google.com/file/d/1qYgbmryXFbqGZ8TEq5kiAf-PxCynNU2z/view?usp=sharing). After that, pleast unzip files into `./datasets`.

PersonaChat Dataset also provides other datasets, such as `train_self_revised_no_cands.txt` of which persona is revised. If you want to use these datasets, you need to:
+ download ConvAI2 dataset
The dataset is available in [ParlAI](https://github.com/facebookresearch/ParlAI), so first install ParlAI:
```
git clone https://github.com/facebookresearch/ParlAI
cd ParlAI
# ParlAI now requires PyTorch==1.4, so revert to history vesion
git reset --hard 1e905fec8ef4876a07305f19c3bbae633e8b33af
# then download data
python examples/display_data.py --task convai2 --datatype train
```
After running this script, a folder `ConvAI2` containing dataset files will be created in `ParlAI/data/`.

+ process data
Then you can process dataset with following script:
```
# Run this script in the root directory of this repo
export PYTHONPATH=./
RAW_DATA=/path/to/raw/dataset/file
CACHE_DATA=/path/to/save/processed/dataset/file
GPT2_VOCAB_PATH=/path/to/gpt2/tokenizer/files
python preprocess_data.py --raw_data $RAW_DATA --cache_data $CACHE_DATA --gpt2_vocab_path $GPT2_VOCAB_PATH
```


### prepare pretrained gpt2 model
The project uses GPT2 pretrained model provided by [Huggingface](https://huggingface.co/transformers/index.html), so you need to download it in advance. You can run the following script to download `gpt2` model:
```
# Run this script in the root directory of this repo
mkdir -p gpt2/model
mkdir -p gpt2/tokenizer

cd gpt2/model
# download model files
wget https://cdn.huggingface.co/gpt2-pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json
# rename files
mv gpt2-pytorch_model.bin pytorch_model.bin
mv gpt2-config.json config.json

cd ../tokenizer
# download tokenizer files
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
# rename files
mv gpt2-vocab.json vocab.json
mv gpt2-merges.txt merge.txt
```

### train
The training script can only be used in single GPU setting:
```
# Run this script in the root directory of this repo
MODIR_DIR=/path/to/save/model/checkpoints
python train.py --save_model_dir $MODEL_DIR
```
You can set the following training arguments:

| Argument | Type | Default vale | Description |
| ---- | ---- | ---- | ---- |
|gpt2_model_dir | `str` | `"./gpt2/model"` | path to GPT2 pretrained model parameters |
|gpt2_vocab_path | `str` | `"./gpt2/tokenizer"` | path to GPT2 tokenizer vocab file |
|train_dataset | `str` | `."/datasets/train_self_original_no_cands.cache"` | cache train_dataset path |
|valid_dataset | `str` | `./datasets/valid_self_original_no_cands.cache` | cache valid_dataset path |
|max_seq_len | `int` | `60` | max sequence length fed into GPT2 | 
|max_history | `int` | `2` | max number of historical conversation turns to use |
|max_context_len| `int` | `100` | type=int | max context sequence length for sentence embedding |
|max_persona_len | `int`| `70` | max persona sequence length for sentence embedding |
|max_response_len | `int` | `30`|max response sequence length for sentence embedding|
|seed|`int`|`0`|random seed|
|device | `str` | `'cuda' if torch.cuda.is_available() else 'cpu'` | "cpu" or "cuda" |
|z_dim | `int` | `200` |latent hidden state dim (z) |
|n_epochs| `int` | `1` | number of training epochs |
|batch_size | `int` | `2` | batch size for training |
|lr | `float` |  `6.25e-5` |learning rate |
|gradient_accumulate_steps| `int` | `1` | accumulate gradient on several steps |
|clip_grad| `float` | `1.0` | clip gradient threshold |
|save_model_dir| `str` | default="./checkpoints" |path to save model checkpoints |
|save_interval| `int` | `1` | save checkpoint every N epochs |
|model_type | type=str | "compressed_cvae" | "decoder", "cvae_memory", "cvae_embedding", "compressed_decoder" or "compressed_cvae", see [here](https://github.com/ssxy00/CVAE-Chatbot/blob/master/docs/models.md) for detailed description|
|bow | `bool` | `False` | add bow loss or not, refer to [(Zhao et al., 2017)](https://arxiv.org/abs/1703.10960) for detailed explanation|
|kl_coef | `float` | `1.0` | kl loss coefficient |
|bow_coef | `float` | `1.0` | bow loss coef coefficient |




### evaluate
After training the model, you can run the following script to evaluate the model. This script will output the prediction results to the file you specified. For each test sample, you will get a json-format result:
```
{
"persona": persona_string, 
"context: context_string, 
"golden_response": target_response_string, 
"predict_responses": [candidate_1_string, ..., candidate_n_string], 
"predict_f1s": [candidate_1_f1, ..., candidate_n_f1]
}
```
When the evaluation ends, average ppl and average f1(max f1 among candidates of each sample) will be output to terminal.

You can set `n_outputs` to modify the number of candidates to predict. For decoder-type model, the model will do beam search (beam_size=`n_outputs`) and return all beams. For cvae-type model, the model will sample z `n_output` times and do greedy search.
```
# Run this script in the root directory of this repo
export PYTHONPATH=./
CHECKPOINT_PATH=/path/to/model/checkpoint
MODEL_TYPE={type of model trained}
OUTPUT_PATH=/path/to/save/predicted/results
N_OUTPUTS={number of candidates to predict}
python evaluation/predict.py \
--checkpoint_path $CHECKPOINT_PATH \
--model_type $MODEL_TYPE \
--output_path #OUTPUT_PATH \
--n_outputs $N_OUTPUTS
```
You can set the following evaluation arguments:

| Argument | Type | Default vale | Description |
| ---- | ---- | ---- | ---- |
|gpt2_model_dir | `str` | `"./gpt2/model"` | path to GPT2 pretrained model parameters |
|gpt2_vocab_path | `str` | `"./gpt2/tokenizer"` | path to GPT2 tokenizer vocab file |
|valid_dataset | `str` | `./datasets/valid_self_original_no_cands.cache` | cache valid_dataset path |
|output_path|`str`| `./result.jsonl` |path to output prediction results |
|batch_size | `int` | `2` | batch size for evaluation |
|max_seq_len | `int` | `60` | max sequence length fed into GPT2 | 
|max_history | `int` | `2` | max number of historical conversation turns to use |
|max_context_len| `int` | `100` | max context sequence length for sentence embedding |
|max_persona_len | `int`| `70` | max persona sequence length for sentence embedding |
|max_response_len | `int` | `30`|max response sequence length for sentence embedding|
|max_predict_len|`int`|`32`|max predicted response sequence length|
|n_outputs|`int`|`3`|how many candidates to generate|
|seed|`int`|`0`|random seed|
|device | `str` | `'cuda' if torch.cuda.is_available() else 'cpu'` | "cpu" or "cuda" |
|z_dim | `int` | `200` |latent hidden state dim (z) |
|checkpoint_path| `str` | default="" |path to load model checkpoint |
|model_type | type=str | "compressed_cvae" | "decoder", "cvae_memory", "cvae_embedding", "compressed_decoder" or "compressed_cvae" |

After getting the prediction results, you can run the following script to get diversity metrics:
```
OUTPUT_PATH=/path/of/generated/results
python evaluation/evaluate_diversity.py --eval_file $OUTPUT_PATH
```
This script will output `distinct-1`, `distinct-2` and `entropy-4`.

