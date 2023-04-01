[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html):  
[LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) finetuned on instruction dataset of size 52k.
The instruction dataset is created using [self-instruct](https://arxiv.org/pdf/2212.10560.pdf) and 175 self instruct type generations from openai text-davinci-003.
Also they use [watermarking](https://arxiv.org/pdf/2301.10226.pdf) to be able to detect the model's outputs.

[GPT4All](https://github.com/nomic-ai/gpt4all):  
trained LLaMA 7B on 437,605 prompt-generation pairs for 4 epochs using [low rank rank adaptation](https://arxiv.org/abs/2106.09685).
[Report](https://s3.amazonaws.com/static.nomic.ai/gpt4all2023_GPT4All_Technical_Report.pdf) 

[RLHF](https://huggingface.co/blog/rlhf):  
![rlhf](rlhf.png)


Ideas:  
Prompts and Memorising or Retrieval Transformers.  
LSTM Transformer variants, and storing pre prompt in hidden state


