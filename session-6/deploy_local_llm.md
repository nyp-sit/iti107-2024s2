# Deploy Custom LLM Model locally

In this exercise, you will learn how to:

1. convert the trained model to GGUF format
2. optimize the model for fast inference through quantization
3. deploy the model locally

## Download the Model

We will first download the model from Hugging face using huggingface-cli.


```bash
pip install "huggingface_hub[cli]"
```

We download our model hosted on huggingface, to a local directory called Llama-3.2-1B-chat-doctor. Replace the model repo ID with your own repo ID.

In your home directory, (e.g. /home/ubuntu), download your model: 

```bash
huggingface-cli download --local-dir Llama-3.2-1B-chat-doctor khengkok/Llama-3.2-1B-chat-doctor
```

## Convert the Model to GGUF

GGUF is a binary format that is optimized for quick loading and saving of models, making it highly efficient for inference purposes.

`llama.cpp` is an open source software library that performs inference on various large language models such as Llama. It also contains a set of utility to convert model to GGUF and also perform quantization using different quantization schemes.

### Setup llama.cpp

We will clone the git repo of llamma.cpp and use the python script to convert the model.
we will also compile the llama.cpp to get the various utility module, one of which is to perform quantization.

To avoid any version conflict with your existing installation of pytorch, and other deep learning libraries, it is strongly suggested that you create a new conda environment to install llama.cpp
You can create the conda environment by the following steps:

```bash
conda create -n llamacpp python=3.10
conda activate llamacpp 
```

Thereafter you do the following: 

```bash
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```

We will now run the script `convert_hf_to_gguf` found inside the `llama.cpp` directory to conver the model that we downloaded into gguf format.


```bash
cd llama.cpp
python convert_hf_to_gguf.py --outfile ~/Llama-3.2-1B-chat-doctor.gguf ~/Llama-3.2-1B-chat-doctor
```

## Quantize the model

We will quantize the model using Q4_K_M schema.  For more information about the different quantization schemas, you can refer to this [document](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md).
We can use the llama-quantize utility to do this. However, you will need to build the llama.cpp to get this utility, by running the Makefile inside llama.cpp. The build will take quite a while.


### build the llama.cpp

```bash
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

Now we run the `llama-quantize`, selecting Q4_K_M as the quantization scheme, to quantize our gguf. We will name our quantized model as `Llama-3.2-1B-chat-doctor-Q4_K_M.gguf` to differentiate from the original model.

In the `llama.cpp` directory, 
```bash
./build/bin/llama-quantize ~/Llama-3.2-1B-chat-doctor.gguf  ~/Llama-3.2-1B-chat-doctor-Q4_K_M.gguf Q4_K_M
```

## Upload the model to Hugging Face

You will need your HF access token to login to the Hugging Face in order to upload your model.
In your jupyter notebook, use the following codes to upload model.

```python
from huggingface_hub import notebook_login

notebook_login()
```

Here, we choose to upload our file to our repo `khengkok/Llama-3.2-1b-chat-doctor` which is hosting the original model. We use huggingface api to upload the file.


```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="/content/Llama-3.2-1B-chat-doctor-Q4_K_M.gguf",
    path_in_repo="Llama-3.2-1B-chat-doctor-Q4_K_M.gguf",
    repo_id="khengkok/Llama-3.2-1b-chat-doctor",
    repo_type="model",
)

```

## Run the model using Ollama   

Ollama is a very popular platform to run your local LLM. It exposes OpenAI compatible API, so you can easily migrate your existing applications built for OpenAI easily to Ollama-hosted models


### Installation

Follow the instructions [here](https://github.com/ollama/ollama/tree/main) for installation for MacOS, Linux and Windows.

If you are using the GPU instance provided (a Ubuntu instance), use the following command to install Ollama

```
cd 
curl -fsSL https://ollama.com/install.sh | sh
```


### Download your model

1. Download your gguf model into a folder of your local PC, say /home/ubuntu/models

2. Create a Modelfile in the same folder with the following content:

```
FROM ./Llama-3.2-1B-chat-doctor-Q4_K_M.gguf
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token"
```

You can also set the temperature, top-K, etc parameters in the Modelfile. You can refer to this [link](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)  for Modelfile documentation

3. Run your model using the following commands:

```
ollama create chatdoctor -f ./Modelfile
ollama run chatdoctor
```

and you will see a prompt, that let you interact with your model:

```
>>Hi doctor, I have stomach pain.
```

You can stop the model by typing:

```
ollama stop chatdoctor
```

4. If you want to run the model as a server instead of the shell:

```
./ollama serve

and in a separate shell, run a model:

./ollama run chatdoctor

```

5. Call API

You can interact with the model using OpenAI Rest API:

```
curl http://localhost:11434/v1/chat/completions \
-d '{
  "model": "chatdoctor",
  "messages": [
    { "role": "user", "content": "Hi, doctor I have stomach pain." }
  ]
}'
```

You can use OpenAI python library to call the endpoint too, e.g.

```
from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

completion = client.chat.completions.create(
  model="chatdoctor",
  messages=[
    {"role": "user", "content": "Hi doctor, I have stomach ache."}
  ]
)

print(completion.choices[0].message)
```



### Other GUI application to run the model

There are quite a number of GUI application you can use to run your model. These GUI application presents a chat interface similar to what is available from ChatGPT.  Some examples are:
1. [Jan](https://jan.ai/)
2. [LM Studio](https://lmstudio.ai/)
3. [GPT4All](https://www.nomic.ai/gpt4all)
