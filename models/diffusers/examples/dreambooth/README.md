# DreamBooth training example

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.
The `train_dreambooth.py` script shows how to implement the training procedure and adapt it for stable diffusion.


## Running locally with PyTorch
### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

### Dog toy example

Now let's get our dataset. Download images from [here](https://drive.google.com/drive/folders/1BO_dyz-p65qhBRRMRA4TbZ8qW4rB99JZ) and save them in a directory. This will be our training data.

And launch the training using

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
```

### Training with prior-preservation loss

Prior-preservation is used to avoid overfitting and language-drift. Refer to the paper to learn more about it. For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data.
According to the paper, it's recommended to generate `num_epochs * num_samples` images for prior-preservation. 200-300 works well for most cases. The `num_class_images` flag sets the number of images to generate with the class prompt. You can place existing images in `class_data_dir`, and the training script will generate any additional images so that `num_class_images` are present in `class_data_dir` during training time.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```


### Training on a 16GB GPU:

With the help of gradient checkpointing and the 8-bit optimizer from bitsandbytes it's possible to run train dreambooth on a 16GB GPU.

To install `bitandbytes` please refer to this [readme](https://github.com/TimDettmers/bitsandbytes#requirements--installation).

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

### Training on a 8 GB GPU:

By using [DeepSpeed](https://www.deepspeed.ai/) it's possible to offload some
tensors from VRAM to either CPU or NVME allowing to train with less VRAM.

DeepSpeed needs to be enabled with `accelerate config`. During configuration
answer yes to "Do you want to use DeepSpeed?". With DeepSpeed stage 2, fp16
mixed precision and offloading both parameters and optimizer state to cpu it's
possible to train on under 8 GB VRAM with a drawback of requiring significantly
more RAM (about 25 GB). See [documentation](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) for more DeepSpeed configuration options.

Changing the default Adam optimizer to DeepSpeed's special version of Adam
`deepspeed.ops.adam.DeepSpeedCPUAdam` gives a substantial speedup but enabling
it requires CUDA toolchain with the same version as pytorch. 8-bit optimizer
does not seem to be compatible with DeepSpeed at the moment.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch --mixed_precision="fp16" train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

### Fine-tune text encoder with the UNet.

The script also allows to fine-tune the `text_encoder` along with the `unet`. It's been observed experimentally that fine-tuning `text_encoder` gives much better results especially on faces. 
Pass the `--train_text_encoder` argument to the script to enable training `text_encoder`.

___Note: Training text encoder requires more memory, with this option the training won't fit on 16GB GPU. It needs at least 24GB VRAM.___

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

### Using DreamBooth for pipelines other than Stable Diffusion

The [AltDiffusion pipeline](https://huggingface.co/docs/diffusers/api/pipelines/alt_diffusion) also supports dreambooth fine-tuning. The process is the same as above, all you need to do is replace the `MODEL_NAME` like this:

```
export MODEL_NAME="CompVis/stable-diffusion-v1-4" --> export MODEL_NAME="BAAI/AltDiffusion-m9"
or
export MODEL_NAME="CompVis/stable-diffusion-v1-4" --> export MODEL_NAME="BAAI/AltDiffusion"
```

### Inference

Once you have trained a model using the above command, you can run inference simply using the `StableDiffusionPipeline`. Make sure to include the `identifier` (e.g. sks in above example) in your prompt.

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "path-to-your-trained-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("dog-bucket.png")
```

### Inference from a training checkpoint

You can also perform inference from one of the checkpoints saved during the training process, if you used the `--checkpointing_steps` argument. Please, refer to [the documentation](https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint) to see how to do it.

## Training with Flax/JAX

For faster training on TPUs and GPUs you can leverage the flax training example. Follow the instructions above to get the model and dataset before running the script.

____Note: The flax example don't yet support features like gradient checkpoint, gradient accumulation etc, so to use flax for faster training we will need >30GB cards.___


Before running the scripts, make sure to install the library's training dependencies:

```bash
pip install -U -r requirements_flax.txt
```


### Training without prior preservation loss

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="path-to-instance-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=400
```


### Training with prior preservation loss

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --num_class_images=200 \
  --max_train_steps=800
```


### Fine-tune text encoder with the UNet.

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=2e-6 \
  --num_class_images=200 \
  --max_train_steps=800
```

### Training with xformers:
You can enable memory efficient attention by [installing xFormers](https://github.com/facebookresearch/xformers#installing-xformers) and padding the `--enable_xformers_memory_efficient_attention` argument to the script. This is not available with the Flax/JAX implementation.

You can also use Dreambooth to train the specialized in-painting model. See [the script in the research folder for details](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/dreambooth_inpaint).

### Experimental results
You can refer to [this blog post](https://huggingface.co/blog/dreambooth) that discusses some of DreamBooth experiments in detail. Specifically, it recommends a set of DreamBooth-specific tips and tricks that we have found to work well for a variety of subjects. 
