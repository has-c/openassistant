# OpenAssistant Chat
- Can be used with any LLama GGUF model variants.
- Best one found so far: [Mistral-7B-OpenOrca-GGUF](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF)

Pre-requisite:
- User will require Python3.11

Dist download link: [Download link](https://drive.google.com/file/d/1lgGzSyj4_lE5f3nebiHaMa8d310HaCqT/view?usp=sharing)

Change the path in `run.sh` to your local path.

![image](https://github.com/has-c/zima/assets/29789857/b2c1b893-d50d-4683-880a-ab7b49117b14)

# Compile the app from source

```bash
python3 setup.py py2app
```

# Beta Features

1. Ability to interpret and process images
- Multi-modal model functionality - adding [LLava](https://llava-vl.github.io) 
- Image interpreted by LLava v1 functionality and further text QA performed by a LLama finetune
