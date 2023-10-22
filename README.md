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

## 1. Ability to interpret and process images
- Multi-modal model functionality - adding [LLava](https://llava-vl.github.io) 
- Image interpreted by LLava v1 functionality and further text QA performed by a LLama finetune
- Currently requires user to download llama_cpp (waiting for Python bindings to come along for adding a dist/production version)

### Example usage:

![Screenshot 2023-10-22 at 7 54 02 pm](https://github.com/has-c/openassistant/assets/29789857/7c59cb67-cf1d-4963-944e-1dcc9de39313)
![example_response](https://github.com/has-c/openassistant/assets/29789857/9d31ba6b-4fc8-4a69-916b-b14d402e27f2)
