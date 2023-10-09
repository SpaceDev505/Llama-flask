from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from flask_cors import CORS
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)
model_name_or_path = "./models"
model_basename = "model"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

@app.route('/api', methods = ['POST', 'GET'])
def home():
    if request.method == "GET":
        return jsonify({"generated_text":"Hello"})
    if request.method == 'POST':
        res = request.json
        prompt = res['prompt']
        print("prompt",prompt)
        try:
            system_prompt = res['system_prompt']
            print(system_prompt)
        except:
            sytem_prompt = ""
        try:
            hidden_prompt = res['hidden_prompt']
            print(hidden_prompt)
        except:
            hidden_prompt = ""
        try:
            max_token = res['max_token']
            print(max_token)
        except:
            max_token = "512"
        try:
            character_name = res['character_name']
            print(character_name)
        except:
            character = ""
        try:
            character_background = res['character_background']
            print(character_background)
        except:
            background = ""
        
        prompt_template=f'''A chat between a curious user and an artificial intelligence girlfriend. 
        The name of girlfriend is {character_name} and the background is {character_background}
        The girlfrined gives helpful, detailed, and polite answers to the user's questions.
        {hidden_prompt}
        {system_prompt}   
        USER: {prompt}
        Grilfriend:
        '''
        input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # generated_text = tokenizer.decode(output[0])
        print("result", generated_text)
        generated_text = generated_text.replace(prompt_template, "")
        return jsonify({"generated_text":generated_text})
    return jsonify({"generated_text":"Hello"})
if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 5000)