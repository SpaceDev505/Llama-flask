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
        revision="main"
        )

@app.route('/api', methods = ['POST', 'GET'])
def home():
    if request.method == "GET":
        return jsonify({"generated_text":"Hello"})
    if request.method == 'POST':
        res = request.json
        prompt = res['prompt']
        print("1111prompt: ",prompt)
        try:
            system_prompt = res['system_prompt']
            print("2222system_prompt:  ", system_prompt)
        except:
            sytem_prompt = ""
        try:
            hidden_prompt = res['hidden_prompt']
            print("33333hidden_prompt:  ", hidden_prompt)
        except:
            hidden_prompt = ""
        try:
            max_token = res['max_token']
            print("44444max_token:  ", max_token)
        except:
            max_token = "512"
        try:
            character_name = res['character_name']
            print("55555character_name: ", character_name)
        except:
            character = ""
        try:
            character_background = res['character_background']
            print("66666character_background:  ", character_background)
        except:
            background = ""
        
        prompt_template=f'''
        {character_background}
        {hidden_prompt}
        USER: {prompt}
        ASSISTANT:
        '''
        # input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        # output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # generated_text = tokenizer.decode(output[0])
        pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    )
        generated_text = pipe(prompt_template)[0]['generated_text']
        print("result", generated_text)
        generated_text = generated_text.replace(prompt_template, "")
        return jsonify({"generated_text":generated_text})
    return jsonify({"generated_text":"Hello"})
if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 5000)