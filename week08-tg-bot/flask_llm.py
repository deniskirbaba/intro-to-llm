from crypt import methods
from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

app = Flask(__name__)
system = 'U are AI assisntant'
llm = LLM(model = "Qwen/Qwen2.5-Math-1.5B-Instruct", tokenizer= "Qwen/Qwen2.5-Math-1.5B-Instruct")
sampling_params = SamplingParams(temperature = 0.7, max_tokens=512)
@app.route('/genetate_text', methods=['POST'])
def genetate_text():
    data = request.json
    promt = data['promt']
    convetatation = [{"role": "system", "content":system},
                     {'role': "user", "content": promt}]

    outputs = llm.chat(convetatation, sampling_params)
    for output in outputs:
        genetated_text = output.outputs[0].text
    return jsonify({"text": genetated_text})



if __name__=="__main__":
    app.run(port=8000)
