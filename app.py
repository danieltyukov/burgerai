from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

# load the fine-tuned GPT-2 model
model = GPT2LMHeadModel.from_pretrained("fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']

    # tokenize user input
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    # generate response using the fine-tuned model
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)