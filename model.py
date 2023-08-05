import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sqlite3

class RewardModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RewardModel, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.out(x)

def fine_tune_model():
    # load the pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # load the ingredients dataset
    conn = sqlite3.connect('recipes.db')
    c = conn.cursor()
    c.execute("SELECT ingredients FROM recipes")
    ingredients_list = c.fetchall()
    conn.close()

    # prepare for fine-tuning
    ingredients_texts = [row[0] for row in ingredients_list]
    training_texts = "\n".join(ingredients_texts)

    # tokenize the dataset
    input_ids = tokenizer.encode(training_texts, return_tensors='pt')

    # fine-tune the model using RL
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 8
    temperature = 1.0  # (higher value -> more random text)

    # create a reward model
    reward_model = RewardModel(input_size=768, hidden_size=128)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_reward = 0.0
        total_steps = 0

        for i in range(0, input_ids.size(0) - batch_size, batch_size):
            inputs = input_ids[i:i+batch_size, :]
            targets = input_ids[i:i+batch_size, 1:]

            # generate text from the model
            outputs = model.generate(inputs, max_length=50, num_return_sequences=1, temperature=temperature)

            # calculate rewards using criteria:
            # (number of unique words in the generated text)
            rewards = torch.tensor([len(set(output.tolist()[0])) for output in outputs])

            # calculate the log-likelihood of the generated text
            log_probs = model(inputs).log_softmax(dim=-1)
            log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

            # calculate the policy gradient loss
            loss = -torch.mean(log_probs * rewards)
            
            # optimize the reward model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += rewards.sum().item()
            total_steps += rewards.size(0)

        avg_reward = total_reward / total_steps if total_steps != 0 else 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg. Reward: {avg_reward}")

    model.save_pretrained("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")

if __name__ == '__main__':
    fine_tune_model()
