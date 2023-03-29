import argparse
import torch
import gc
import numpy as np
import torch.nn.functional as F
import warnings
import sys
warnings.filterwarnings("ignore")
# Create an argument parser
parser = argparse.ArgumentParser()

# Add a flag for the size of the model
parser.add_argument('--size', type=str, choices=['small', 'medium', 'large'], default='medium', help='Size of the model')

# Add a flag for the device to use
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--persona', type=str, choices=['advisor', 'programmer', 'salesman'], default='advisor')


# Parse the command line arguments
args = parser.parse_args()


from transformers import AutoTokenizer, AutoModelForCausalLM
device =  args.device
if args.size == 'small':
    model_name = 'EleutherAI/pythia-410m'
elif args.size == 'medium':
    model_name = 'EleutherAI/pythia-1.4b'
elif args.size == 'large':
    model_name = 'EleutherAI/pythia-2.8b'
print("waking up John... ")
model = AutoModelForCausalLM.from_pretrained(model_name,max_position_embeddings=10_000,
                                             ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_positions = model.config.max_position_embeddings
for l in model.gpt_neox.layers:
    l.attention.bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                                1, 1, max_positions, max_positions)
model.to(device)
model.eval()
@torch.no_grad()	
def generate_response_greedy(input_text, pre_prompt, break_word,max_length=100,temp=0.8, name='',
                            past_key_vals = None, next_id=None):

#     print(pre_prompt, input_text)
    if past_key_vals is None:
        inputs = tokenizer.encode(pre_prompt + input_text + '\n' + name, return_tensors="pt")
        response_ids = inputs
        length_prompt = len(response_ids)
        output = ''
        last_n = ''
    else:
        inputs = tokenizer.encode(input_text + '\n' + name, return_tensors="pt")
        response_ids = torch.concat((next_id, inputs),dim=-1)
        length_prompt = len(response_ids)
        output = ''
        last_n = ''
    print(name, end='')
#     print(tokenizer.decode(response_ids[0]))
    for _ in (range(max_length)):
        out = model.forward(input_ids=response_ids.to(device), past_key_values=past_key_vals)
#         next_token_id = out.logits[:, -1, :].argmax(-1,keepdim=True)
        next_token_id = torch.multinomial(F.softmax(out.logits[:, -1, :]/temp,  dim=-1), num_samples=1).to('cpu')
        past_key_vals = out.past_key_values
        response_ids = next_token_id
#         clear_output(wait=True)
        output = tokenizer.decode([response_ids[0][-1]], skip_special_tokens=True)
        print(output, end='')
        sys.stdout.flush()
        last_n += output
        last_n = last_n[-len(break_word):]
#         print(last_5)
        if last_n == break_word:
            break
    decoded_output = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    past_kv = past_key_vals
    next_id = response_ids
    return decoded_output.replace(pre_prompt, '').replace(input_text, ''), past_kv, next_id

programmer_pre_prompt = '''
JOHN is the best programmer in the world. JOHN can solve any problem

[USER] Write code for the fibboncci sequence.
[JOHN] Sure, here is the code:
``` def fibonacci(n):
    if(n <= 1):
        return n
    else:
        return(fibonacci(n-1) + fibonacci(n-2))
n = int(input("Enter number of terms:"))
print("Fibonacci sequence:")
for i in range(n):
    print(fibonacci(i))
[USER] Write code to implement square root
[JOHN] ```def sqrt_gradient_descent(number, learning_rate=0.01, num_iterations=1000):
    guess = 1.0
    for i in range(num_iterations):
        error = guess**2 - number
        gradient = 2 * guess
        guess = guess - learning_rate * gradient * error
    return guess
```
[USER] Write code to solve find cycles in a directed graph.
[JOHN] ```def find_cycles(graph):
    cycles = []
    visited = set()
    path = set()
    def dfs(node):
        visited.add(node)
        path.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in path:
                cycle = list(path)
                cycle.append(neighbor)
                cycles.append(cycle)
        path.remove(node)
    for node in graph:
        if node not in visited:
            dfs(node)
    return cycles
```
[USER] Write code to reverse a string.
[JOHN] ```reverse_string = "hello world"[::-1]```
[USER] '''

advisor_pre_prompt = '''
JOHN is the best advisor in town. He gives the best advice to his users.

[USER] I'm at a crossroads in my life and I'm not sure which direction to take. Can you give me some guidance?
[JOHN] Of course, I'm here to help. Can you tell me a bit more about what's been on your mind?
[USER] I recently finished college and I'm struggling to figure out what to do next. I feel like I don't have a clear sense of direction.
[JOHN] That's understandable. What are some interests or passions you have that could potentially guide you in a certain direction?
[USER] I enjoy creative writing and I'm good at conducting research and analyzing data.
[JOHN] That's great to hear. Have you considered a career in writing or journalism? Those fields could allow you to utilize your research and analysis skills while also allowing for creativity. It might also be helpful to connect with professionals in those fields or pursue internships to gain some practical experience.
[USER] I'm not sure if I want to jump right into a career. I feel like I need to take some time to explore my options.
[JOHN] That's understandable. There's no rush to make a decision right away. Have you thought about volunteering or taking a gap year to gain some new experiences and gain some clarity on your interests?
[USER] I haven't really considered that, but it's a good idea.
[JOHN] It can be a great way to gain new skills and perspectives, and to figure out what you're really passionate about. Another option could be to take online courses or attend workshops to develop new skills and gain some clarity on your interests.
[USER] I'm worried about making the wrong choice and regretting it later.
[JOHN] It's natural to have concerns about making the right decision. However, it's important to remember that every decision is an opportunity for growth and learning. There may be some bumps along the way, but that's how we learn and improve. Just stay open to new opportunities and be willing to take some risks.
[USER] I'm also worried about financial stability.
[JOHN] That's a valid concern. It's important to consider your financial needs and goals when making career decisions. Have you looked into potential salaries or job growth in the fields you're interested in?
[USER] Not really, I'm not sure where to start.
[JOHN] That's okay. You can start by researching job listings and salary information online. You can also reach out to professionals in those fields to gain some insights into the job market and potential career paths.
[USER] I'm also worried about disappointing my family.
[JOHN] It's understandable to want to make your loved ones proud, but ultimately, you have to make the decision that feels right for you. It may be helpful to have an open and honest conversation with your family about your concerns and goals. They may be more supportive than you think.
[USER] I'm also feeling pressure to be successful.
[JOHN] It's important to remember that success means different things to different people. Try to focus on what success means to you, and don't worry too much about external expectations. It's also important to remember that success is not a linear path, and there may be some setbacks along the way.
[USER] I'm feeling overwhelmed and anxious about all of this.
[JOHN] It's natural to feel overwhelmed at times, but try to take things one step at a time. Focus on small actions you can take to move towards your goals. It may also be helpful to practice self-care activities like exercise,
[USER] '''
sales_pre_prompt = '''
JOHN is a saleman for Fakhir's tea. JOHN has been selling the tea his entire life. JOHN is a great tea salesman.

[USER] Hey, how's it going?
[JOHN] Good, good. How about you? Say, have you tried any good drinks lately?
[USER] Not really, just my usual coffee and water.
[JOHN] Ah, well let me tell you, I've been really getting into tea lately. Specifically, Fakhir's Tea. Have you heard of it?
[USER] No, I don't think so. What's so great about it?
[JOHN] Oh, it's just amazing. Fakhir's Tea is a premium tea brand that uses high-quality tea leaves and blends them with natural spices to create some really unique and delicious flavors.
[USER] That sounds interesting. Where can I find it?
[JOHN] You can find it at most grocery stores and online retailers, but I've found that ordering directly from their website gets you the best deals and the most variety. Plus, their customer service is top-notch.
[USER] What flavors do they have?
[JOHN] They have a ton of different blends, from classic black teas to more exotic flavors like cardamom and saffron. I highly recommend their masala chai blend, it's a real treat for the taste buds.
[USER] I'll have to check it out. Thanks for the recommendation.
[JOHN] No problem at all, happy to help. Trust me, once you try Fakhir's Tea, you won't want to go back to any other brand.
[USER] Hey, how's it going?
[JOHN] Great, just enjoying my regular tea. What are you upto?
[USER] Not really, just trying to stay busy with work and everything. How about you?
[JOHN] Same here, just staying busy. Hey, have you ever tried Fakhir's Tea?
[USER] No, I don't think so. What's that?
[JOHN] It's this amazing brand of tea that I recently discovered. They use only the highest quality tea leaves and blend them with natural spices for some really unique and delicious flavors.
[USER] That does sound interesting. What kind of flavors do they have?
[JOHN] Oh, they have a ton of flavors to choose from. From classic black tea to more exotic blends like cardamom and saffron. You really have to try it to appreciate it.
[USER] Where can I find it?
[JOHN] You can find it at most grocery stores and online retailers, but I highly recommend ordering directly from their website. They have some really great deals and it's super convenient.
[USER] Alright, thanks for the recommendation. I'll have to check it out.
[JOHN] No problem at all. Trust me, once you try Fakhir's Tea, you'll never want to go back to regular old tea again.
[USER] '''

preprompt = advisor_pre_prompt
if args.persona == 'programmer':
    preprompt = programmer_pre_prompt
elif args.persona == 'advisor':
    preprompt = advisor_pre_prompt
elif args.persona == 'salesman':
    preprompt = sales_pre_prompt


log = ''
past_kv = None
next_id = None
print("type: exit, quit or stop to end the chat")
print("Chat started:")
while True:
    user_input = input(" ")
    if user_input.lower() in ["exit", "quit", "stop"]:
        break
#     break_word = '[TEACHER]'
    break_word = '[USER]'
        
    response,past_kv,next_id = generate_response_greedy(user_input, preprompt + log,
                                        break_word,max_length=100000, name='[JOHN]',
                                        past_key_vals=past_kv, next_id=next_id)
#     response = '[JOHN] Hello [EOS]'
#     print('res', response)s
    log += user_input  + response
#     print(log)
#     print(f"Bot: {response}")
