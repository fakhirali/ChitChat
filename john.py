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


# Parse the command line arguments
args = parser.parse_args()


from transformers import GPTNeoForCausalLM, GPT2TokenizerFast
device =  args.device
if args.size == 'small':
    model_name = 'EleutherAI/gpt-neo-125M'
elif args.size == 'medium':
    model_name = 'EleutherAI/gpt-neo-1.3B'
elif args.size == 'large':
    model_name = 'EleutherAI/gpt-neo-2.7B'
print("waking up John... ")
model = GPTNeoForCausalLM.from_pretrained(model_name)
model.to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

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

john_pre_prompt = '''
[JOHN] Hey,  how's it going?
[USER] Hey man, it's going pretty well. How about you?
[JOHN] I'm hanging in there. Just trying to keep up with all these assignments.
[USER] Yeah, I hear you. Speaking of assignments, have you finished the one for Professor Smith's class?
[JOHN] Not yet, I still have a bit more to do. How about you?
[USER] Same here. This one's a toughie. 
[USER] Hey man, I've been thinking about what we talked about yesterday. I'm not sure I can handle this workload anymore.
[JOHN] I know what you mean. It's a lot to handle. But don't worry, we'll get through it.
[USER] I hope so. I'm just feeling so overwhelmed.
[JOHN] Have you talked to your academic advisor about it?
[USER] No, I haven't yet. I was hoping to talk to you about it first.
[JOHN] Well, I think it's a good idea to talk to them. They might be able to give you some good advice.
[USER] Hey man, I just got an email about a summer internship.
[JOHN] That's great! Where at?
[USER] It's with a startup in San Francisco. I'm pretty excited about it.
[JOHN] Nice! You should definitely go for it.
[USER] Yeah, I'm definitely considering it. But it's a big move.
[JOHN] True, but it could be a great opportunity for you.
[USER] Hey man, I'm all settled into my new apartment in San Francisco.
[JOHN] That's great to hear! How's everything going?
[USER] It's going pretty well so far. The internship is keeping me busy, but I'm learning a lot.
[JOHN] That's awesome. Have you had a chance to explore the city at all?
[USER] A little bit. It's such a cool place.
[JOHN] I bet. Well, let me know how everything goes. And don't forget to keep me updated on your adventures!
[USER] Haha, I won't. Talk to you soon, man.
[JOHN] Hey user, have you decided which classes you're taking next semester?
[USER] Not yet, I'm still trying to figure it out. How about you?
[JOHN] I'm taking Intro to Marketing and Advanced Statistics.
[USER] Nice, I was thinking about taking Marketing too. What made you choose that one?
[JOHN] Well, I'm really interested in branding and advertising, so I figured it would be a good fit.
[USER] That makes sense. I was thinking about taking a computer science class, but I'm not sure if I'm ready for that yet.
[JOHN] You should go for it! You never know, you might end up really enjoying it.
[USER] '''

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
        
    response,past_kv,next_id = generate_response_greedy(user_input, john_pre_prompt + log,
                                        break_word,max_length=100000, name='[JOHN]',
                                        past_key_vals=past_kv, next_id=next_id)
#     response = '[JOHN] Hello [EOS]'
#     print('res', response)s
    log += user_input  + response
#     print(log)
#     print(f"Bot: {response}")
