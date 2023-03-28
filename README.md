# Chat Bot
Trying to make a chatbot that has a better personality than me. 

In the meantime talk to john:  
`python john.py --size medium --device cpu --persona advisor`  
to get options do  
`python john.py --help`

TODO:
- [x] Pre prompting
- [ ] Optimized pre prompting
	- Is it possible to tune the pre prompts in a way that produces the desired output. (Prompt Manager? RLHF?)
- [ ] Connecting to the internet
	- [ ] Identify query and get wikipedia summary inside the pre prompt
	- [ ] Give web search results in prompt. The chatbot should also cite the sources.
- [ ] Making it much faster
	- [ ] Flash attention?
	- [ ] Pruning?
	- [ ] fp16
	- [ ] Quantization

