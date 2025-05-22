class LLm:
    def __init__(self):
        print("LLm initialized")
    def predict(self, prompt):
        print(f"Predicting with prompt: {prompt}")
        return "Predicted response"
    


class tempTemplate:
    def __init__(self, template,input_variables):
        self.input_variables = input_variables
        self.template = template
    def format(self, **kwargs):
        return self.template.format(**kwargs)



class tempChain:
    def __init__(self,llm,prompt):
        self.llm=llm
        self.prompt=prompt
    def run(self, **kwargs):
        prompt=self.prompt.format(**kwargs)
        return self.llm.predict(prompt)
    
llm=LLm()

temp=tempTemplate(template="Summarize, {topic}", input_variables=["topic"])

chain=tempChain(llm=llm, prompt=temp)
chain.run(topic="LLM") # This should print "Predicting with prompt: Summarize, LLM" and return "Predicted response">>>