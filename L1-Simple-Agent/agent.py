import re

from agents_in_langgraph.utils import open_ai
from agents_in_langgraph.utils.util import known_actions, average_dog_weight

class Agent:
    def __init__(self, system: str="") -> None:
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})
        self.client = open_ai.new_open_ai()
        # python regular expression to selection action
        self.known_actions = known_actions
        self.action_re = re.compile('^Action: (\w+): (.*)$')
    
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
    
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def execute(self, model: str="qwen2.5-it:3b", temperature: float=0.0):
        completion = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=self.messages,
        )

        return completion.choices[0].message.content
    
    def query(self, question: str, max_turns: int=5):
        i = 0
        next_prompt = question
        while i < max_turns:
            i += 1
            result = self(next_prompt)
            print(result)
            actions = [
                self.action_re.match(a) 
                for a in result.split('\n') 
                if self.action_re.match(a)
            ]
            if actions:
                # There is an action to run
                action, action_input = actions[0].groups()
                if action not in self.known_actions:
                    raise Exception("Unknown action: {}: {}".format(action, action_input))
                print(" -- running {} {}".format(action, action_input))
                observation = self.known_actions[action](action_input)
                print("Observation:", observation)
                next_prompt = "Observation: {}".format(observation)
            else:
                return

if __name__ == '__main__':
    print(f"example - completions with prompts and function call results")
    prompt = """
        You run in a loop of Thought, Action, PAUSE, Observation.
        At the end of the loop you output an Answer
        Use Thought to describe your thoughts about the question you have been asked.
        Use Action to run one of the actions available to you - then return PAUSE.
        Observation will be the result of running those actions.

        Your available actions are:

        calculate:
        e.g. calculate: 4 * 7 / 3
        Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

        average_dog_weight:
        e.g. average_dog_weight: Collie
        returns average weight of a dog when given the breed

        Example session:

        Question: How much does a Bulldog weigh?
        Thought: I should look the dogs weight using average_dog_weight
        Action: average_dog_weight: Bulldog
        PAUSE

        You will be called again with this:

        Observation: A Bulldog weights 51 lbs

        You then output:

        Answer: A bulldog weights 51 lbs
        """.strip()
    agent = Agent(prompt)

    question = "How much does a toy poodle weigh?"
    result = agent(question)
    print(f"result: \n{result}")
    print(f"=================================================================================================")

    result = average_dog_weight("Toy Poodle")
    next_prompt = "Observation: {}".format(result)
    result = agent(next_prompt)
    print(f"result: \n{result}")
    print(f"agent messages:\n {agent.messages}")
    print(f"=================================================================================================")

    # another example
    print(f"another example - multiple calls on complemetion tasks")
    abot = Agent(prompt)
    question = """I have 2 dogs, a border collie and a scottish terrier. \
    What is their combined weight"""
    abot(question)
    next_prompt = "Observation: {}".format(average_dog_weight("Border Collie"))
    print(next_prompt)
    abot(next_prompt)
    next_prompt = "Observation: {}".format(average_dog_weight("Scottish Terrier"))
    print(next_prompt)
    abot(next_prompt)
    next_prompt = "Observation: {}".format(eval("37 + 20"))
    print(next_prompt)
    abot(next_prompt)
    print(f"=================================================================================================")
    print(f"third example - using a query loop wrapped function")
    abot = Agent(prompt)
    question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
    abot.query(question)
