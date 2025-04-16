from agent import Agent

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

import os

api_key = ''
base_url = ''
model_name = 'deepseek-ai/DeepSeek-R1'

model = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    temperature=0.6,
    max_tokens=16000
)

tavily_api_key = ''
os.environ['TAVILY_API_KEY'] = tavily_api_key
search = TavilySearchResults()

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

duck_search = DuckDuckGoSearchRun()

tools = [search, wikipedia, duck_search]

system_prompt = """
你是一个得力的助手。
"""
def main():
    history = []
    agent = Agent(model, tools, system_prompt)
    print("\nDeepSeek R1 Agent 客户端已启动！输入 'quit' 退出")

    while True:
        try:
            query = input("\nUser: ").strip()
            if query.lower() == 'quit':
                break

            history.append({
                'role': 'user',
                'content': query
            })

            history = history[-10:]
            response = agent.invoke({'history': history}, verbose=True)

            print(f"\n🤖 DeepSeek: {response['output']}")

            history.append({
                'role': 'ai',
                'content': response['output']
            })
        except Exception as e:
            print(f"\n⚠️ 发生错误: {str(e)}")

if __name__ == '__main__':
    main()
