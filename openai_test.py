from openai_agent import OpenAIAgent

from openai import OpenAI
from datetime import datetime
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

import os

api_key = ''
base_url = 'https://api.deepseek.com'
model_name = 'deepseek-reasoner'

client = OpenAI(api_key=api_key, base_url=base_url)

tavily_api_key = ''
os.environ['TAVILY_API_KEY'] = tavily_api_key
search = TavilySearchResults()

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

duck_search = DuckDuckGoSearchRun()

pythonREPLTool = PythonREPLTool()

tools = [search, wikipedia, duck_search, pythonREPLTool]

system_prompt = f"""
你是一个得力的助手。
现在时间是：{datetime.now()}
"""
def main():
    history = []
    agent = OpenAIAgent(client, model_name, tools, system_prompt)
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
            response = agent.invoke(history, verbose=True)

            print(f"\n🤖 DeepSeek: {response['output']}")

            history.append({
                'role': 'assistant',
                'content': response['output']
            })
        except Exception as e:
            print(f"\n⚠️ 发生错误: {str(e)}")

if __name__ == '__main__':
    main()
