from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.render import render_text_description_and_args
import re

from agent_prompt import tool_declare_prompt


class Agent:
    def __init__(self, model: ChatOpenAI, tools: Sequence[BaseTool], system_prompt: str):
        """
        初始化 Agent 类。

        :param model: ChatOpenAI 模型实例。
        :param tools: 工具列表。
        :param system_prompt: 系统提示。
        """
        self.model = model
        self.tools = tools

        # 初始化工具映射
        self.tools_map = {}
        self.init_tools_map(tools)

        # 生成工具声明
        self.tool_descriptions = []
        self.tool_names = []
        self.generate_tools_declaration(tools)

        # 初始化输出解析器
        self.parser = StrOutputParser()

        # 初始化提示模板
        self.prompt = ChatPromptTemplate([
            ('system', system_prompt),
            ('system', tool_declare_prompt),
            MessagesPlaceholder(variable_name='history')
        ])

        # 构建链式调用
        self.chain = self.prompt | self.model | self.parser | self.divide_thought

    def invoke(self, params: dict, max_iteration: int = 5, verbose: bool = False) -> dict:
        """
        调用 Agent。

        :param params: 输入参数。
        :param max_iteration: 最大迭代次数，默认为 3。
        :param verbose: 调试模式，默认关闭。
        :return: 输出结果。
        """
        # 添加工具描述和工具名称到参数中
        params['tools'] = self.tool_descriptions
        params['tool_names'] = self.tool_names

        # 初始化历史记录
        if 'history' not in params:
            params['history'] = []

        for iteration in range(max_iteration):
            try:
                # 调用链式调用
                response = self.chain.invoke(params)
            except Exception as e:
                print(f"\033[91m链式调用失败：{e}\033[0m")
                return {}

            # 添加工具调用带历史记录
            params['history'].append({
                'role': 'ai',
                'content': response['output']
            })

            # 打印调试日志
            if verbose:
                print('\033[94mThought: ', response['thought'] , '\nOutput: ', response['output'] , '\033[0m')

            # 解析响应
            parsed_response = self.parse_response(response['output'])

            # 如果有工具调用
            if parsed_response['tool_call']:
                try:
                    # 调用工具
                    call_response = self.call_function(parsed_response['tool_call']['tool_name'],
                                                       parsed_response['tool_call']['parameters'],
                                                       verbose)
                except Exception as e:
                    print(f"\033[91m工具调用失败：{e}\033[0m")
                    return {}

                # 添加工具调用结果到历史记录
                params['history'].append({
                    'role': 'user',
                    'content': call_response
                })

            # 如果有最终响应
            if parsed_response['final_response']:
                return {
                    'thought': response['thought'],
                    'output': parsed_response['final_response']
                }

        return {}

    def init_tools_map(self, tools: Sequence[BaseTool]):
        """
        初始化工具映射。

        :param tools: 工具列表。
        """
        for tool in tools:
            self.tools_map[tool.name] = tool

    def generate_tools_declaration(self, tools: Sequence[BaseTool]):
        """
        生成工具声明。

        :param tools: 工具列表。
        """
        self.tool_descriptions = render_text_description_and_args(tools)
        self.tool_names = ", ".join([t.name for t in tools])

    @staticmethod
    def divide_thought(response: str) -> dict:
        """
        分割思考过程。

        :param response: 响应内容。
        :return: 分割后的思考过程和输出。
        """
        if '</think>' in response:
            thought, output = response.split('</think>', 1)
            return {
                'thought': thought.lstrip(),
                'output': output.lstrip()
            }
        else:
            return {
                'thought': '',
                'output': response.lstrip()
            }

    def parse_response(self, response: str) -> dict:
        """
        解析响应。

        :param response: 响应内容。
        :return: 解析后的结果。
        """
        result = {'tool_call': None, 'final_response': None}

        # 工具解析
        tool_call_pattern = r"\[TOOL\]\s*tool:\s*'(\w+)'\s*parameters:\s*({.*?})\s*\[/TOOL\]"
        tool_call_match = re.search(tool_call_pattern, response, re.DOTALL)
        if tool_call_match:
            tool_name = tool_call_match.group(1)
            try:
                parameters = eval(tool_call_match.group(2))
            except Exception as e:
                print(f"\033[91m工具参数解析失败：{e}\033[0m")
                return result

            result['tool_call'] = {'tool_name': tool_name, 'parameters': parameters}

        # 响应输出解析
        final_response_pattern = r"\[END\]\s*(.*?)\s*\[/END\]"
        final_response_match = re.search(final_response_pattern, response, re.DOTALL)
        if final_response_match:
            result['final_response'] = final_response_match.group(1).strip()

        return result

    def call_function(self, function_name: str, parameters: dict, verbose: bool = False) -> str:
        """
        调用工具函数。

        :param function_name: 工具名称。
        :param parameters: 参数。
        :return: 调用结果。
        """

        if verbose:
            print('\033[94mFunction Call：', function_name, '\033[0m')

        if function_name not in self.tools_map:
            print(f"\033[91m工具 {function_name} 不存在\033[0m")
            return ""

        try:
            call_result = self.tools_map[function_name].invoke(parameters)
        except Exception as e:
            print(f"\033[91m工具 {function_name} 调用失败：{e}\033[0m")
            return ""

        if verbose:
            print('\033[94mFunction Response：\n', str(call_result), '\033[0m')

        return str(call_result)