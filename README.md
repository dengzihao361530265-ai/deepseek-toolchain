# deepseek-toolchain

#### 介绍
允许本地部署DeekSeek R1等推理模型通过Function Call的方式调用外部工具。

#### 使用说明
1. 如果使用官方API，建议使用openai_test.py进行测试，本地部署API建议使用langchain_test.py进行测试
1. 设置模型的base_url与api_key
2. 设置tavily的api_key
3. 运行对应的测试脚本