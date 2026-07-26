import os
from openai import OpenAI

API_KEY = "sk-13568098fcf54bf9a4e81236f17a1500"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

system_prompt = """你是一个专业的Python程序员。用户会给你一个文件的完整代码作为上下文，然后描述他的需求。
你需要根据需求修改代码，并输出修改后的完整文件内容。
规则：
1. 如果用户说"修改"、"改"、"修复"，你需要找到对应部分修改，保持其他代码不变
2. 如果用户说"新增"、"添加"、"加一个"，你需要在合适位置插入新代码
3. 如果用户说"重写"、"重构"，你可以重新组织整个文件
4. 只输出修改后的完整代码，不要解释，不要多余的话
5. 代码要完整、安全、有注释"""

messages = [{"role": "system", "content": system_prompt}]


def select_file():
    """让用户选择要操作的文件，返回文件路径和文件内容"""
    while True:
        file_path = input("\n请选择文件：").strip()

        if file_path.lower() == 'quit':
            return None, None

        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            print(f"已读取文件：{file_path}")
            print(f"文件大小：{len(code)} 字符")
            return file_path, code
        else:
            print(f"文件不存在：{file_path}，请重新输入。")


# ========== 主程序 ==========
print("=" * 50)
print("AI 代码助手已就绪！")
print("输入 'quit' 退出")
print("=" * 50)

current_file = None
current_code = ""

# 第一步：选择文件
current_file, current_code = select_file()

if current_file is None:
    print("再见！")
    exit()

print("现在你可以描述修改需求了。")

# 第二步：进入对话循环
while True:
    user_input = input("\nlooveashe request：")

    if user_input.lower() == 'quit':
        print("再见！")
        break

    # 构建上下文消息
    context = f"当前文件：{current_file}\n\n完整代码：\n```python\n{current_code}\n```\n\n用户需求：{user_input}"
    messages.append({"role": "user", "content": context})

    print("\nAI 正在生成代码...")
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        temperature=0.1,
        max_tokens=4000,
        stream=True
    )

    full_response = ""
    print("\n生成的代码：\n")
    print("-" * 40)
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    print("\n" + "-" * 40)

    # 提取代码块
    new_code = full_response
    if "```python" in new_code:
        new_code = new_code.split("```python")[1].split("```")[0].strip()
    elif "```" in new_code:
        new_code = new_code.split("```")[1].split("```")[0].strip()

    # 询问是否写入文件
    choice = input(f"\n要写入 {current_file} 吗？(y/n)：")
    if choice.lower() == 'y':
        with open(current_file, 'w', encoding='utf-8') as f:
            f.write(new_code)
        print(f"已写入 {current_file}")
        current_code = new_code
        messages.append({"role": "assistant", "content": full_response})
    else:
        print("已取消写入。")
