import os
import re
from openai import OpenAI
from tools import search_tool

class ReActAgent:
    def __init__(self):
        # 建立 OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Few-Shot Prompting
        # 「失敗並反思(Reflection)」的例子。
        # 這個範例教導模型遇到找不到答案時，要懂得「換關鍵字」而不是放棄。
        self.system_prompt = """You are a smart, reasoning AI agent. You solve complex queries by executing a Thought -> Action -> Observation loop.

AVAILABLE ACTIONS:
1. Search[query]: Search the internet for information.
2. Finish[answer]: Provide the final answer to the user.

RULES:
- You MUST format your output EXACTLY as:
Thought: <your reasoning, planning, or reflection>
Action: <ToolName>[<argument>]
- Only output ONE Thought and ONE Action at a time.
- DO NOT generate the "Observation:". The system will provide it.
- If a search fails or doesn't return the exact information, REFLECT in your next Thought and try a different search query (Self-Correction).
- Break down complex questions into multiple search steps before finishing.

=== FEW-SHOT EXAMPLE ===
User: Who is the wife of the founder of Apple?
Thought: The user is asking for the wife of Apple's founder. First, I need to find out who founded Apple.
Action: Search[founder of Apple]
Observation: Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.
Thought: Steve Jobs is the primary founder. Now I need to find out who his wife is.
Action: Search[Steve Jobs wife]
Observation: No relevant results found.
Thought: The search didn't return a clear answer. Since he passed away, I should search for his widow.
Action: Search[Steve Jobs widow]
Observation: Laurene Powell Jobs is the widow of Steve Jobs.
Thought: I have found the answer.
Action: Finish[Laurene Powell Jobs]
=== END OF EXAMPLE ===
"""

    def execute(self, query: str):
        # 每次執行新任務時重置對話歷史 (messages)，確保不同問題之間不會互相干擾
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": query})
        
        print(f"\n[User Query]: {query}")
        print("-" * 50)
        
        iteration = 0
        max_steps = 7  # 防止 Agent 無限發散
        
        # The Loop Mechanism
        while iteration < max_steps:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2, 
                # Stop Logic 防幻覺機制
                # 強制 LLM 在準備輸出 Observation 之前就「停筆」，不准它自己幻想出假的搜尋結果！
                stop=["Observation:", "Observation"] 
            )
            
            output = response.choices[0].message.content.strip()
            print(output) # 即時印出 Thought 與 Action
            
            # 將 LLM 的思考與動作存回對話歷史
            messages.append({"role": "assistant", "content": output})
            
            # 使用正則表達式解析出 Action 的動作與參數
            action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", output, re.DOTALL)
            
            if not action_match:
                obs = "Observation: Format error. You must use 'Action: Search[query]' or 'Action: Finish[answer]'."
                print(obs)
                messages.append({"role": "user", "content": obs})
                iteration += 1
                continue
                
            tool_name = action_match.group(1).strip()
            tool_args = action_match.group(2).strip()
            
            if tool_name == "Finish":
                print(f"\n[Final Answer]: {tool_args}")
                return tool_args
                
            elif tool_name == "Search":
                print(f"[System: 正在搜尋 '{tool_args}'...]")
                search_result = search_tool(tool_args)
                obs_text = f"Observation: {search_result}"
                
                print(f"Observation: {search_result[:150]}...\n")
                
                # 將真實的搜尋結果以 User 的身分餵回給 LLM 的 context window
                messages.append({"role": "user", "content": obs_text})
                
            else:
                obs = f"Observation: Tool '{tool_name}' not recognized."
                print(obs)
                messages.append({"role": "user", "content": obs})
                
            iteration += 1
            
        print("\n[Warning]: Max iterations reached.")
        return "Failed to find the answer within the limit."