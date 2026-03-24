import os
from dotenv import load_dotenv
from agent import ReActAgent

def main():
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("Error: Missing API Keys in .env file.")
        return

    # 實例化「單一」的通用型 Agent
    agent = ReActAgent()
    
    tasks = [
        "What fraction of Japan's population is Taiwan's population as of 2025?",
        "Compare the main display specs of iPhone 15 and Samsung S24.",
        "Who is the CEO of the startup 'Morphic' AI search?"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n\n{'='*20} Starting Task {i} {'='*20}")
        agent.execute(task)

if __name__ == "__main__":
    main()