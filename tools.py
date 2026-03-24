import os
from tavily import TavilyClient

def search_tool(query: str) -> str:
    """使用 Tavily API 進行網路搜尋，並回傳格式化結果字串"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set. Please check your .env file."
    
    try:
        client = TavilyClient(api_key=api_key)
        
        # 執行搜尋。限制 max_results=3 是為了不要給 LLM 太多廢話，節省 Token
        response = client.search(query=query, search_depth="basic", max_results=3)
        
        results = []
        # 將搜尋結果提取出來，排版成乾淨的字串
        for res in response.get('results', []):
            results.append(f"- {res['title']}: {res['content']}")
            
        # 如果搜尋成功但沒找到東西，主動提示 LLM 進行反思
        if not results:
            return "No relevant results found. Please REFLECT and try a different keyword."
            
        return "\n".join(results)
        
    except Exception as e:
        # 捕捉 API 錯誤，不讓程式崩潰，而是讓錯誤變成 Observation 回傳給 Agent
        return f"Search API error occurred: {str(e)}. Please REFLECT and try again."

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("測試搜尋...\n")
    test_result = search_tool("What is Agentic AI?")
    print(test_result)