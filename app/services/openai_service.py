from app.utils.pinecone_utils import query_similar, client
from app.utils.shared_state import all_info, chat_history

def analyze_health_data(user_data):
    """
    Analyze user health data and calculate a health score
    """
    if not user_data or not isinstance(user_data, dict) or "data" not in user_data:
        return {
            "score": 0,
            "analysis": "No health data available for analysis"
        }

    # Extract available data and check what's valid
    available_data = {}
    data_sources = []
    
    if "data" in user_data:
        # Check emotions data
        if ("emotions" in user_data["data"] and 
            isinstance(user_data["data"]["emotions"], dict) and
            "error" not in user_data["data"]["emotions"]):
            available_data["emotions"] = user_data["data"]["emotions"]
            data_sources.append("emotions")

        # Check watch/wearable data
        if ("watch_data" in user_data["data"] and 
            isinstance(user_data["data"]["watch_data"], dict) and
            "error" not in user_data["data"]["watch_data"]):
            available_data["watch_data"] = user_data["data"]["watch_data"]
            data_sources.append("watch_data")

        # Check medical reports
        if ("medical_reports" in user_data["data"] and 
            isinstance(user_data["data"]["medical_reports"], dict) and
            "error" not in user_data["data"]["medical_reports"]):
            available_data["medical_reports"] = user_data["data"]["medical_reports"]
            data_sources.append("medical_reports")

    if not available_data:
        return {
            "score": 0,
            "analysis": "No valid health data available for analysis",
            "available_sources": []
        }

    prompt = f"""
Analyze the following health data and provide:
1. A health score from 0-100 (where 100 is perfect health)
2. A brief explanation of areas that need improvement or attention

Available Health Data:
{available_data}

Consider only the following available data sources: {data_sources}
Note: Some data might not be available, focus analysis only on the data provided.

Format your response exactly like this:
SCORE: [number between 0-100]
ANALYSIS: [1-2 sentences explaining the score and key areas for improvement]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise health data analyzer."},
            {"role": "user", "content": prompt}
        ]
    )

    result_text = response.choices[0].message.content
    
    # Parse the response
    score = 0
    analysis = ""
    
    for line in result_text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = int(line.replace("SCORE:", "").strip())
            except ValueError:
                score = 0
        elif line.startswith("ANALYSIS:"):
            analysis = line.replace("ANALYSIS:", "").strip()
    
    return {
        "score": score,
        "analysis": analysis,
        "available_sources": data_sources
    }

def summarize_chat_history(chat_history_data):
    """
    Generate a summary and title for the chat history
    """
    if not chat_history_data:
        return {
            "title": "No chat history",
            "summary": "No conversations found"
        }

    # Format chat history for the prompt
    chat_text = "\n".join([
        f"User: {exchange['query']}\nAI: {exchange['response']}"
        for exchange in chat_history_data
    ])

    prompt = f"""
Analyze the following conversation between a user and an AI wellness assistant:

{chat_text}

Please provide: A brief title (5-8 words) that captures the main topic or theme

Format your response exactly like this:
TITLE: [Your title here]
SUMMARY: [Your summary here]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise summarization assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    summary_text = response.choices[0].message.content
    
    # Parse the response
    title = ""
    summary = ""
    
    for line in summary_text.split("\n"):
        if line.startswith("TITLE:"):
            title = line.replace("TITLE:", "").strip()
        elif line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()
    
    return {
        "title": title,
        "summary": summary
    }

def chat_with_memory(query: str, user_id: str):
    # Retrieve similar docs
    results = query_similar(query)
    
    context = all_info.get(user_id, {})
    user_chat_history = chat_history.get(user_id, [])
    
    # Format chat history
    chat_history_text = "\n".join([
        f"User: {exchange['query']}\nAI: {exchange['response']}"
        for exchange in user_chat_history[-5:]  # Get last 5 conversations
    ])

    # Combine context + chat history + user query
    prompt = f"""
You are a friendly, evidence-aware wellness coach. You will be given:
1) CONTEXT: short anonymized data snippets about the user (sleep, vitals, nutrition, labs, previous important info), if available.
2) query: the user's query.

Your task:
- Use the context to provide practical and concise (3-6 points) recommendations if user want.
- No need to make any recommendations until user ask for it.
- If the context contains relevant personal info, prioritize it in your advice.
- If context does not cover the topic, provide safe, general, evidence-based guidance on health, wellness, nutrition, exercise, sleep, mental health, or lifestyle.
- Always keep your advice friendly, concise, and actionable.
- Highlight which data points from context you used (by timestamp and source), if any.
- If there is potential clinical risk (e.g., dangerously abnormal labs, chest pain), instruct the user to seek medical attention.

CONTEXT (User Data):
{context}

RECENT CHAT HISTORY:
{chat_history_text}

CURRENT QUERY:
{query}

RESPONSE:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful wellness assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content