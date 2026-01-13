from app.utils.pinecone_utils import query_similar, client
from app.utils.shared_state import all_info, chat_history

def analyze_health_data(user_data):
    """
    Analyze user health data and calculate a health score.

    Scoring rules implemented:
    - If no valid data sources -> score 0.
    - If `medical_reports` is present -> split weight ~33% each across emotions, watch_data, medical_reports.
    - If `medical_reports` is absent -> emotions and watch_data split 50% each (or single source gets 100%).

    The function asks the model to provide per-source scores (0-100), then combines them using the above weights.
    """
    if not user_data or not isinstance(user_data, dict) or "data" not in user_data:
        return {
            "score": 0,
            "analysis": "No health data available for analysis"
        }

    # Extract available data and check what's valid (use most recent entry when available)
    available_data = {}
    data_sources = []

    def _most_recent_entry(obj):
        # Return the most recent record from a list/dict if possible.
        try:
            import dateutil.parser
        except Exception:
            dateutil = None
        def _date_of(item):
            if not isinstance(item, dict):
                return None
            for k in ["date", "timestamp", "created_at", "collected_at", "time"]:
                if k in item:
                    v = item[k]
                    if isinstance(v, (int, float)):
                        try:
                            return float(v)
                        except Exception:
                            continue
                    if isinstance(v, str) and 'dateutil' in globals():
                        try:
                            return dateutil.parser.parse(v)
                        except Exception:
                            continue
                    # try parse ISO without dateutil
                    try:
                        from datetime import datetime
                        return datetime.fromisoformat(v)
                    except Exception:
                        continue
            return None

        if isinstance(obj, list) and obj:
            # prefer element with the latest timestamp if available
            best = None
            best_ts = None
            for el in obj:
                ts = _date_of(el)
                if ts is not None:
                    if best_ts is None or ts >= best_ts:
                        best = el
                        best_ts = ts
            if best is not None:
                return best
            # fallback to last element
            return obj[-1]

        if isinstance(obj, dict):
            # if dict contains lists of records, return most recent from those lists
            for k in ["data", "entries", "records", "samples", "measurements", "results", "workoutData", "sleepData"]:
                v = obj.get(k)
                if isinstance(v, list) and v:
                    return _most_recent_entry(v)
            return obj

        return obj

    if "data" in user_data:
        # Emotions (accept dict or list; pick most recent)
        if "emotions" in user_data["data"]:
            em = user_data["data"]["emotions"]
            if isinstance(em, dict) and "error" not in em:
                available_data["emotions"] = _most_recent_entry(em)
                data_sources.append("emotions")
            elif isinstance(em, list) and len(em) > 0:
                available_data["emotions"] = _most_recent_entry(em)
                data_sources.append("emotions")

        # Watch/wearable data (accept dict or list; pick most recent)
        if "watch_data" in user_data["data"]:
            wd = user_data["data"]["watch_data"]
            if isinstance(wd, dict) and "error" not in wd:
                available_data["watch_data"] = _most_recent_entry(wd)
                data_sources.append("watch_data")
            elif isinstance(wd, list) and len(wd) > 0:
                available_data["watch_data"] = _most_recent_entry(wd)
                data_sources.append("watch_data")

        # Medical reports (may be dict or list; treat empty list/dict as absent; pick most recent)
        if "medical_reports" in user_data["data"]:
            mr = user_data["data"]["medical_reports"]
            has_medical = False
            # dict: must be non-empty and not contain an "error"
            if isinstance(mr, dict):
                if mr and "error" not in mr:
                    has_medical = True
            # list: must contain at least one item
            elif isinstance(mr, list):
                if len(mr) > 0:
                    has_medical = True
            # if valid, include most recent
            if has_medical:
                available_data["medical_reports"] = _most_recent_entry(mr)
                data_sources.append("medical_reports")

    if not available_data:
        return {
            "score": 0,
            "analysis": "No valid health data available for analysis",
            "available_sources": []
        }

    # Determine weighting depending on presence of medical reports
    weight_map = {}
    if "medical_reports" in data_sources:
        # All three contribute equally (≈33% each)
        for s in ["emotions", "watch_data"]:
            if s in data_sources:
                weight_map[s] = 1.0 / 2.0
    else:
        present = [s for s in ["emotions", "watch_data"] if s in data_sources]
        if len(present) == 2:
            weight_map = {present[0]: 0.5, present[1]: 0.5}
        elif len(present) == 1:
            weight_map = {present[0]: 1.0}

    # Build prompt asking for per-source scores
    score_lines = []
    if "emotions" in weight_map:
        score_lines.append("EMOTIONS_SCORE: [0-100]")
    if "watch_data" in weight_map:
        score_lines.append("WATCH_SCORE: [0-100]")
    # if "medical_reports" in weight_map:
    #     score_lines.append("MEDICAL_REPORTS_SCORE: [0-100]")

    prompt = f"""
For each of the following user data sources, provide a SCORE from 0-100 (where 100 means excellent for that data source) and a brief note.

Available Health Data:
{available_data}

Only consider these sources and the data provided. Return in the exact format below (only include lines for sources that are present):
{chr(10).join(score_lines)}
ANALYSIS: [1-2 sentences summarizing overall health and key areas for improvement]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise health data analyzer. For each provided data source, return a numeric score 0-100 and a short analysis line."},
            {"role": "user", "content": prompt}
        ]
    )

    result_text = response.choices[0].message.content

    # Parse per-source scores
    per_source_scores = {}
    analysis = ""
    fallback_score = None

    for line in result_text.splitlines():
        line = line.strip()
        if line.startswith("EMOTIONS_SCORE:"):
            try:
                per_source_scores["emotions"] = int(line.replace("EMOTIONS_SCORE:", "").strip())
            except ValueError:
                per_source_scores["emotions"] = 0
        elif line.startswith("WATCH_SCORE:"):
            try:
                per_source_scores["watch_data"] = int(line.replace("WATCH_SCORE:", "").strip())
            except ValueError:
                per_source_scores["watch_data"] = 0
        # elif line.startswith("MEDICAL_REPORTS_SCORE:"):
        #     try:
        #         per_source_scores["medical_reports"] = int(line.replace("MEDICAL_REPORTS_SCORE:", "").strip())
        #     except ValueError:
        #         per_source_scores["medical_reports"] = 0
        elif line.startswith("SCORE:") and fallback_score is None:
            # Backwards-compatible single SCORE field
            try:
                fallback_score = int(line.replace("SCORE:", "").strip())
            except ValueError:
                fallback_score = 0
        elif line.startswith("ANALYSIS:"):
            analysis = line.replace("ANALYSIS:", "").strip()

    # Compute final weighted score
    final_score = 0
    if per_source_scores:
        total = 0.0
        for src, wt in weight_map.items():
            src_score = per_source_scores.get(src, 0)
            total += src_score * wt
        final_score = int(round(total))
    elif fallback_score is not None:
        final_score = int(fallback_score)
    else:
        final_score = 0

    return {
        "score": final_score,
        "analysis": analysis,
        "available_sources": data_sources,
        "per_source_scores": per_source_scores
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
- If the user's query reports current symptoms or an acute health concern, **start your response with exactly one short clarifying question** (1 sentence) to gather important details (e.g., "When did the fever start?", "Do you have any other symptoms?").
- After that clarifying question, provide practical and concise recommendations (3-6 bullet points) labelled "Recommendations:". Recommendations should be evidence-aware, actionable, and concise.
- If the user did not report symptoms or there is no acute issue, you may respond directly with helpful guidance without the initial clarifying question.
- Always highlight any context data you used (by timestamp and source), if applicable.
- If there is potential clinical risk (e.g., dangerously abnormal labs, chest pain, persistent high fever), **explicitly instruct the user to seek immediate medical attention**.
- Use a friendly, empathetic tone and keep guidance brief.

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