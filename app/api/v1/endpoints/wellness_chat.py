from fastapi import APIRouter, Query, Depends, HTTPException
from app.services.openai_service import chat_with_memory, summarize_chat_history, analyze_health_data
from app.schemas.user_data import get_user
from app.utils.shared_state import all_info, chat_history, prompt_counters
import requests
import os
import datetime

router = APIRouter()

@router.get("/health-score")
async def get_health_score(user_id: str = Query(..., description="User ID"), 
                         access_token: str = Query(..., description="User access token")):
    """
    Calculate and return a health score based on user's data
    """
    global all_info
    # Construct the URLs for all endpoints
    base_url = os.getenv("BASE_URL")
    emotion_url = f"{base_url}/api/v1/emotions/user/{user_id}"
    watch_url = f"{base_url}/api/v1/health-data/user/{user_id}/debug"
    medical_url = f"{base_url}/api/v1/medical-reports"
    meal_url = f"{base_url}/api/v1/meals"
    
    # Set up headers with the access token
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    # Initialize response dictionary
    result = {
        "status": "success",
        "data": {}
    }

    try:
        # Get emotions data
        emotions_response = requests.get(emotion_url, headers=headers)
        if emotions_response.status_code == 200:
            result["data"]["emotions"] = emotions_response.json()
        else:
            result["data"]["emotions"] = {
                "error": f"Failed with status {emotions_response.status_code}",
                "details": emotions_response.text
            }

        # Get watch data
        watch_response = requests.get(watch_url, headers=headers)
        if watch_response.status_code == 200:
            result["data"]["watch_data"] = watch_response.json()
        else:
            result["data"]["watch_data"] = {
                "error": f"Failed with status {watch_response.status_code}",
                "details": watch_response.text
            }

        # Get medical reports
        medical_response = requests.get(medical_url, headers=headers)
        if medical_response.status_code == 200:
            result["data"]["medical_reports"] = medical_response.json()
        else:
            result["data"]["medical_reports"] = {
                "error": f"Failed with status {medical_response.status_code}",
                "details": medical_response.text
            }

        # Get meal data
        meal_response = requests.get(meal_url, headers=headers)
        if meal_response.status_code == 200:
            result["data"]["meals"] = meal_response.json()
        else:
            result["data"]["meals"] = {
                "error": f"Failed with status {meal_response.status_code}",
                "details": meal_response.text
            }
        
        # Store the result in all_info
        all_info[user_id] = {
            "data": result
        }
        
        user_data = result
    except Exception as e:
        error_response = {
            "status": "error",
            "message": "Failed to fetch data",
            "details": str(e)
        }
        all_info[user_id] = {
            "data": error_response
        }
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch health data: {str(e)}"
        )

    # Analyze the health data
    analysis_result = analyze_health_data(user_data)

    # Post health score and analysis to DB
    db_url = f"{base_url}/api/v1/chat/health-score/{user_id}"
    db_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    db_payload = {
        "health_score": analysis_result["score"],
        "analysis": analysis_result["analysis"],
        "date": datetime.datetime.now().isoformat()
    }
    try:
        requests.post(db_url, headers=db_headers, json=db_payload)
    except Exception as e:
        print(f"Failed to post health score to DB: {str(e)}")

    return {
        "user_id": user_id,
        "health_score": analysis_result["score"],
        "analysis": analysis_result["analysis"],
    }



############################################################################################

@router.post("/get_info")
async def start_new_chat(user_id: str = Query(..., description="User ID"),
                         access_token: str = Query(..., description="User access token"),
                        room_id: str = Query(..., description="Room ID")):
    """
    Initialize a new chat session with stored user data
    """
    if user_id not in all_info:
        raise HTTPException(
            status_code=404,
            detail="No data found for this user. Please call health-score endpoint first to fetch data."
        )
    
    base_url = os.getenv("BASE_URL")
    wellness_score_url = f"{base_url}/api/v1/chat/health-score/{user_id}"
    
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    # Initialize response dictionary
    score_result = {
        "status": "success",
        "data": {}
    }

    try:
        wellness_score_response = requests.get(wellness_score_url, headers=headers)
        if wellness_score_response.status_code == 200:
            score_result["data"]["wellness_score"] = wellness_score_response.json()
        else:
            score_result["data"]["wellness_score"] = {
                "error": f"Failed with status {wellness_score_response.status_code}",
                "details": wellness_score_response.text
            }
        
        
    except Exception as e:
        score_result["data"]["wellness_score"] = {
            "error": "Exception occurred while fetching wellness score",
            "details": str(e)
        }

    # Update all_info with the new room_id
    user_info = all_info[user_id]
    user_info["room_id"] = room_id

    # Merge the fetched wellness_score into stored data so it is returned
    user_info.setdefault("data", {})
    if "wellness_score" in score_result.get("data", {}):
        user_info["data"]["wellness_score"] = score_result["data"]["wellness_score"]

    all_info[user_id] = user_info
    
    


    
    return {
        "status": "success",
        "message": "Chat session initialized",
        "user_id": user_id,
        "room_id": room_id,
        "data": user_info["data"]
    }



@router.post("/chat")
async def wellness_chat(
    query: str = Query(..., description="User question or message"),
    user_id: str = Query(..., description="User ID"),
    access_token: str = Query(..., description="User access token"),
    max_prompt: int = Query(10, description="Maximum number of prompts allowed")
):
    """
    Chat endpoint for user wellness questions.
    """
    global chat_history
    
    # Get room_id from all_info
    if user_id not in all_info:
        raise HTTPException(max_prompt,
            status_code=400,
            detail="Please call start_new_chat endpoint first to initialize the session"
        )
    
    room_id = all_info[user_id]["room_id"]
    
    # Initialize chat history for user if it doesn't exist
    if user_id not in chat_history:
        chat_history[user_id] = []
    
    # Get AI response
    response = chat_with_memory(query, user_id)
    
    # Increment prompt counter for the room
    if room_id not in prompt_counters:
        prompt_counters[room_id] = 0
    prompt_counters[room_id] += 1
    
    # Create chat data
    chat_data = {
        "query": query,
        "response": response,
        "promptUsed": str(prompt_counters[room_id]),  # Incremented prompt counter. global variable from shared_state.py
        "user_id": user_id,
        "room_id": room_id,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Store the exchange in chat history
    chat_history[user_id].append(chat_data)

    # Determine and persist the room's aiTitle (first message) if not already set
    try:
        user_info = all_info.get(user_id, {})
        room_titles = user_info.get("room_titles", {})
        # If no title for this room, the first message becomes the aiTitle
        if room_id not in room_titles:
            room_titles[room_id] = query
            user_info["room_titles"] = room_titles
            all_info[user_id] = user_info
        ai_title = room_titles.get(room_id, "")
    except Exception:
        ai_title = ""
    
    # # Save to database
    # try:
    #     # Make POST request to DB service
    #     db_url = f"{os.getenv('DB_SERVICE_URL')}/api/v1/chats"
    #     requests.post(db_url, json=chat_data)
    # except Exception as e:
    #     print(f"Failed to save chat to database: {str(e)}")
    
    # Also push the latest exchange to wellness backend ai-response endpoint
    try:
        db_url = f"{os.getenv('BASE_URL')}"

        ai_response_url = f"{db_url}/api/v1/chat/ai-response/{room_id}"
        # max_prompt = int(os.getenv("MAX_PROMPT", "10"))
        ai_payload = {
            "maxPrompt": max_prompt,
            "promptUsed": prompt_counters.get(room_id, 0),
            "aiTitle": ai_title,
            "chat": [
                {
                    "content": query,
                    "responseData": response
                }
            ]
        }
        ai_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        requests.post(ai_response_url, json=ai_payload, headers=ai_headers)
    except Exception as e:
        print(f"Failed to post ai-response to wellness backend: {e}")
    
    return {
        "query": query,
        "response": response,
        "promptUsed": str(prompt_counters[room_id]),
        "aiTitle": ai_title
    }

@router.get("/chat-history")
async def get_chat_history(
    user_id: str = Query(..., description="User ID"),
    room_id: str = Query(..., description="Room ID")
):
    """
    Get chat history for a specific user in a specific room
    """
    user_history = chat_history.get(user_id, [])
    # Filter history by room_id
    room_history = [chat for chat in user_history if chat.get("room_id") == room_id]
    
    return {
        "user_id": user_id,
        "room_id": room_id,
        "history": room_history
    }

# @router.get("/chat-summary")
# async def get_chat_summary(
#     user_id: str = Query(..., description="User ID"),
#     room_id: str = Query(..., description="Room ID"),
#     limit: int = Query(5, description="Number of recent messages to summarize")
# ):
#     """
#     Get a summary of recent chat history for a specific user in a specific room
#     """
#     user_history = chat_history.get(user_id, [])
    
#     # Filter history by room_id first
#     room_history = [chat for chat in user_history if chat.get("room_id") == room_id]
    
#     # Get the most recent conversations based on limit
#     recent_history = room_history[-limit:] if room_history else []
    
#     # Generate summary and title
#     summary_data = summarize_chat_history(recent_history)
    
#     return {
#         "user_id": user_id,
#         "room_id": room_id,
#         "conversation_count": len(recent_history),
#         "title": summary_data["title"],
#         "summary": summary_data["summary"],
#         "recent_messages": recent_history
#     }


